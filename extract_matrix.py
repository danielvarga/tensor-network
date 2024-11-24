#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('..')

import numpy as np
import torch
from src import models, data, lens, functional
from src.utils import experiment_utils
from baukit import Menu, show


# In[3]:


# In[4]:


dataset = data.load_dataset()

relation_names = [r.name for r in dataset.relations]
relation_options = Menu(choices = relation_names, value = relation_names)
# show(relation_options) # !caution: tested in a juputer-notebook. baukit visualizations are not supported in vscode.


# In[5]:


filename = sys.argv[1]
relation_name = filename.replace('_', ' ')
relation = dataset.filter(relation_names=[relation_name])[0]
print(f"{relation.name} -- {len(relation.samples)} samples")
print("------------------------------------------------------")

experiment_utils.set_seed(12345) # set seed to a constant value for sampling consistency
train, test = relation.split(5)
print("\n".join([sample.__str__() for sample in train.samples]))


# In[6]:


################### hparams ###################
layer = 5
beta = 2.5
###############################################



device = "cuda:0"
print("loading model")
mt = models.load_model("gptj", device=device, fp16=True)
print(f"dtype: {mt.model.dtype}, device: {mt.model.device}, memory: {mt.model.get_memory_footprint()}")


# In[7]:


from src.operators import JacobianIclMeanEstimator

estimator = JacobianIclMeanEstimator(
    mt = mt, 
    h_layer = layer,
    beta = beta
)
operator = estimator(
    relation.set(
        samples=train.samples, 
    )
)


# # Checking $faithfulness$

# In[8]:


test = functional.filter_relation_samples_based_on_provided_fewshots(
    mt=mt, test_relation=test, prompt_template=operator.prompt_template, batch_size=4
)


# In[9]:


sample = test.samples[0]
print(sample)
operator(subject = sample.subject).predictions


# In[10]:


hs_and_zs = functional.compute_hs_and_zs(
    mt = mt,
    prompt_template = operator.prompt_template,
    subjects = [sample.subject],
    h_layer= operator.h_layer,
)
h = hs_and_zs.h_by_subj[sample.subject]


# ## Approximating LM computation $F$ as an affine transformation
# 
# ### $$ F(\mathbf{s}, c_r) \approx \beta \, W_r \mathbf{s} + b_r $$

# In[11]:

print("operator.weight.shape", operator.weight.shape)
print("operator.bias.shape", operator.bias.shape)
matrix = operator.beta * operator.weight
matrix = matrix.detach().cpu().numpy()
bias = operator.bias.detach().cpu().numpy()
extended_matrix = np.hstack((matrix, bias.T))
np.save(filename, extended_matrix)

exit()

z = operator.beta * (operator.weight @ h) + operator.bias

lens.logit_lens(
    mt = mt,
    h = z,
    get_proba = True
)


# In[12]:


correct = 0
wrong = 0
for sample in test.samples:
    predictions = operator(subject = sample.subject).predictions
    known_flag = functional.is_nontrivial_prefix(
        prediction=predictions[0].token, target=sample.object
    )
    print(f"{sample.subject=}, {sample.object=}, ", end="")
    print(f'predicted="{functional.format_whitespace(predictions[0].token)}", (p={predictions[0].prob}), known=({functional.get_tick_marker(known_flag)})')
    
    correct += known_flag
    wrong += not known_flag
    
faithfulness = correct/(correct + wrong)

print("------------------------------------------------------------")
print(f"Faithfulness (@1) = {faithfulness}")
print("------------------------------------------------------------")


# # $causality$

# In[13]:


################### hparams ###################
rank = 100
###############################################


# In[14]:


experiment_utils.set_seed(12345) # set seed to a constant value for sampling consistency
test_targets = functional.random_edit_targets(test.samples)


# ## setup

# In[15]:


source = test.samples[0]
target = test_targets[source]

f"Changing the mapping ({source}) to ({source.subject} -> {target.object})"


# ### Calculate $\Delta \mathbf{s}$ such that $\mathbf{s} + \Delta \mathbf{s} \approx \mathbf{s}'$
# 
# <p align="center">
#     <img align="center" src="causality-crop.png" style="width:80%;"/>
# </p>
# 
# Under the relation $r =\, $*plays the instrument*, and given the subject $s =\, $*Miles Davis*, the model will predict $o =\, $*trumpet* **(a)**; and given the subject $s' =\, $*Cat Stevens*, the model will now predict $o' =\, $*guiter* **(b)**. 
# 
# If the computation from $\mathbf{s}$ to $\mathbf{o}$ is well-approximated by $operator$ parameterized by $W_r$ and $b_r$ **(c)**, then $\Delta{\mathbf{s}}$ **(d)** should tell us the direction of change from $\mathbf{s}$ to $\mathbf{s}'$. Thus, $\tilde{\mathbf{s}}=\mathbf{s}+\Delta\mathbf{s}$ would be an approximation of $\mathbf{s}'$ and patching $\tilde{\mathbf{s}}$ in place of $\mathbf{s}$ should change the prediction to $o'$ = *guitar* 

# In[16]:


def get_delta_s(
    operator, 
    source_subject, 
    target_subject,
    rank = 100,
    fix_latent_norm = None, # if set, will fix the norms of z_source and z_target
):
    w_p_inv = functional.low_rank_pinv(
        matrix = operator.weight,
        rank=rank,
    )
    hs_and_zs = functional.compute_hs_and_zs(
        mt = mt,
        prompt_template = operator.prompt_template,
        subjects = [source_subject, target_subject],
        h_layer= operator.h_layer,
        z_layer=-1,
    )

    z_source = hs_and_zs.z_by_subj[source_subject]
    z_target = hs_and_zs.z_by_subj[target_subject]
    
    z_source *= fix_latent_norm / z_source.norm() if fix_latent_norm is not None else 1.0
    z_target *= z_source.norm() / z_target.norm() if fix_latent_norm is not None else 1.0

    delta_s = w_p_inv @  (z_target.squeeze() - z_source.squeeze())

    return delta_s, hs_and_zs

delta_s, hs_and_zs = get_delta_s(
    operator = operator,
    source_subject = source.subject,
    target_subject = target.subject,
    rank = rank
)


# In[17]:


import baukit

def get_intervention(h, int_layer, subj_idx):
    def edit_output(output, layer):
        if(layer != int_layer):
            return output
        functional.untuple(output)[:, subj_idx] = h 
        return output
    return edit_output

prompt = operator.prompt_template.format(source.subject)

h_index, inputs = functional.find_subject_token_index(
    mt=mt,
    prompt=prompt,
    subject=source.subject,
)

h_layer, z_layer = models.determine_layer_paths(model = mt, layers = [layer, -1])

with baukit.TraceDict(
    mt.model, layers = [h_layer, z_layer],
    edit_output=get_intervention(
#         h = hs_and_zs.h_by_subj[source.subject],         # let the computation proceed as usual
        h = hs_and_zs.h_by_subj[source.subject] + delta_s, # replace s with s + delta_s
        int_layer = h_layer, 
        subj_idx = h_index
    )
) as traces:
    outputs = mt.model(
        input_ids = inputs.input_ids,
        attention_mask = inputs.attention_mask,
    )

lens.interpret_logits(
    mt = mt, 
    logits = outputs.logits[0][-1], 
    get_proba=True
)


# ## Measuring causality

# In[18]:


from src.editors import LowRankPInvEditor

svd = torch.svd(operator.weight.float())
editor = LowRankPInvEditor(
    lre=operator,
    rank=rank,
    svd=svd,
)


# In[19]:


# precomputing latents to speed things up
hs_and_zs = functional.compute_hs_and_zs(
    mt = mt,
    prompt_template = operator.prompt_template,
    subjects = [sample.subject for sample in test.samples],
    h_layer= operator.h_layer,
    z_layer=-1,
    batch_size = 2
)

success = 0
fails = 0

for sample in test.samples:
    target = test_targets.get(sample)
    assert target is not None
    edit_result = editor(
        subject = sample.subject,
        target = target.subject
    )
    
    success_flag = functional.is_nontrivial_prefix(
        prediction=edit_result.predicted_tokens[0].token, target=target.object
    )
    
    print(f"Mapping {sample.subject} -> {target.object} | edit result={edit_result.predicted_tokens[0]} | success=({functional.get_tick_marker(success_flag)})")
    
    success += success_flag
    fails += not success_flag
    
causality = success / (success + fails)

print("------------------------------------------------------------")
print(f"Causality (@1) = {causality}")
print("------------------------------------------------------------")


# In[ ]:





# In[ ]:




