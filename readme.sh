git clone git@github.com:danielvarga/tensor-network.git
cd tensor-network/
git clone git@github.com:evandez/relations.git


cd relations/
cd demo/
mamba install conda-forge::dataclasses-json
pip install git+https://github.com/davidbau/baukit
jupyter nbconvert --to script demo.ipynb
CUDA_VISIBLE_DEVICES=1 python demo.py 
# -> that does not work perfectly, but it is the basis for ./extract_matrix.py
# which must be started from the demo directory:

CUDA_VISIBLE_DEVICES=1 python ../../extract_matrix.py superhero_person

cat ../../relationship_names | tr ' ' '_' | head -3 | while read r ; do echo "=========" ; echo $r ; echo "=========" ; CUDA_VISIBLE_DEVICES=1 python ../../extract_matrix.py $r ; done
