cat ../../relationship_names | tr ' ' '_' | while read r ; do echo "=========" ; echo $r ; echo "=========" ; CUDA_VISIBLE_DEVICES=1 python ../../extract_matrix.py $r ; done
