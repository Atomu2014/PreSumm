# train ext
python train.py -task ext -mode train -model_path MODEL_PATH -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert_xsum

# train abs
python train.py -task abs -mode train -sep_optim true -use_bert_emb true -model_path ../models/ -log_file ../logs/abs_bert_xsum

# train extabs
python train.py -task abs -mode train -sep_optim true -use_bert_emb true -model_path ../models/ -log_file ../logs/abs_bert_xsum  -load_from_extractive EXT_CKPT

# eval cnndm
python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -log_file ../logs/val_abs_bert_xsum -model_path ../models/xsum_bertextabs -sep_optim true -use_interval true -min_length 50 -max_length 200 -alpha 0.95 -result_path ../logs/abs_bert_xsum

# eval xsum
python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -log_file ../logs/val_abs_bert_xsum -model_path ../models/xsum_bertextabs -sep_optim true -use_interval true -min_length 20 -max_length 100 -alpha 0.9 -result_path ../logs/abs_bert_xsum
