# python eval.py --alg "rfsac" --sigma "0.0" --max_timesteps "100000" --rand_feat_num "256" --learn_rf "False" & 

# python eval.py --alg "rfsac" --sigma "0.0" --max_timesteps "100000" --rand_feat_num "256" --learn_rf "True"

#python eval.py --alg "rfsac" --sigma "0.0" --max_timesteps "100000" --rand_feat_num "512" --learn_rf "False" &

# python eval.py --alg "rfsac" --sigma "0.0" --max_timesteps "100000" --rand_feat_num "512" --learn_rf "True" & 

# python eval.py --alg "rfsac" --sigma "0.0" --max_timesteps "100000" --rand_feat_num "1024" --learn_rf "False" &

# python eval.py --alg "rfsac" --sigma "0.0" --max_timesteps "100000" --rand_feat_num "1024" --learn_rf "True"


# rfsac with q loss shake

python eval.py --alg "rfsac" --sigma "0.0" --max_timesteps "100000" --rand_feat_num "512" --learn_rf "False"

# sac on quadrotor

#python eval_drones.py --alg "sac" --sigma "0.0" --max_timesteps "400000" --rand_feat_num "512" --learn_rf "False"
