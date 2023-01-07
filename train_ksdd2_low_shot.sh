for k in $(seq 0 109)
do
    env PYTHONPATH=src python run_patchcore.py --config 'ksdd2_low_shot.yml' --label_times $k
done