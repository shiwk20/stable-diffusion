PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

export PYTHONPATH=./:$PYTHONPATH && CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT tools/ffhq256/sample_ffhq256.py\
    --outdir outputs/ffhq256_ddim10_0 --skip_grid --steps 10 --eta 0 --max_img 1000 --batch_size 32