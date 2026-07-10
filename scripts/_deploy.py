import paramiko, time, base64

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westb.seetacloud.com", port=53373, username="root", password="Yr6SOovahWDH")

script = """#!/bin/bash
set -e
LOG=/outputs/act_aloha.log
OUT=/outputs/act_aloha
echo "[auto] Waiting for training to finish..."
while pgrep -f lerobot-train > /dev/null 2>&1; do sleep 120; done
echo "[auto] Training done! Starting evaluation..."
export HF_ENDPOINT=https://hf-mirror.com
for ckpt in 020000 040000 060000 080000 100000; do
  name=$((10#${ckpt} / 1000))K
  echo "[auto] Evaluating ${name}..."
  xvfb-run -a /opt/miniconda/envs/lerobot/bin/lerobot-eval \
    --policy.path=${OUT}/checkpoints/${ckpt}/pretrained_model \
    --env.type=aloha --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=50 --eval.use_async_envs=false \
    --output_dir=${OUT}/eval/${name} \
    > ${OUT}/eval_${ckpt}.log 2>&1
done
grep "step:" ${LOG} > ${OUT}/training_metrics.txt
echo "[auto] All done!"
"""

encoded = base64.b64encode(script.encode()).decode()
_, out, _ = c.exec_command(f"echo {encoded} | base64 -d > /outputs/auto_finish.sh")
time.sleep(0.5)
_, out, _ = c.exec_command("chmod +x /outputs/auto_finish.sh && wc -l /outputs/auto_finish.sh")
print("Script lines:", out.read().decode().strip())
_, out, _ = c.exec_command("head -3 /outputs/auto_finish.sh")
print(out.read().decode().strip())
_, out, _ = c.exec_command("nohup /outputs/auto_finish.sh > /outputs/auto_finish.log 2>&1 & echo PID=$!")
print("Started:", out.read().decode().strip())
time.sleep(1)
_, out, _ = c.exec_command("ps aux | grep auto_finish | grep -v grep")
print("Running:", out.read().decode().strip()[:200])
c.close()
