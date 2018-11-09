set -x
set -e

#GPU_NMS=$1
NET=$1

LOG="experiments/logs/faster_rcnn/faster_rcnn_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/test_net.py \
    --cfg experiments/cfgs/e2e_faster_rcnn_${NET}.yaml \
    --dataset coco_daofeishi_test \
    --load_ckpt experiments/output/faster_rcnn/resnet-50/e2e_faster_rcnn_resnet-50/Nov07-15-41-46_iim321_step/ckpt/model_step9999.pth