CONFIG=$1
GPUS=$2
WORKDIR=$3

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
echo $PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py   --seed 0 $CONFIG --launcher pytorch ${@:3} --work-dir $WORKDIR
