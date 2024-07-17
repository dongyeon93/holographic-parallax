for str in DragonBunny
do
for f in 1
do
    for ch in 0 1 2
    do 
        python main.py \
        --channel $ch \
        --data_path ./data/rgbd_dataset/$str \
        --out_path ./results/2.5d/$str/$f/ch_$ch \
        --num_frames $f \
        --num_planes 9 \
        --eyepiece 0.04 \
        --physical_iris True \
        --target_type 2.5d \
        --qt_method gumbel \
        --mem_eff False \
        --is_perspective False
    done
done
done