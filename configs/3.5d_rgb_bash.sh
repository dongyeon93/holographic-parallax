for str in DragonBunny
do
for f in 1
do
    for ch in 0 1 2
    do 
        python main.py \
        --channel $ch \
        --data_path ./data/lf_fs_dataset/$str/ch_$ch \
        --out_path ./results/3.5d/$str/$f/ch_$ch \
        --num_frames $f \
        --num_planes 9 \
        --eyepiece 0.04 \
        --physical_iris True \
        --target_type 3d \
        --qt_method gumbel \
        --mem_eff False 
    done
done
done