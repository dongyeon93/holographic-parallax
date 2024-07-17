for str in DragonBunny
do
for f in 1
do
for v in 9 
do
if [ \( "$f" -eq 16 -a "$v" -eq 7 \) -o \( "$f" -eq 16 -a "$v" -eq 9 \) -o \( "$f" -eq 24 -a "$v" -eq 7 \) -o \( "$f" -eq 24 -a "$v" -eq 9 \) ]
then 
    for ch in 0 1 2
    do 
        python main.py \
        --channel $ch \
        --data_path ./data/lf_dataset/$str \
        --out_path ./results/4d/$str/$f/$v/ch_$ch \
        --ang_res $v,$v \
        --num_frames $f \
        --eyepiece 0.04 \
        --physical_iris True \
        --target_type 4d \
        --qt_method gumbel \
        --mem_eff True  
    done
else
    for ch in 0 1 2
    do 
        python main.py \
        --channel $ch \
        --data_path ./data/lf_dataset/$str \
        --out_path ./results/4d/$str/$f/$v/ch_$ch \
        --ang_res $v,$v \
        --num_frames $f \
        --eyepiece 0.04 \
        --physical_iris True \
        --target_type 4d \
        --qt_method gumbel \
        --mem_eff False 
    done
fi
done
done
done