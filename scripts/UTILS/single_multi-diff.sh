method=('n,n' 'y,y')
#method=('n,y' 'y,n') # Not possible



for i in "${method[@]}"; do
    log_file="./scripts/result/diff_log/${i}.log"
    mkdir -p ./scripts/result/diff_log/
    echo "Running: python3 ./scripts/sh_diff_single-multi.py $i > $log_file"
    python3 ./scripts/sh_diff_single-multi.py "$i" #> "$log_file"
done