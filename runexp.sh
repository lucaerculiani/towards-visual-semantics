set -e


datadir=dataset
dataurl="https://ndownloader.figshare.com/files/28243401"
dataname="embedded.zip"
tmpdataname="embedded.zip.partial"
if [ -f "$dataname" ] ; then
    echo "dataset '$dataname' already present, skipping download"
else 
    echo "downloading dataset to '$datadir' folder"
    wget -O $tmpdataname $dataurl
    mv $tmpdataname $dataname
fi
if [ -d "$datadir" ] ; then
    echo " '$datadir' folder already present, skipping unpaking"
else 
    echo "unpackiong '$dataname' to '$datadir' folder"
    mkdir -p dataset
    unzip -q $dataname   -d $datadir
    mv ${datadir}/discriminative_frames/* ${datadir}/
    rmdir ${datadir}/discriminative_frames/
fi

PYTHONPATH=. python  scripts/fs2desc.py --ambiguity  dataset/ambiguity descriptor.json
nexp=$(ls -1 inputs | wc -l) 
echo "loading $nexp experiments" 
counter=1 
mkdir -p results
 for i in inputs/* ; do 
     o=results/$(basename $i).npy.lz4
     if [ -f "$o" ] ; then
        echo [${counter}/${nexp}]: ${i} already done, skipping 
    else 
        echo -n [${counter}/${nexp}]": "
        PYTHONPATH=. python scripts/json_train.py  --results ${o} ${i} 
    fi 
    : $((counter++)) 
 done 
 echo creating figures... 

tmpf=$(mktemp -d)
PYTHONPATH=. python  scripts/plot_ambiguous.py  results/random.json.npy.lz4 results/random_s{3,2,1}0.json.npy.lz4 --labels "1.0,0.3,0.2,0.1" --o ${tmpf}/ >/dev/null

echo "done!"

mkdir -p outputs
mv  ${tmpf}/{diff_acc,genus_acc}.png outputs/
rm -r ${tmpf}

