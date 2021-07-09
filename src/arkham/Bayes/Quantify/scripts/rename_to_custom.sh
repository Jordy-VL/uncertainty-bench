original=$1
res=$(jq .command $1/params.json | grep -oP "\-n \K[\w_-]*")
echo $res
mv "$original" /mnt/lerna/models/"$res"