#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
if [ "$1" == "" ]; then
  echo "Usage $0: <config-file>"
  exit 1
fi
source ${DIR}/../util.sh
source ${1}

GCLOUD="gcloud"
GSUTIL="gsutil"

${GCLOUD} config set project ${PROJECT}
echo "Checking for files in gcs"
${GSUTIL} -q stat gs://${PROJECT}/curl-loader 
if [ $? -ne 0 ]; then
  echo "Please build curl-loader and copy it to gs://${PROJECT}/curl_loader"
  exit 1
fi

echo -e "${LOADER_CONF}" > /tmp/loader.conf
${GSUTIL} cp /tmp/loader.conf gs://${PROJECT}/loader.conf

echo "
ulimit -n 100000
echo 1 > /proc/sys/net/ipv4/tcp_tw_recycle
echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse

while [ \${MY_IP}x == 'x' ]; do
  export MY_IP=\$(ip addr show eth0 | awk '/inet / {split(\$2, ip, \"/\"); print ip[1]}')
  echo .
  sleep 1
done
echo My IP: \${MY_IP}
gsutil -m cp gs://${PROJECT}/curl-loader .
gsutil -m cp gs://${PROJECT}/loader.conf .
chmod 755 curl-loader
sed -i s/@IP@/\${MY_IP}/ loader.conf
yes | ./curl-loader -r -f loader.conf" > /tmp/load-init.sh

Header "Creating load instances..."
for i in `seq 0 $((${LOAD_NUM_REPLICAS} - 1))`; do
  echo "Creating instance ${i}"

  ${GCLOUD} compute instances create -q load-${i} \
      --zone=${ZONE} \
      --machine-type n1-standard-2 \
      --image ubuntu-14-10 \
      --tags load \
      --metadata-from-file startup-script=/tmp/load-init.sh &
done
wait

for i in `seq 0 $((${LOAD_NUM_REPLICAS} - 1))`; do
  echo "Waiting for instance ${i}..."
  WaitForStatus instances load-${i} ${ZONE} RUNNING &
done
wait
