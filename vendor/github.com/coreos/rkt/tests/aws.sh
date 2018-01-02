#!/bin/bash

set -e

SCRIPTPATH=$(dirname "$0")
cd $SCRIPTPATH
SCRIPTPATH=$PWD

KEY_PAIR_NAME=rkt-testing-${USER}
SECURITY_GROUP=rkt-testing-${USER}-security-group
REGION=us-west-1

## First time only
if [ "$1" = "setup" ] ; then
  MYIP=$(curl --silent http://checkip.amazonaws.com/)

  aws --region $REGION ec2 create-key-pair --key-name $KEY_PAIR_NAME --query 'KeyMaterial' --output text > ${KEY_PAIR_NAME}.pem
  chmod 0600 ${KEY_PAIR_NAME}.pem
  aws --region $REGION ec2 create-security-group --group-name $SECURITY_GROUP --description "Security group for rkt testing"
  aws --region $REGION ec2 authorize-security-group-ingress --group-name $SECURITY_GROUP --protocol tcp --port 22 --cidr $MYIP/32
  exit 0
fi

DISTRO=$1

test -f cloudinit/${DISTRO}.cloudinit
CLOUDINIT=$PWD/cloudinit/${DISTRO}.cloudinit

if [ "$DISTRO" = "fedora-24" ] ; then
  # https://getfedora.org/en/cloud/download/
  # Search on aws --region $REGION or look at
  # https://apps.fedoraproject.org/datagrepper/raw?category=fedimg
  # Sources: https://github.com/fedora-infra/fedimg/blob/develop/bin/list-the-amis.py

  # Fedora-Cloud-Base-24-20160507.n.0.x86_64-us-west-1-HVM-standard-0
  AMI=ami-8b4c35eb
  AWS_USER=fedora
elif [ "$DISTRO" = "fedora-25" ] ; then
  # Fedora-Cloud-Base-25-20161220.0.x86_64-us-west-1-HVM-standard-0
  AMI=ami-c70d5ca7
  AWS_USER=fedora
elif [ "$DISTRO" = "fedora-rawhide" ] ; then
  # Fedora-Cloud-Base-rawhide-20160129.x86_64-us-west-1-HVM-standard-0
  AMI=ami-a18dfac1
  AWS_USER=fedora
elif [ "$DISTRO" = "ubuntu-1604" ] ; then
  # https://cloud-images.ubuntu.com/locator/ec2/
  # ubuntu/images/hvm-ssd/ubuntu-xenial-16.04-amd64-server-20160627
  AMI=ami-b20542d2
  AWS_USER=ubuntu
elif [ "$DISTRO" = "debian-testing" ] ; then
  # https://wiki.debian.org/Cloud/AmazonEC2Image/Jessie
  # Debian 8.6+1
  AMI=ami-db6c39bb
  AWS_USER=admin
fi

test -n "$AMI"
test -n "$AWS_USER"
test -f "${KEY_PAIR_NAME}.pem"

INSTANCE_ID=$(aws --region $REGION ec2 run-instances \
	--image-id $AMI \
	--count 1 \
	--key-name $KEY_PAIR_NAME \
	--security-groups $SECURITY_GROUP \
	--instance-type m4.large \
	--instance-initiated-shutdown-behavior terminate \
	--user-data file://$CLOUDINIT \
	--output text \
	--query 'Instances[*].InstanceId' \
	)
echo INSTANCE_ID=$INSTANCE_ID

aws --region $REGION ec2 create-tags \
	--resources ${INSTANCE_ID} \
	--tags \
	Key=Name,Value=rkt-tst-${DISTRO} \
	Key=BaseAmi,Value=${AMI} \
	Key=User,Value=${AWS_USER}


while state=$(aws --region $REGION ec2 describe-instances \
	--instance-ids $INSTANCE_ID \
	--output text \
	--query 'Reservations[*].Instances[*].State.Name' \
	); test "$state" = "pending"; do
  sleep 1; echo -n '.'
done; echo " $state"

AWS_IP=$(aws --region $REGION ec2 describe-instances \
	--instance-ids $INSTANCE_ID \
	--output text \
	--query 'Reservations[*].Instances[*].PublicIpAddress' \
	)
echo AWS_IP=$AWS_IP

echo "Waiting for the instance to boot..."
sleep 60
echo "Waiting for the instance to be initialized..."
echo "To check the logs:"
echo ssh -i ${SCRIPTPATH}/${KEY_PAIR_NAME}.pem ${AWS_USER}@${AWS_IP} tail -f /var/log/cloud-init-output.log
while ! ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=20 -o ConnectTimeout=10 -o ConnectionAttempts=60 -i ${SCRIPTPATH}/${KEY_PAIR_NAME}.pem ${AWS_USER}@${AWS_IP} stat /var/lib/cloud/instances/$INSTANCE_ID/boot-finished >/dev/null 2>&1
do
  echo -n '.'
  sleep 30
done


NAME=rkt-ci-jenkins-$(date +"%Y-%m-%d")-$DISTRO
AMI_ID=$(aws --region $REGION ec2 create-image --instance-id $INSTANCE_ID --name $NAME --output text)

echo -e "\nWaiting for the AMI to be avaliable..."
while ! aws --region $REGION ec2 describe-images --image-id $AMI_ID | grep -q available
do
  echo -n '.'
  sleep 30
done

echo -e "\nRemoving instance..."

aws --region $REGION ec2 terminate-instances --instance-ids $INSTANCE_ID --output text

echo "${DISTRO} AMI available: $AMI_ID (Region $REGION)"
