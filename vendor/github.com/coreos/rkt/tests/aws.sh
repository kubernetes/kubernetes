#!/bin/bash

set -e

SCRIPTPATH=$(dirname "$0")
cd $SCRIPTPATH
SCRIPTPATH=$PWD

KEY_PAIR_NAME=rkt-testing-${USER}
SECURITY_GROUP=rkt-testing-${USER}-security-group

## First time only
if [ "$1" = "setup" ] ; then
  MYIP=$(curl --silent http://checkip.amazonaws.com/)

  aws ec2 create-key-pair --key-name $KEY_PAIR_NAME --query 'KeyMaterial' --output text > ${KEY_PAIR_NAME}.pem
  chmod 0600 ${KEY_PAIR_NAME}.pem
  aws ec2 create-security-group --group-name $SECURITY_GROUP --description "Security group for rkt testing"
  aws ec2 authorize-security-group-ingress --group-name $SECURITY_GROUP --protocol tcp --port 22 --cidr $MYIP/32
  exit 0
fi

DISTRO=$1
GIT_URL=${2-https://github.com/coreos/rkt.git}
GIT_BRANCH=${3-master}
FLAVOR=${4-coreos}

if [ "$DISTRO" = "all" ] ; then
  gnome-terminal \
	--tab --command="$SCRIPTPATH/aws.sh fedora-22      $GIT_URL $GIT_BRANCH $FLAVOR" \
	--tab --command="$SCRIPTPATH/aws.sh fedora-23      $GIT_URL $GIT_BRANCH $FLAVOR" \
	--tab --command="$SCRIPTPATH/aws.sh fedora-rawhide $GIT_URL $GIT_BRANCH $FLAVOR" \
	--tab --command="$SCRIPTPATH/aws.sh ubuntu-1604    $GIT_URL $GIT_BRANCH $FLAVOR" \
	--tab --command="$SCRIPTPATH/aws.sh ubuntu-1510    $GIT_URL $GIT_BRANCH $FLAVOR" \
	--tab --command="$SCRIPTPATH/aws.sh debian         $GIT_URL $GIT_BRANCH $FLAVOR" \
	--tab --command="$SCRIPTPATH/aws.sh centos         $GIT_URL $GIT_BRANCH $FLAVOR"

  exit 0
fi

test -f cloudinit/${DISTRO}.cloudinit
CLOUDINIT_IN=$PWD/cloudinit/${DISTRO}.cloudinit

if [ "$DISTRO" = "fedora-22" ] ; then
  # https://getfedora.org/en/cloud/download/
  # Search on AWS or look at
  # https://apps.fedoraproject.org/datagrepper/raw?category=fedimg
  # Sources: https://github.com/fedora-infra/fedimg/blob/develop/bin/list-the-amis.py

  # Fedora-Cloud-Base-22-20160218.x86_64-eu-central-1-HVM-standard-0
  AMI=ami-7a1b0116
  AWS_USER=fedora

  # Workarounds
  DISABLE_SELINUX=true
elif [ "$DISTRO" = "fedora-23" ] ; then
  # Fedora-Cloud-Base-23-20160323.x86_64-eu-central-1-HVM-standard-0
  AMI=ami-d59670ba
  AWS_USER=fedora

  # Workarounds
  DISABLE_SELINUX=true
elif [ "$DISTRO" = "fedora-rawhide" ] ; then
  # Fedora-Cloud-Base-Rawhide-20160321.0.x86_64-eu-central-1-HVM-standard-0
  AMI=ami-69967006
  AWS_USER=fedora

  # Workarounds
  # systemd in stage1 does not have the fixes for SELinux yet
  FLAVOR=host
  DISABLE_OVERLAY=true
elif [ "$DISTRO" = "ubuntu-1604" ] ; then
  # https://cloud-images.ubuntu.com/locator/ec2/
  # ubuntu/images-milestone/hvm/ubuntu-xenial-alpha2-amd64-server-20160125
  AMI=ami-b4a5b9d8
  AWS_USER=ubuntu
elif [ "$DISTRO" = "ubuntu-1510" ] ; then
  # https://cloud-images.ubuntu.com/locator/ec2/
  # ubuntu/images/hvm/ubuntu-wily-15.10-amd64-server-20160123
  AMI=ami-e9869f85
  AWS_USER=ubuntu
elif [ "$DISTRO" = "debian" ] ; then
  # https://wiki.debian.org/Cloud/AmazonEC2Image/Jessie
  # Debian 8.1
  AMI=ami-02b78e1f
  AWS_USER=admin
elif [ "$DISTRO" = "centos" ] ; then
  # Needs to subscribe first, see:
  # https://wiki.centos.org/Cloud/AWS
  # CentOS-7 x86_64 HVM
  AMI=ami-e68f82fb
  AWS_USER=centos
fi

test -n "$AMI"
test -n "$AWS_USER"
test -f "${KEY_PAIR_NAME}.pem"

CLOUDINIT=$(mktemp --tmpdir rkt-cloudinit.XXXXXXXXXX)
sed -e "s#@GIT_URL@#${GIT_URL}#g" \
    -e "s#@GIT_BRANCH@#${GIT_BRANCH}#g" \
    -e "s#@FLAVOR@#${FLAVOR}#g" \
    -e "s#@DISABLE_SELINUX@#${DISABLE_SELINUX}#g" \
    -e "s#@DISABLE_OVERLAY@#${DISABLE_OVERLAY}#g" \
    < $CLOUDINIT_IN >> $CLOUDINIT

INSTANCE_ID=$(aws ec2 run-instances \
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

aws ec2 create-tags --resources $INSTANCE_ID \
	--tags \
	Key=Name,Value=rkt-tst-${DISTRO}-${GIT_BRANCH}-${FLAVOR} \
	Key=Distro,Value=${DISTRO} \
	Key=Repo,Value=${GIT_URL} \
	Key=Branch,Value=${GIT_BRANCH} \
	Key=Flavor,Value=${FLAVOR} \
	Key=User,Value=${USER}

while state=$(aws ec2 describe-instances \
	--instance-ids $INSTANCE_ID \
	--output text \
	--query 'Reservations[*].Instances[*].State.Name' \
	); test "$state" = "pending"; do
  sleep 1; echo -n '.'
done; echo " $state"

AWS_IP=$(aws ec2 describe-instances \
	--instance-ids $INSTANCE_ID \
	--output text \
	--query 'Reservations[*].Instances[*].PublicIpAddress' \
	)
echo AWS_IP=$AWS_IP

rm -f $CLOUDINIT

sleep 5
aws ec2 get-console-output --instance-id $INSTANCE_ID --output text |
  perl -ne 'print if /BEGIN SSH .* FINGERPRINTS/../END SSH .* FINGERPRINTS/'

echo
echo "Check the logs with:"
echo tail -n 5000 -f /var/tmp/rkt-test.log

repeat="Y"
while [ "$repeat" = "Y" ] ; do
  ssh -o ServerAliveInterval=20 -o ConnectTimeout=10 -o ConnectionAttempts=60 -i ${SCRIPTPATH}/${KEY_PAIR_NAME}.pem ${AWS_USER}@${AWS_IP}

  echo -n "Reconnect? (Y/N)"
  read -n1 Input
  echo
  case $Input in
    [Nn]):
    repeat="N"
    ;;
  esac
done
