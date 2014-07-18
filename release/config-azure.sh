
AZ_LOCATION='West US'
AZ_SSH_KEY=$HOME/.ssh/azure
AZ_SSH_CERT=$HOME/.ssh/azure.pem
AZ_IMAGE=b39f27a8b8c64d52b05eac6a62ebad85__Ubuntu-14_04-LTS-amd64-server-20140618.1-en-us-30GB
AZ_SUBNET=Subnet-1
TAG=testing

if [ -z "$(which azure)" ]; then
    echo "Couldn't find azure in PATH"
    exit 1
fi

if [ -z "$(azure account list | grep true)" ]; then
    echo "Default azure account not set"
    exit 1
fi

account=$(azure account list | grep true | awk '{ print $2 }')

if which md5 > /dev/null 2>&1; then
  hsh=$(md5 -q -s $account)
else
  hsh=$(echo -n "$account" | md5sum)
fi
hsh=${hsh:0:7}

STG_ACCOUNT=kube$hsh

AZ_VNET=kube-$hsh
AZ_CS=kube-$hsh

CONTAINER=kube-$TAG

FULL_URL="https://${STG_ACCOUNT}.blob.core.windows.net/$CONTAINER/master-release.tgz"
