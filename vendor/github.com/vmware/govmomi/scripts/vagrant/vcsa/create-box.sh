#!/bin/sh

set -e

if [ "$(uname -s)" == "Darwin" ]; then
  PATH="/Applications/VMware Fusion.app/Contents/Library:$PATH"
  PATH="/Applications/VMware Fusion.app/Contents/Library/VMware OVF Tool:$PATH"
fi

ovf="$1"

if [ -z "$ovf" ]; then
  ovf="./VMware-vCenter-Server-Appliance-5.5.0.10300-2000350_OVA10.ova"
fi

# check for greadlink and gmktemp
readlink=$(type -p greadlink readlink | head -1)
mktemp=$(type -p gmktemp mktemp | head -1)

dir=$($readlink -nf $(dirname $0))
tmp=$($mktemp -d)
trap "rm -rf $tmp" EXIT

cd $tmp

echo "Converting ovf..."
ovftool \
  --noSSLVerify \
  --acceptAllEulas \
  --overwrite \
  --powerOffTarget \
  $ovf vcsa.vmx

echo "Starting vm..."
vmrun start vcsa.vmx nogui

echo "Waiting for vm ip..."
ip=$(vmrun getGuestIPAddress vcsa.vmx -wait)

echo "Configuring vm for use with vagrant..."
vmrun -gu root -gp vmware CopyFileFromHostToGuest vcsa.vmx \
      $dir/vagrant.sh /tmp/vagrant.sh

vmrun -gu root -gp vmware runProgramInGuest vcsa.vmx \
      /bin/sh -e /tmp/vagrant.sh

vmrun -gu root -gp vmware deleteFileInGuest vcsa.vmx \
      /tmp/vagrant.sh

echo "Configuring vCenter Server Appliance..."

ssh_opts="-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -T"

ssh ${ssh_opts} -i ~/.vagrant.d/insecure_private_key vagrant@$ip <<EOS
echo "Accepting EULA ..."
sudo /usr/sbin/vpxd_servicecfg eula accept

echo "Configuring Embedded DB ..."
sudo /usr/sbin/vpxd_servicecfg db write embedded

echo "Configuring SSO..."
sudo /usr/sbin/vpxd_servicecfg sso write embedded

echo "Starting VCSA ..."
sudo /usr/sbin/vpxd_servicecfg service start
EOS

echo "Stopping vm..."
vmrun stop vcsa.vmx

rm -f vmware.log

sed -i -e 's/"bridged"/"nat"/' vcsa.vmx

echo '{"provider":"vmware_desktop"}' > ./metadata.json

cd $dir

tar -C $tmp -cvzf vcsa.box .

vagrant box add --name vcsa vcsa.box
