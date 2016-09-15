#!/bin/bash
#
# This script is useful for creating a devstack environment to run gophercloud
# acceptance tests on.
#
# This can be considered a "legacy" devstack environment since it uses
# Keystone v2 and LBaaS v1.
#
# To run, simply execute this script within a virtual machine.
#
# The following OpenStack versions are installed:
# * OpenStack Mitaka
# * Keystone v2
# * Glance v1 and v2
# * Nova v2 and v2.1
# * Cinder v1 and v2
# * Trove v1
# * Swift v1
# * Neutron v2
# * Neutron LBaaS v1.0
# * Neutron FWaaS v2.0
# * Manila v2
#
# Go 1.6 is also installed.

set -e

cd
sudo apt-get update
sudo apt-get install -y git make mercurial

sudo wget -O /usr/local/bin/gimme https://raw.githubusercontent.com/travis-ci/gimme/master/gimme
sudo chmod +x /usr/local/bin/gimme
gimme 1.6 >> .bashrc

mkdir ~/go
eval "$(/usr/local/bin/gimme 1.6)"
echo 'export GOPATH=$HOME/go' >> .bashrc
export GOPATH=$HOME/go
source .bashrc

go get golang.org/x/crypto/ssh
go get github.com/gophercloud/gophercloud

git clone https://git.openstack.org/openstack-dev/devstack -b stable/mitaka
cd devstack
cat >local.conf <<EOF
[[local|localrc]]
# OpenStack version
OPENSTACK_VERSION="mitaka"

# devstack password
DEVSTACK_PASSWORD="password"

# Configure passwords and the Swift Hash
MYSQL_PASSWORD=\$DEVSTACK_PASSWORD
RABBIT_PASSWORD=\$DEVSTACK_PASSWORD
SERVICE_TOKEN=\$DEVSTACK_PASSWORD
ADMIN_PASSWORD=\$DEVSTACK_PASSWORD
SERVICE_PASSWORD=\$DEVSTACK_PASSWORD
SWIFT_HASH=\$DEVSTACK_PASSWORD

# Configure the stable OpenStack branches used by DevStack
# For stable branches see
# http://git.openstack.org/cgit/openstack-dev/devstack/refs/
CINDER_BRANCH=stable/\$OPENSTACK_VERSION
CEILOMETER_BRANCH=stable/\$OPENSTACK_VERSION
GLANCE_BRANCH=stable/\$OPENSTACK_VERSION
HEAT_BRANCH=stable/\$OPENSTACK_VERSION
HORIZON_BRANCH=stable/\$OPENSTACK_VERSION
KEYSTONE_BRANCH=stable/\$OPENSTACK_VERSION
NEUTRON_BRANCH=stable/\$OPENSTACK_VERSION
NOVA_BRANCH=stable/\$OPENSTACK_VERSION
SWIFT_BRANCH=stable/\$OPENSTACK_VERSION
ZAQAR_BRANCH=stable/\$OPENSTACK_VERSION

# Enable Swift
enable_service s-proxy
enable_service s-object
enable_service s-container
enable_service s-account

# Disable Nova Network and enable Neutron
disable_service n-net
enable_service q-svc
enable_service q-agt
enable_service q-dhcp
enable_service q-l3
enable_service q-meta
#enable_service q-flavors

# Disable Neutron metering
disable_service q-metering

# Enable LBaaS V1
enable_service q-lbaas

# Enable FWaaS
enable_service q-fwaas

# Enable LBaaS v2
#enable_plugin neutron-lbaas https://git.openstack.org/openstack/neutron-lbaas stable/\$OPENSTACK_VERSION
#enable_plugin octavia https://git.openstack.org/openstack/octavia stable/\$OPENSTACK_VERSION
#enable_service q-lbaasv2
#enable_service octavia
#enable_service o-cw
#enable_service o-hk
#enable_service o-hm
#enable_service o-api

# Enable Trove
enable_plugin trove git://git.openstack.org/openstack/trove.git stable/\$OPENSTACK_VERSION
enable_service trove,tr-api,tr-tmgr,tr-cond

# Disable Temptest
disable_service tempest

# Disable Horizon
disable_service horizon

# Disable Keystone v2
#ENABLE_IDENTITY_V2=False

# Enable SSL/tls
#enable_service tls-proxy
#USE_SSL=True

# Enable Ceilometer
#enable_service ceilometer-acompute
#enable_service ceilometer-acentral
#enable_service ceilometer-anotification
#enable_service ceilometer-collector
#enable_service ceilometer-alarm-evaluator
#enable_service ceilometer-alarm-notifier
#enable_service ceilometer-api

# Enable Zaqar
#enable_plugin zaqar https://github.com/openstack/zaqar
#enable_service zaqar-server

# Enable Manila
enable_plugin manila https://github.com/openstack/manila

# Automatically download and register a VM image that Heat can launch
# For more information on Heat and DevStack see
# http://docs.openstack.org/developer/heat/getting_started/on_devstack.html
#IMAGE_URLS+=",http://cloud.fedoraproject.org/fedora-20.x86_64.qcow2"
#IMAGE_URLS+=",https://cloud-images.ubuntu.com/trusty/current/trusty-server-cloudimg-amd64-disk1.img"

# Logging
LOGDAYS=1
LOGFILE=/opt/stack/logs/stack.sh.log
LOGDIR=/opt/stack/logs
EOF
./stack.sh

# Prep the testing environment by creating the required testing resources and environment variables
source openrc admin
wget http://download.cirros-cloud.net/0.3.4/cirros-0.3.4-x86_64-disk.img
glance image-create --name CirrOS --disk-format qcow2 --container-format bare < cirros-0.3.4-x86_64-disk.img
nova flavor-create m1.acctest 99 512 5 1 --ephemeral 10
nova flavor-create m1.resize 98 512 6 1 --ephemeral 10
_NETWORK_ID=$(nova net-list | grep private | awk -F\| '{print $2}' | tr -d ' ')
_EXTGW_ID=$(nova net-list | grep public | awk -F\| '{print $2}' | tr -d ' ')
_IMAGE_ID=$(nova image-list | grep CirrOS | awk -F\| '{print $2}' | tr -d ' ' | head -1)
echo export OS_IMAGE_NAME="cirros-0.3.4-x86_64-uec" >> openrc
echo export OS_IMAGE_ID="$_IMAGE_ID" >> openrc
echo export OS_NETWORK_ID=$_NETWORK_ID >> openrc
echo export OS_EXTGW_ID=$_EXTGW_ID >> openrc
echo export OS_POOL_NAME="public" >> openrc
echo export OS_FLAVOR_ID=99 >> openrc
echo export OS_FLAVOR_ID_RESIZE=98 >> openrc
source openrc demo
