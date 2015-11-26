#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# exit on any error
set -e

#setup kubelet config
mkdir -p "/var/lib/kubelet"
(umask 077;
cat > "/var/lib/kubelet/kubeconfig" << EOF
apiVersion: v1
kind: Config
users:
- name: kubelet
user:
  token: ${KUBELET_TOKEN}
clusters:
- name: local
cluster:
  insecure-skip-tls-verify: true
contexts:
- context:
  cluster: local
  user: kubelet
name: service-account-context
current-context: service-account-context
EOF
)

#setup proxy config
mkdir -p "/var/lib/kube-proxy/"
# Make a kubeconfig file with the token.
# TODO(etune): put apiserver certs into secret too, and reference from authfile,
# so that "Insecure" is not needed.
(umask 077;
cat > "/var/lib/kube-proxy/kubeconfig" << EOF
apiVersion: v1
kind: Config
users:
- name: kube-proxy
user:
  token: ${KUBE_PROXY_TOKEN}
clusters:
- name: local
cluster:
   insecure-skip-tls-verify: true
contexts:
- context:
  cluster: local
  user: kube-proxy
name: service-account-context
current-context: service-account-context
EOF
)



# Set the host name explicitly
# See: https://github.com/mitchellh/vagrant/issues/2430
hostnamectl set-hostname ${MINION_NAME}

if [[ "$(grep 'VERSION_ID' /etc/os-release)" =~ ^VERSION_ID=21 ]]; then
  # Workaround to vagrant inability to guess interface naming sequence
  # Tell system to abandon the new naming scheme and use eth* instead
  rm -f /etc/sysconfig/network-scripts/ifcfg-enp0s3

  # Disable network interface being managed by Network Manager (needed for Fedora 21+)
  NETWORK_CONF_PATH=/etc/sysconfig/network-scripts/
  if_to_edit=$( find ${NETWORK_CONF_PATH}ifcfg-* | xargs grep -l VAGRANT-BEGIN )
  for if_conf in ${if_to_edit}; do
    grep -q ^NM_CONTROLLED= ${if_conf} || echo 'NM_CONTROLLED=no' >> ${if_conf}
    sed -i 's/#^NM_CONTROLLED=.*/NM_CONTROLLED=no/' ${if_conf}
  done;
  systemctl restart network
fi

NETWORK_IF_NAME=`echo ${if_to_edit} | awk -F- '{ print $3 }'`

# Setup hosts file to support ping by hostname to master
if [ ! "$(cat /etc/hosts | grep $MASTER_NAME)" ]; then
  echo "Adding $MASTER_NAME to hosts file"
  echo "$MASTER_IP $MASTER_NAME" >> /etc/hosts
fi
echo "$MINION_IP $MINION_NAME" >> /etc/hosts

# Setup hosts file to support ping by hostname to each minion in the cluster
for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
  minion=${MINION_NAMES[$i]}
  ip=${MINION_IPS[$i]}
  if [ ! "$(cat /etc/hosts | grep $minion)" ]; then
    echo "Adding $minion to hosts file"
    echo "$ip $minion" >> /etc/hosts
  fi
done

# Configure the openvswitch network
if [ "${NETWORK_PROVIDER}" == "calico" ]; then
  echo "Using default networking for Calico on minion"
else
  echo "Provisioning flannel network on minion"
  provision-network-minion
fi

# Placeholder for any other manifests that may be per-node.
mkdir -p /etc/kubernetes/manifests

# Let the minion know who its master is
# Recover the salt-minion if the salt-master network changes
## auth_timeout - how long we want to wait for a time out
## auth_tries - how many times we will retry before restarting salt-minion
## auth_safemode - if our cert is rejected, we will restart salt minion
## ping_interval - restart the minion if we cannot ping the master after 1 minute
## random_reauth_delay - wait 0-3 seconds when reauthenticating
## recon_default - how long to wait before reconnecting
## recon_max - how long you will wait upper bound
## state_aggregrate - try to do a single yum command to install all referenced packages where possible at once, should improve startup times
##
mkdir -p /etc/salt/minion.d
cat <<EOF >/etc/salt/minion.d/master.conf
master: '$(echo "$MASTER_NAME" | sed -e "s/'/''/g")'
auth_timeout: 10
auth_tries: 2
auth_safemode: True
ping_interval: 1
random_reauth_delay: 3
state_aggregrate:
  - pkg
EOF

cat <<EOF >/etc/salt/minion.d/log-level-debug.conf
log_level: debug
log_level_logfile: debug
EOF

# Our minions will have a pool role to distinguish them from the master.
cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  cloud: vagrant
  node_ip: '$(echo "$MINION_IP" | sed -e "s/'/''/g")'
  api_servers: '$(echo "$MASTER_IP" | sed -e "s/'/''/g")'
  networkInterfaceName: '$(echo "$NETWORK_IF_NAME" | sed -e "s/'/''/g")'
  roles:
    - kubernetes-pool
  cbr-cidr: '$(echo "$MINION_CONTAINER_CIDR" | sed -e "s/'/''/g")'
  container_subnet: '$(echo "$MINION_CONTAINER_SUBNET" | sed -e "s/'/''/g")'
  hostname_override: '$(echo "$MINION_IP" | sed -e "s/'/''/g")'
  docker_opts: '$(echo "$DOCKER_OPTS" | sed -e "s/'/''/g")'
EOF

# QoS support requires that swap memory is disabled on each of the minions
echo "Disable swap memory to ensure proper QoS"
swapoff -a

# we will run provision to update code each time we test, so we do not want to do salt install each time
if ! which salt-minion >/dev/null 2>&1; then
  # Install Salt
  curl -sS -L --connect-timeout 20 --retry 6 --retry-delay 10 https://bootstrap.saltstack.com | sh -s
  
  # Edit the Salt minion unit file to do restart always
  # needed because vagrant uses this as basis for registration of nodes in cloud provider
  # set a oom_score_adj to -999 to prevent our node from being killed with salt-master and then making kubelet NotReady
  # because its not found in salt cloud provider call
  cat <<EOF >/usr/lib/systemd/system/salt-minion.service 
[Unit]
Description=The Salt Minion
After=syslog.target network.target

[Service]
Type=simple
ExecStart=/usr/bin/salt-minion
Restart=Always
OOMScoreAdjust=-999

[Install]
WantedBy=multi-user.target
EOF
  
  systemctl daemon-reload
  systemctl restart salt-minion.service

else
  # Sometimes the minion gets wedged when it comes up along with the master.
  # Restarting it here un-wedges it.
  systemctl restart salt-minion.service
fi
