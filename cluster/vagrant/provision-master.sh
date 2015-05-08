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

function release_not_found() {
  echo "It looks as if you don't have a compiled version of Kubernetes.  If you" >&2
  echo "are running from a clone of the git repo, please run ./build/release.sh." >&2
  echo "Note that this requires having Docker installed.  If you are running " >&2
  echo "from a release tarball, something is wrong.  Look at " >&2
  echo "http://kubernetes.io/ for information on how to contact the development team for help." >&2
  exit 1
}

# Look for our precompiled binary releases.  When running from a source repo,
# these are generated under _output.  When running from an release tarball these
# are under ./server.
server_binary_tar="/vagrant/server/kubernetes-server-linux-amd64.tar.gz"
if [[ ! -f "$server_binary_tar" ]]; then
  server_binary_tar="/vagrant/_output/release-tars/kubernetes-server-linux-amd64.tar.gz"
fi
if [[ ! -f "$server_binary_tar" ]]; then
  release_not_found
fi

salt_tar="/vagrant/server/kubernetes-salt.tar.gz"
if [[ ! -f "$salt_tar" ]]; then
  salt_tar="/vagrant/_output/release-tars/kubernetes-salt.tar.gz"
fi
if [[ ! -f "$salt_tar" ]]; then
  release_not_found
fi


# Setup hosts file to support ping by hostname to each minion in the cluster from apiserver
for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
  minion=${MINION_NAMES[$i]}
  ip=${MINION_IPS[$i]}
  if [ ! "$(cat /etc/hosts | grep $minion)" ]; then
    echo "Adding $minion to hosts file"
    echo "$ip $minion" >> /etc/hosts
  fi
done
echo "127.0.0.1 localhost" >> /etc/hosts # enables cmds like 'kubectl get pods' on master.

# Update salt configuration
mkdir -p /etc/salt/minion.d
cat <<EOF >/etc/salt/minion.d/master.conf
master: '$(echo "$MASTER_NAME" | sed -e "s/'/''/g")'
master: '$(echo "$MASTER_NAME" | sed -e "s/'/''/g")'
auth_timeout: 10
auth_tries: 2
auth_safemode: True
ping_interval: 1
random_reauth_delay: 3
state_aggregrate:
  - pkg
EOF

cat <<EOF >/etc/salt/minion.d/grains.conf
grains:
  node_ip: '$(echo "$MASTER_IP" | sed -e "s/'/''/g")'
  publicAddressOverride: '$(echo "$MASTER_IP" | sed -e "s/'/''/g")'
  network_mode: openvswitch
  networkInterfaceName: eth1
  api_servers: '$(echo "$MASTER_IP" | sed -e "s/'/''/g")'
  cloud: vagrant
  roles:
    - kubernetes-master
  runtime_config: '$(echo "$RUNTIME_CONFIG" | sed -e "s/'/''/g")'
EOF

mkdir -p /srv/salt-overlay/pillar
cat <<EOF >/srv/salt-overlay/pillar/cluster-params.sls
  portal_net: '$(echo "$PORTAL_NET" | sed -e "s/'/''/g")'
  cert_ip: '$(echo "$MASTER_IP" | sed -e "s/'/''/g")'
  enable_cluster_monitoring: '$(echo "$ENABLE_CLUSTER_MONITORING" | sed -e "s/'/''/g")'
  enable_node_monitoring: '$(echo "$ENABLE_NODE_MONITORING" | sed -e "s/'/''/g")'
  enable_cluster_logging: '$(echo "$ENABLE_CLUSTER_LOGGING" | sed -e "s/'/''/g")'
  enable_node_logging: '$(echo "$ENABLE_NODE_LOGGING" | sed -e "s/'/''/g")'
  logging_destination: '$(echo "$LOGGING_DESTINATION" | sed -e "s/'/''/g")'
  elasticsearch_replicas: '$(echo "$ELASTICSEARCH_LOGGING_REPLICAS" | sed -e "s/'/''/g")'
  enable_cluster_dns: '$(echo "$ENABLE_CLUSTER_DNS" | sed -e "s/'/''/g")'
  dns_replicas: '$(echo "$DNS_REPLICAS" | sed -e "s/'/''/g")'
  dns_server: '$(echo "$DNS_SERVER_IP" | sed -e "s/'/''/g")'
  dns_domain: '$(echo "$DNS_DOMAIN" | sed -e "s/'/''/g")'
  instance_prefix: '$(echo "$INSTANCE_PREFIX" | sed -e "s/'/''/g")'
  admission_control: '$(echo "$ADMISSION_CONTROL" | sed -e "s/'/''/g")'  
EOF

# Configure the salt-master
# Auto accept all keys from minions that try to join
mkdir -p /etc/salt/master.d
cat <<EOF >/etc/salt/master.d/auto-accept.conf
open_mode: True
auto_accept: True
EOF

cat <<EOF >/etc/salt/master.d/reactor.conf
# React to new minions starting by running highstate on them.
reactor:
  - 'salt/minion/*/start':
    - /srv/reactor/highstate-new.sls
EOF

cat <<EOF >/etc/salt/master.d/salt-output.conf
# Minimize the amount of output to terminal
state_verbose: False
state_output: mixed
log_level: debug
log_level_logfile: debug
EOF

cat <<EOF >/etc/salt/minion.d/log-level-debug.conf
log_level: debug
log_level_logfile: debug
EOF


# Generate and distribute a shared secret (bearer token) to
# apiserver and kubelet so that kubelet can authenticate to
# apiserver to send events.
known_tokens_file="/srv/salt-overlay/salt/kube-apiserver/known_tokens.csv"
if [[ ! -f "${known_tokens_file}" ]]; then
  kubelet_token=$(cat /dev/urandom | base64 | tr -d "=+/" | dd bs=32 count=1 2> /dev/null)
  kube_proxy_token=$(cat /dev/urandom | base64 | tr -d "=+/" | dd bs=32 count=1 2> /dev/null)

  mkdir -p /srv/salt-overlay/salt/kube-apiserver
  known_tokens_file="/srv/salt-overlay/salt/kube-apiserver/known_tokens.csv"
  (umask u=rw,go= ;
   echo "$kubelet_token,kubelet,kubelet" > $known_tokens_file;
   echo "$kube_proxy_token,kube_proxy,kube_proxy" >> $known_tokens_file)

  mkdir -p /srv/salt-overlay/salt/kubelet
  kubelet_auth_file="/srv/salt-overlay/salt/kubelet/kubernetes_auth"
  (umask u=rw,go= ; echo "{\"BearerToken\": \"$kubelet_token\", \"Insecure\": true }" > $kubelet_auth_file)

  mkdir -p /srv/salt-overlay/salt/kube-proxy
  kube_proxy_kubeconfig_file="/srv/salt-overlay/salt/kube-proxy/kubeconfig"
  # Make a kubeconfig file with the token.
  # TODO(etune): put apiserver certs into secret too, and reference from authfile,
  # so that "Insecure" is not needed.
  (umask 077;
  cat > "${kube_proxy_kubeconfig_file}" <<EOF
apiVersion: v1
kind: Config
users:
- name: kube-proxy
  user:
    token: ${kube_proxy_token}
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

  # Generate tokens for other "service accounts".  Append to known_tokens.
  #
  # NB: If this list ever changes, this script actually has to
  # change to detect the existence of this file, kill any deleted
  # old tokens and add any new tokens (to handle the upgrade case).
  service_accounts=("system:scheduler" "system:controller_manager" "system:logging" "system:monitoring" "system:dns")
  for account in "${service_accounts[@]}"; do
    token=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
    echo "${token},${account},${account}" >> "${known_tokens_file}"
  done
fi

# Configure nginx authorization
mkdir -p /srv/salt-overlay/salt/nginx
if [[ ! -f /srv/salt-overlay/salt/nginx/htpasswd ]]; then
  python "${KUBE_ROOT}/third_party/htpasswd/htpasswd.py" \
    -b -c "/srv/salt-overlay/salt/nginx/htpasswd" \
    "$MASTER_USER" "$MASTER_PASSWD"
fi

echo "Running release install script"
rm -rf /kube-install
mkdir -p /kube-install
pushd /kube-install
  tar xzf "$salt_tar"
  cp "$server_binary_tar" .
  ./kubernetes/saltbase/install.sh "${server_binary_tar##*/}"
popd

# we will run provision to update code each time we test, so we do not want to do salt installs each time
if ! which salt-master &>/dev/null; then

  # Configure the salt-api
  cat <<EOF >/etc/salt/master.d/salt-api.conf
# Set vagrant user as REST API user
external_auth:
  pam:
    vagrant:
      - .*
rest_cherrypy:
  port: 8000
  host: 127.0.0.1
  disable_ssl: True
  webhook_disable_auth: True
EOF

  # Install Salt Master
  #
  # -M installs the master
  # -N does not install the minion
  curl -sS -L --connect-timeout 20 --retry 6 --retry-delay 10 https://bootstrap.saltstack.com | sh -s -- -M -N

  # Install salt-api
  #
  # This is used to provide the network transport for salt-api
  yum install -y python-cherrypy
  # This is used to inform the cloud provider used in the vagrant cluster
  yum install -y salt-api
  # Set log level to a level higher than "info" to prevent the message about
  # enabling the service (which is not an error) from being printed to stderr.
  SYSTEMD_LOG_LEVEL=notice systemctl enable salt-api
  systemctl start salt-api
fi

if ! which salt-minion >/dev/null 2>&1; then

  # Install Salt minion
  curl -sS -L --connect-timeout 20 --retry 6 --retry-delay 10 https://bootstrap.saltstack.com | sh -s

else
  # Only run highstate when updating the config.  In the first-run case, Salt is
  # set up to run highstate as new minions join for the first time.
  echo "Executing configuration"
  salt '*' mine.update
  salt --show-timeout --force-color '*' state.highstate
fi
