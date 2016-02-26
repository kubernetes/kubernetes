#!/bin/sh
set -e
# deploy k8s cluster on clc
#
# Examples:
#
# Make a cluster with default values:
# > bash kube-up.sh
#
# Make a cluster with custom values (cluster of VMs)
# > bash kube-up.sh --clc_cluster_name=k8s_vm101 --minion_type=standard --minion_count=6 --datacenter=VA1 --vm_memory=4 --vm_cpu=2
#
# Make a cluster with custom values (cluster of physical servers)
# > bash kube-up.sh --clc_cluster_name=k8s_vm101 --minion_type=bareMetal --minion_count=4 --datacenter=VA1
#
# Make a cluster with custom values (cluster of VMs with a separate cluster of etcd nodes)
# > bash kube-up.sh --clc_cluster_name=k8s_vm101 --minion_type=standard --minion_count=6 --datacenter=VA1 --etcd_separate_cluster
#

# Usage info
function show_help() {
cat << EOF
Usage: ${0##*/} [OPTIONS]
Create servers in the CenturyLinkCloud environment and initialize a Kubernetes cluster
Environment variables CLC_V2_API_USERNAME and CLC_V2_API_PASSWD must be set in
order to access the CenturyLinkCloud API

Most options (both short and long form) require arguments, and must include "="
between option name and option value. _--help_ and _--etcd_separate_cluster_ do
not take arguments

     -h (--help)                   display this help and exit
     -c= (--clc_cluster_name=)     set the name of the cluster, as used in CLC group names
     -d= (--datacenter=)           VA1 (default)
     -m= (--minion_count=)         number of kubernetes minion nodes
     -mem= (--vm_memory=)          number of GB ram for each minion
     -cpu= (--vm_cpu=)             number of virtual cps for each minion node
     -t= (--minion_type=)          standard -> VM (default), bareMetal -> physical]
     -phyid= (--server_config_id=) if obtaining a bareMetal server, this configuration id
                                   must be set to one of:
                                      physical_server_20_core
                                      physical_server_12_core
                                      physical_server_4_core
     --etcd_separate_cluster       create a separate cluster of three etcd nodes,
                                   otherwise run etcd on the master node
EOF
}

function exit_message() {
    echo "ERROR: $1" >&2
    exit 1
}

# default values before reading the command-line args
datacenter=VA1
etcd_group=kube-master
minion_count=2
minion_type=standard
server_config_id=default
vm_memory=4
vm_cpu=2
skip_minion=False
async_time=7200
async_poll=5

for i in "$@"
do
case $i in
    -h|--help)
    show_help && exit 0
    shift # past argument=value
    ;;
    -c=*|--clc_cluster_name=*)
    CLC_CLUSTER_NAME="${i#*=}"
    shift # past argument=value
    ;;
    -d=*|--datacenter=*)
    datacenter="${i#*=}"
    shift # past argument=value
    ;;
    -m=*|--minion_count=*)
    minion_count="${i#*=}"
    shift # past argument=value
    ;;
    -mem=*|--vm_memory=*)
    vm_memory="${i#*=}"
    shift # past argument=value
    ;;
    -cpu=*|--vm_cpu=*)
    vm_cpu="${i#*=}"
    shift # past argument=value
    ;;

    -t=*|--minion_type=*)
    minion_type="${i#*=}"
    shift # past argument=value
    ;;
    -phyid=*|--server_config_id=*)
    server_config_id="${i#*=}"
    shift # past argument=value
    ;;

    --etcd_separate_cluster*)
    # the ansible variable "etcd_group" has default value "master"
    etcd_separate_cluster=yes
    etcd_group=etcd
    shift # past argument with no value
    ;;

    *)
    echo "Unknown option: $1"
    echo
    show_help
  	exit 1
    ;;

esac
done

if [ -z ${CLC_V2_API_USERNAME:-} ] || [ -z ${CLC_V2_API_PASSWD:-} ]
  then
  exit_message 'Environment variables CLC_V2_API_USERNAME, CLC_V2_API_PASSWD must be set'
fi

if [ -z ${CLC_CLUSTER_NAME} ]
  then
  exit_message 'Cluster name must be set with either command-line argument or as environment variable CLC_CLUSTER_NAME'
fi

if [[ ${minion_type} == "standard" ]]
then
  if [[ ${server_config_id} != "default" ]]
  then
    exit_message "Server configuration of \"${server_config_id}\" is not compatible with ${minion_type} VM, use \"default\""
  fi
elif [[ ${minion_type} == "bareMetal" ]]
then
    true # do nothing, validate internally in ansible
else
  exit_message "Minion type \"${minion_type}\" unknown"
fi




CLC_CLUSTER_HOME=~/.clc_kube/${CLC_CLUSTER_NAME}

mkdir -p ${CLC_CLUSTER_HOME}/hosts
mkdir -p ${CLC_CLUSTER_HOME}/config
created_flag=${CLC_CLUSTER_HOME}/created_on

cd ansible
if [ -e $created_flag ]
then
  echo "cluster file $created_flag already exists, skipping host creation"
else

  echo "Creating Kubernetes Cluster on CenturyLink Cloud"
  echo ""

  cat <<CONFIG > ${CLC_CLUSTER_HOME}/config/master_config.yml
clc_cluster_name: ${CLC_CLUSTER_NAME}
server_group: kube-master
etcd_group: ${etcd_group}
server_group_tag: master
datacenter: ${datacenter}
server_count: 1
server_config_id: default
server_memory: 4
server_cpu: 2
skip_minion: True
async_time: 7200
async_poll: 5
CONFIG

  cat <<CONFIG > ${CLC_CLUSTER_HOME}/config/minion_config.yml
clc_cluster_name: ${CLC_CLUSTER_NAME}
server_group: kube-node
server_group_tag: node
datacenter: ${datacenter}
server_count: ${minion_count}
minion_type: ${minion_type}
server_config_id: ${server_config_id}
server_memory: ${vm_memory}
server_cpu: ${vm_cpu}
skip_minion: False
async_time: 7200
async_poll: 5
CONFIG



  #### Part0
  echo "Part0a - Create local sshkey if necessary"
  ansible-playbook create-local-sshkey.yml \
     -e server_cert_store=${CLC_CLUSTER_HOME}/ssh

  echo "Part0b - Create parent group"
  ansible-playbook create-parent-group.yml \
      -e config_vars=${CLC_CLUSTER_HOME}/config/master_config.yml

  #### Part1a
  echo "Part1a - Building out the infrastructure on CLC"

  # background these in order to run them in parallel
  pids=""

  { ansible-playbook create-master-hosts.yml \
      -e config_vars=${CLC_CLUSTER_HOME}/config/master_config.yml;
  } &
  pids="$pids $!"

  { ansible-playbook create-minion-hosts.yml \
      -e config_vars=${CLC_CLUSTER_HOME}/config/minion_config.yml;
  } &
  pids="$pids $!"

  if [ -z ${etcd_separate_cluster+x} ]; then
    echo "ETCD will be installed on master server"
  else
    echo "ETCD will be installed on 3 separate VMs not part of k8s cluster"
    { ansible-playbook create-etcd-hosts.yml  \
        -e config_vars=${CLC_CLUSTER_HOME}/config/master_config.yml;
    } &
    pids="$pids $!"
  fi

  # -----------------------------------------------------
  # a _wait_ checkpoint to make sure these CLC hosts were
  # created safely, exiting if there were problems
  # -----------------------------------------------------
  set +e
  failed=0
  ps $pids
  for pid in $pids
  do
    wait $pid
    exit_val=$?
    if [ $exit_val != 0 ]
    then
      echo "process $pid failed with exit value $exit_val"
      failed=$exit_val
    fi
  done

  if [ $failed != 0 ]
  then
    exit $failed
  fi
  set -e
  # -----------------------------------------------------

  # write timestamp into flag file
  date +%Y-%m-%dT%H-%M-%S%z > $created_flag

fi # checking [ -e $created_flag ]

#### verify access
ansible -i ${CLC_CLUSTER_HOME}/hosts -m shell -a uptime all

#### Part2
echo "Part2 - Setting up etcd"
#install etcd on master or on separate cluster of vms
ansible-playbook -i ${CLC_CLUSTER_HOME}/hosts install_etcd.yml \
    -e config_vars=${CLC_CLUSTER_HOME}/config/master_config.yml

#### Part3
echo "Part3 - Setting up kubernetes"
ansible-playbook -i ${CLC_CLUSTER_HOME}/hosts install_kubernetes.yml \
    -e config_vars=${CLC_CLUSTER_HOME}/config/master_config.yml

#### Part4
echo "Part4 - Installing standard addons"
standard_addons='{"k8s_apps":["skydns","dashboard","kube-ui","monitoring"]}'
ansible-playbook -i ${CLC_CLUSTER_HOME}/hosts deploy_kube_applications.yml \
     -e ${standard_addons}

cat <<MESSAGE

Cluster build is complete. To administer the cluster, install and configure
kubectl with

  export CLC_CLUSTER_NAME=$CLC_CLUSTER_NAME
  ./install-kubectl.sh

If accessing the cluster services with a browser, the basic-authentication password
for the admin user is found in ${CLC_CLUSTER_HOME}/kube/admin_password.txt

MESSAGE
