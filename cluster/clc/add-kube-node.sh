#!/bin/sh
set -e
# Add node to existing CLC kubernetes cluster

# Usage info
function show_help() {
cat << EOF
Usage: ${0##*/} [OPTIONS]
Create servers in the CenturyLinkCloud environment and add to an
existing CLC kubernetes cluster

Environment variables CLC_V2_API_USERNAME and CLC_V2_API_PASSWD must be set in
order to access the CenturyLinkCloud API

     -h (--help)                   display this help and exit
     -c= (--clc_cluster_name=)     set the name of the cluster, as used in CLC group names
     -m= (--minion_count=)         number of kubernetes minion nodes to add
EOF
}

function exit_message() {
    echo "ERROR: $1" >&2
    exit 1
}


for i in "$@"
do
case $i in
    -h|--help)
    show_help && exit 0
    shift # past argument=value
    ;;
    -c=*|--clc_cluster_name=*)
    CLC_CLUSTER_NAME="${i#*=}"
    extra_args="$extra_args clc_cluster_name=$CLC_CLUSTER_NAME"
    shift # past argument=value
    ;;
    -m=*|--minion_count=*)
    minion_count="${i#*=}"
    extra_args="$extra_args minion_count=$minion_count"
    shift # past argument=value
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

CLC_CLUSTER_HOME=~/.clc_kube/${CLC_CLUSTER_NAME}
hosts_dir=${CLC_CLUSTER_HOME}/hosts/
config_dir=${CLC_CLUSTER_HOME}/config/

if [ ! -d ${config_dir} ]
  then
  exit_message "Configuration directory ${config_dir} not found"
fi

cd ansible

# set the _add_nodes_ variable
ansible-playbook create-minion-hosts.yml \
  -e add_nodes=1 \
  -e minion_count=$minion_count \
  -e config_vars=${CLC_CLUSTER_HOME}/config/minion_config.yml

#### verify access
ansible -i $hosts_dir   -m shell -a uptime all

#### Part3
echo "Part3 - Setting up kubernetes"
ansible-playbook -i $hosts_dir  -e config_vars=${config_dir}/minion_config.yml   install_kubernetes.yml  --limit minion
