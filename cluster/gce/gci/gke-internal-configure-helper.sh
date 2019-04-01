#!/bin/bash

function start_internal_cluster_autoscaler {
  if [[ "${ENABLE_NAP:-}" == "true" ]]; then
    echo "Start Node Auto-Provisioning (NAP)"
    start_internal_ca "${NAP_CONFIG:-} --node-autoprovisioning-enabled=true"
  elif [[ "${ENABLE_GKE_CLUSTER_AUTOSCALER:-}" == "true" ]]; then
    echo "Start Cluster Autoscaler from closed source"
    start_internal_ca "${GKE_CLUSTER_AUTOSCALER_CONFIG:-}"
  else
    echo "Not using closed source Cluster Autoscaler"
  fi
}

function start_internal_ca {
  local -r manifests_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty"

  # Re-using Cluster Autoscaler setup functions from OSS
  setup-addon-manifests "addons" "rbac/cluster-autoscaler"
  create-clusterautoscaler-kubeconfig
  prepare-log-file /var/log/cluster-autoscaler.log

  # Add our GKE specific CRD
  mkdir -p "${manifests_dir}/autoscaling"
  cp "${manifests_dir}/internal-capacity-request-crd.yaml" "${manifests_dir}/autoscaling"
  setup-addon-manifests "addons" "autoscaling"

  # Prepare Autoscaler manifest
  local -r src_file="${manifests_dir}/internal-cluster-autoscaler.manifest"
  local params="${CLOUD_CONFIG_OPT} $1"

  sed -i -e "s@{{params}}@${params:-}@g" "${src_file}"
  sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
  sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
  sed -i -e "s@{%.*%}@@g" "${src_file}"

  cp "${src_file}" /etc/kubernetes/manifests
}

function add_vpa_admission_webhook_host {
  original_ipv6_loopback_line=`grep "^::1[[:space:]]" /etc/hosts`
  tmp_file=`mktemp`
  grep -v "^::1[[:space:]]" /etc/hosts >${tmp_file}
  cat ${tmp_file} >/etc/hosts
  if [[ -n "${original_ipv6_loopback_line:-}" ]]; then
    echo "${original_ipv6_loopback_line} vpa.admissionwebhook.localhost" >>/etc/hosts
  else
    echo "::1 vpa.admissionwebhook.localhost" >>/etc/hosts
  fi
}

function start_vertical_pod_autoscaler {
  if [[ "${ENABLE_VERTICAL_POD_AUTOSCALER:-}" == "true" ]]; then
    echo "Start Vertical Pod Autoscaler (VPA)"
    generate_vertical_pod_autoscaler_admission_controller_certs
    add_vpa_admission_webhook_host

    local -r manifests_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty"

    mkdir -p "${manifests_dir}/vertical-pod-autoscaler"

    cp "${manifests_dir}/internal-vpa-crd.yaml" "${manifests_dir}/vertical-pod-autoscaler"
    cp "${manifests_dir}/internal-vpa-rbac.yaml" "${manifests_dir}/vertical-pod-autoscaler"
    setup-addon-manifests "addons" "vertical-pod-autoscaler"

    for component in admission-controller recommender updater; do
      token=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
      append_or_replace_prefixed_line /etc/srv/kubernetes/known_tokens.csv "${token}," "vpa-${component},uid:vpa-${component}"
      create-vpa-kubeconfig vpa-${component} ${token}

      # Prepare manifest
      local src_file="${manifests_dir}/internal-vpa-${component}.manifest"

      sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
      sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
      sed -i -e "s@{%.*%}@@g" "${src_file}"

      cp "${src_file}" /etc/kubernetes/manifests
    done

  fi
}

function base64_decode_or_die {
  local variable_name=$1
  local out_file=$2
  if [[ -n "${!variable_name}" ]]; then
    if ! base64 -d - <<<${!variable_name} >${out_file}; then
      echo "==error base 64 decoding ${variable_name}=="
      echo "==the value of the variable is ${!variable_name}=="
      exit 1
    fi
  else
    echo "==VPA enabled but ${variable_name} is not set=="
    exit 1
  fi
}

function generate_vertical_pod_autoscaler_admission_controller_certs {
  local certs_dir="/etc/tls-certs" #TODO: what is the best place for certs?
  echo "Generating certs for the VPA Admission Controller in ${certs_dir}."
  mkdir -p ${certs_dir}
  if [[ -n "${CA_CERT:-}" ]] && [[ -n "${VPA_AC_KEY:-}" ]] && [[ -n "${VPA_AC_CERT:-}" ]]; then
    base64_decode_or_die "CA_CERT" ${certs_dir}/caCert.pem
    base64_decode_or_die "VPA_AC_KEY" ${certs_dir}/serverKey.pem
    base64_decode_or_die "VPA_AC_CERT" ${certs_dir}/serverCert.pem
  else
    echo "==At least one of CA_CERT, VPA_AC_KEY, VPA_AC_CERT is missing=="
    exit 1
  fi
}

function create-vpa-kubeconfig {
  component_=$1
  token_=$2
  echo "Creating kubeconfig file for VPA component ${component_}"
  mkdir -p /etc/srv/kubernetes/${component_}
  cat <<EOF >/etc/srv/kubernetes/${component_}/kubeconfig
apiVersion: v1
kind: Config
users:
- name: ${component_}
  user:
    token: ${token_}
clusters:
- name: local
  cluster:
    insecure-skip-tls-verify: true
    server: https://localhost:443
contexts:
- context:
    cluster: local
    user: ${component_}
  name: ${component_}
current-context: ${component_}
EOF
}

function gke-internal-master-start {
  echo "Internal GKE configuration start"
  compute-master-manifest-variables
  start_internal_cluster_autoscaler
  start_vertical_pod_autoscaler
  echo "Internal GKE configuration done"
}
