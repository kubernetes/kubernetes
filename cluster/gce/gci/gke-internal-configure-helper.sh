#!/bin/bash

function start_nap {
  if [[ "${ENABLE_NAP:-}" == "true" ]]; then
    echo "Start Node Auto-Provisioning (NAP)"
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
    local params="${CLOUD_CONFIG_OPT} ${NAP_CONFIG:-}"

    sed -i -e "s@{{params}}@${params:-}@g" "${src_file}"
    sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
    sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
    sed -i -e "s@{%.*%}@@g" "${src_file}"

    cp "${src_file}" /etc/kubernetes/manifests
  fi
}

function gke-internal-master-start {
  echo "Internal GKE configuration start"
  compute-master-manifest-variables
  start_nap
  echo "Internal GKE configuration done"
}
