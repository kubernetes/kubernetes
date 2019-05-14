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
  local -r manifests_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty"
  mkdir -p "${manifests_dir}/vertical-pod-autoscaler"
  if [[ "${ENABLE_VERTICAL_POD_AUTOSCALER:-}" == "true" ]]; then
    echo "Start Vertical Pod Autoscaler (VPA)"
    generate_vertical_pod_autoscaler_admission_controller_certs
    add_vpa_admission_webhook_host

    cp "${manifests_dir}/internal-vpa-crd.yaml" "${manifests_dir}/vertical-pod-autoscaler"
    cp "${manifests_dir}/internal-vpa-rbac.yaml" "${manifests_dir}/vertical-pod-autoscaler"
    setup-addon-manifests "addons" "vertical-pod-autoscaler"

    for component in admission-controller recommender updater; do
      setup_vertical_pod_autoscaler_component ${component} ${manifests_dir}
    done
  elif [[ "${ENABLE_UNIFIED_AUTOSCALING:-}" == "true" ]]; then
    cp "${manifests_dir}/internal-kuba-rbac.yaml" "${manifests_dir}/vertical-pod-autoscaler"
    setup-addon-manifests "addons" "vertical-pod-autoscaler"

    echo "Start Kubernetes Adapter for Unified Autoscaler (KUBA)"
    setup_vertical_pod_autoscaler_component "recommender" ${manifests_dir}
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

function setup_vertical_pod_autoscaler_component {
  local component=$1
  local manifests_dir=$2
  create-static-auth-kubeconfig-for-component vpa-${component}

  # Prepare manifest
  local src_file="${manifests_dir}/internal-vpa-${component}.manifest"

  if [[ ${component} == "recommender" ]]; then
    local uas_params="${VPA_UAS_PARAMS:-}"
    sed -i -e "s@{{uas_params}}@${uas_params}@g" "${src_file}"
  fi

  sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
  sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
  sed -i -e "s@{%.*%}@@g" "${src_file}"

  cp "${src_file}" /etc/kubernetes/manifests
}

function configure_healthcheck_component {
  # TODO: is there a reliable way to make this a no-op?
  create-static-auth-kubeconfig-for-component "gke-master-healthcheck"
}

function configure_pdcsi_component {
  create-static-auth-kubeconfig-for-component "pdcsi-controller"
}

function configure_controller_manager_component {
  # TODO: is there a reliable way to make this a no-op?
  setup-addon-manifests "addons" "rbac/gcp-controller-manager"
  create-static-auth-kubeconfig-for-component "gcp-controller-manager"
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

function create-static-auth-kubeconfig-for-component {
  local component=$1
  echo "Creating token for component ${component}"
  local token="$(secure_random 32)"
  append_or_replace_prefixed_line /etc/srv/kubernetes/known_tokens.csv "${token}," "system:${component},uid:system:${component}"
  create-kubeconfig ${component} ${token}
}

function create-kubeconfig {
  local component=$1
  local token=$2
  echo "Creating kubeconfig file for component ${component}"
  mkdir -p /etc/srv/kubernetes/${component}
  cat <<EOF >/etc/srv/kubernetes/${component}/kubeconfig
apiVersion: v1
kind: Config
users:
- name: ${component}
  user:
    token: ${token}
clusters:
- name: local
  cluster:
    insecure-skip-tls-verify: true
    server: https://localhost:443
contexts:
- context:
    cluster: local
    user: ${component}
  name: ${component}
current-context: ${component}
EOF
}

function gke-internal-master-start {
  echo "Internal GKE configuration start"
  configure_healthcheck_component
  configure_pdcsi_component
  configure_machine_controller_component
  compute-master-manifest-variables
  start_internal_cluster_autoscaler
  start_vertical_pod_autoscaler
  echo "Internal GKE configuration done"
}

# Configure the node kernel parameters.
#
# This function expects no arguments.
#
# This function
#   - Reads the kernel parameter default values from release artifacts and the
#     overrides from SYSCTL_OVERRIDES, generates the sysctl conf files under
#     /etc/sysctl.d/, and applies them using systemd-sysctl.
#   - Sets the variable POD_SYSCTLS with the namespaced GKE fleetwide kernel
#     parameters and the user overrides. The variable is expected to be read
#     by the start-kubelet function.
function configure-node-sysctls {
  local -r dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty/sysctl"
  local -r sysctl_overrides="${SYSCTL_OVERRIDES:-}"
  local -r namespaced_sysctl_names="${dir}/namespaced-sysctl-names.yaml"
  # Use the GKE fleetwide default values if ENABLE_SYSCTL_TUNING is "true".
  if [[ "${ENABLE_SYSCTL_TUNING:-}" == "true" ]]; then
    local -r sysctl_defaults="${dir}/sysctl-defaults.yaml"
  else
    local -r sysctl_defaults="/dev/null"
  fi

  local -r conf_dir="/etc/sysctl.d"
  # The overrides must be applied after the defaults. This is guaranteed by
  # the alphabetical order of the file names.
  #
  # It's guaranteed that 99-sysctl.conf is the only file that comes after
  # 99-gke-defaults.conf and 99-gke-overrides.conf in the current sysctl config
  # layout on COS and Ubuntu.
  #
  # On both images, 99-sysctl.conf is a symlink to /etc/sysctl.conf, which
  # contains no settings.
  #
  # TODO(b/131158180): Allow GKE to provide sysctl config files in a more
  # reliable way by renaming the existing 99-*.conf files to 8x-*.conf.
  local -r output_defaults="${conf_dir}/99-gke-defaults.conf"
  local -r output_overrides="${conf_dir}/99-gke-overrides.conf"

  # Create the directory in case it doesn't exist.
  mkdir -p "${conf_dir}"

  echo "Sysctl overrides: ${sysctl_overrides}"

  # Generate the kernel parameter defaults and overrides configs in
  # /etc/sysctl.d/. They will be loaded by systemd-sysctl on reboot.
  python "${dir}/generate-conf-files.py" \
    --sysctl-defaults="${sysctl_defaults}" \
    --sysctl-overrides="${sysctl_overrides}" \
    --output-defaults=${output_defaults} \
    --output-overrides=${output_overrides}

  # Extract the namespaced kernel parameter defaults and overrides that should
  # be passed to kubelet and set inside pod namespaces.
  POD_SYSCTLS=$(python "${dir}/extract-namespaced.py" \
    --sysctl-defaults="${sysctl_defaults}" \
    --sysctl-overrides="${sysctl_overrides}" \
    --namespaced-sysctl-names="${namespaced_sysctl_names}")

  echo "Sysctls to be set in pod namespaces: ${POD_SYSCTLS}"

  # Run systemd-sysctl to apply the kernel parameters on node.
  if [[ -e "/usr/lib/systemd/systemd-sysctl" ]]; then
    /usr/lib/systemd/systemd-sysctl
  else
    /lib/systemd/systemd-sysctl
  fi
}
