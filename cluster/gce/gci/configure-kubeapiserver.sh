#!/usr/bin/env bash
# Copyright 2016 The Kubernetes Authors.
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


# Configures etcd related flags of kube-apiserver.
function configure-etcd-params {
  local -n params_ref=$1

  if [[ -n "${ETCD_APISERVER_CA_KEY:-}" && -n "${ETCD_APISERVER_CA_CERT:-}" && -n "${ETCD_APISERVER_SERVER_KEY:-}" && -n "${ETCD_APISERVER_SERVER_CERT:-}" && -n "${ETCD_APISERVER_CLIENT_KEY:-}" && -n "${ETCD_APISERVER_CLIENT_CERT:-}" ]]; then
      params_ref+=" --etcd-servers=${ETCD_SERVERS:-https://127.0.0.1:2379}"
      params_ref+=" --etcd-cafile=${ETCD_APISERVER_CA_CERT_PATH}"
      params_ref+=" --etcd-certfile=${ETCD_APISERVER_CLIENT_CERT_PATH}"
      params_ref+=" --etcd-keyfile=${ETCD_APISERVER_CLIENT_KEY_PATH}"
  elif [[ -z "${ETCD_APISERVER_CA_KEY:-}" && -z "${ETCD_APISERVER_CA_CERT:-}" && -z "${ETCD_APISERVER_SERVER_KEY:-}" && -z "${ETCD_APISERVER_SERVER_CERT:-}" && -z "${ETCD_APISERVER_CLIENT_KEY:-}" && -z "${ETCD_APISERVER_CLIENT_CERT:-}" ]]; then
      params_ref+=" --etcd-servers=${ETCD_SERVERS:-http://127.0.0.1:2379}"
      echo "WARNING: ALL of ETCD_APISERVER_CA_KEY, ETCD_APISERVER_CA_CERT, ETCD_APISERVER_SERVER_KEY, ETCD_APISERVER_SERVER_CERT, ETCD_APISERVER_CLIENT_KEY and ETCD_APISERVER_CLIENT_CERT are missing, mTLS between etcd server and kube-apiserver is not enabled."
  else
      echo "ERROR: Some of ETCD_APISERVER_CA_KEY, ETCD_APISERVER_CA_CERT, ETCD_APISERVER_SERVER_KEY, ETCD_APISERVER_SERVER_CERT, ETCD_APISERVER_CLIENT_KEY and ETCD_APISERVER_CLIENT_CERT are missing, mTLS between etcd server and kube-apiserver cannot be enabled. Please provide all mTLS credential."
      exit 1
  fi

  if [[ -z "${ETCD_SERVERS:-}" ]]; then
    params_ref+=" --etcd-servers-overrides=${ETCD_SERVERS_OVERRIDES:-/events#http://127.0.0.1:4002}"
  elif [[ -n "${ETCD_SERVERS_OVERRIDES:-}" ]]; then
    params_ref+=" --etcd-servers-overrides=${ETCD_SERVERS_OVERRIDES:-}"
  fi

  if [[ -n "${STORAGE_BACKEND:-}" ]]; then
    params_ref+=" --storage-backend=${STORAGE_BACKEND}"
  fi

  if [[ -n "${STORAGE_MEDIA_TYPE:-}" ]]; then
    params_ref+=" --storage-media-type=${STORAGE_MEDIA_TYPE}"
  fi

  if [[ -n "${ETCD_COMPACTION_INTERVAL_SEC:-}" ]]; then
    params_ref+=" --etcd-compaction-interval=${ETCD_COMPACTION_INTERVAL_SEC}s"
  fi
}

# Starts kubernetes apiserver.
# It prepares the log file, loads the docker image, calculates variables, sets them
# in the manifest file, and then copies the manifest file to /etc/kubernetes/manifests.
#
# Assumed vars (which are calculated in function compute-master-manifest-variables)
#   CLOUD_CONFIG_OPT
#   CLOUD_CONFIG_VOLUME
#   CLOUD_CONFIG_MOUNT
#   DOCKER_REGISTRY
#   INSECURE_PORT_MAPPING
function start-kube-apiserver {
  echo "Start kubernetes api-server"
  prepare-log-file "${KUBE_API_SERVER_LOG_PATH:-/var/log/kube-apiserver.log}" "${KUBE_API_SERVER_RUNASUSER:-0}"
  prepare-log-file "${KUBE_API_SERVER_AUDIT_LOG_PATH:-/var/log/kube-apiserver-audit.log}" "${KUBE_API_SERVER_RUNASUSER:-0}"

  # Calculate variables and assemble the command line.
  local params="${API_SERVER_TEST_LOG_LEVEL:-"--v=2"} ${APISERVER_TEST_ARGS:-} ${CLOUD_CONFIG_OPT}"
  params+=" --allow-privileged=true"
  params+=" --cloud-provider=${CLOUD_PROVIDER_FLAG:-external}"
  params+=" --client-ca-file=${CA_CERT_BUNDLE_PATH}"

  # params is passed by reference, so no "$"
  configure-etcd-params params

  params+=" --secure-port=443"
  params+=" --tls-cert-file=${APISERVER_SERVER_CERT_PATH}"
  params+=" --tls-private-key-file=${APISERVER_SERVER_KEY_PATH}"
  if [[ -n "${OLD_MASTER_IP:-}" ]]; then
    local old_ips="${OLD_MASTER_IP}"
    if [[ -n "${OLD_LOAD_BALANCER_IP:-}" ]]; then
      old_ips+=",${OLD_LOAD_BALANCER_IP}"
    fi
    if [[ -n "${OLD_PRIVATE_VIP:-}" ]]; then
      old_ips+=",${OLD_PRIVATE_VIP}"
    fi
    params+=" --tls-sni-cert-key=${OLD_MASTER_CERT_PATH},${OLD_MASTER_KEY_PATH}:${old_ips}"
  fi
  if [[ -n "${TLS_CIPHER_SUITES:-}" ]]; then
    params+=" --tls-cipher-suites=${TLS_CIPHER_SUITES}"
  fi
  if [[ -e "${KUBE_HOME}/bin/gke-internal-configure-helper.sh" ]]; then
    params+=" $(gke-kube-apiserver-internal-sni-param)"
  fi
  params+=" --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname"
  if [[ -s "${REQUESTHEADER_CA_CERT_PATH:-}" ]]; then
    params+=" --requestheader-client-ca-file=${REQUESTHEADER_CA_CERT_PATH}"
    params+=" --requestheader-allowed-names=aggregator"
    params+=" --requestheader-extra-headers-prefix=X-Remote-Extra-"
    params+=" --requestheader-group-headers=X-Remote-Group"
    params+=" --requestheader-username-headers=X-Remote-User"
    params+=" --proxy-client-cert-file=${PROXY_CLIENT_CERT_PATH}"
    params+=" --proxy-client-key-file=${PROXY_CLIENT_KEY_PATH}"
  fi
  params+=" --enable-aggregator-routing=true"
  if [[ -e "${APISERVER_CLIENT_CERT_PATH}" ]] && [[ -e "${APISERVER_CLIENT_KEY_PATH}" ]]; then
    params+=" --kubelet-client-certificate=${APISERVER_CLIENT_CERT_PATH}"
    params+=" --kubelet-client-key=${APISERVER_CLIENT_KEY_PATH}"
  fi
  if [[ -n "${SERVICEACCOUNT_CERT_PATH:-}" ]]; then
    params+=" --service-account-key-file=${SERVICEACCOUNT_CERT_PATH}"
  fi
  local known_tokens_file='/etc/srv/kubernetes/known_tokens.csv'
  if [[ -f "${known_tokens_file}" ]]; then
    chown "${KUBE_API_SERVER_RUNASUSER:-0}":"${KUBE_API_SERVER_RUNASGROUP:-0}" "${known_tokens_file}"
  fi
  params+=" --token-auth-file=${known_tokens_file}"

  if [[ -n "${KUBE_APISERVER_REQUEST_TIMEOUT_SEC:-}" ]]; then
    params+=" --request-timeout=${KUBE_APISERVER_REQUEST_TIMEOUT_SEC}s"
  fi
  if [[ -n "${ENABLE_GARBAGE_COLLECTOR:-}" ]]; then
    params+=" --enable-garbage-collector=${ENABLE_GARBAGE_COLLECTOR}"
  fi
  if [[ -n "${NUM_NODES:-}" ]]; then
    # If the cluster is large, increase max-requests-inflight limit in apiserver.
    if [[ "${NUM_NODES}" -gt 3000 ]]; then
      params=$(append-param-if-not-present "${params}" "max-requests-inflight" 3000)
      params=$(append-param-if-not-present "${params}" "max-mutating-requests-inflight" 1000)
    elif [[ "${NUM_NODES}" -gt 500 ]]; then
      params=$(append-param-if-not-present "${params}" "max-requests-inflight" 1500)
      params=$(append-param-if-not-present "${params}" "max-mutating-requests-inflight" 500)
    fi
  fi
  if [[ -n "${SERVICE_CLUSTER_IP_RANGE:-}" ]]; then
    params+=" --service-cluster-ip-range=${SERVICE_CLUSTER_IP_RANGE}"
  fi
  params+=" --service-account-issuer=${SERVICEACCOUNT_ISSUER}"
  params+=" --api-audiences=${SERVICEACCOUNT_ISSUER}"
  params+=" --service-account-signing-key-file=${SERVICEACCOUNT_KEY_PATH}"

  local audit_policy_config_mount=""
  local audit_policy_config_volume=""
  local audit_webhook_config_mount=""
  local audit_webhook_config_volume=""
  if [[ "${ENABLE_APISERVER_ADVANCED_AUDIT:-}" == "true" ]]; then
    local -r audit_policy_file="/etc/audit_policy.config"
    params+=" --audit-policy-file=${audit_policy_file}"
    # Create the audit policy file, and mount it into the apiserver pod.
    create-master-audit-policy "${audit_policy_file}" "${ADVANCED_AUDIT_POLICY:-}"
    audit_policy_config_mount="{\"name\": \"auditpolicyconfigmount\",\"mountPath\": \"${audit_policy_file}\", \"readOnly\": true},"
    audit_policy_config_volume="{\"name\": \"auditpolicyconfigmount\",\"hostPath\": {\"path\": \"${audit_policy_file}\", \"type\": \"FileOrCreate\"}},"

    if [[ "${ADVANCED_AUDIT_BACKEND:-log}" == *"log"* ]]; then
      # The advanced audit log backend config matches the basic audit log config.
      params+=" --audit-log-path=/var/log/kube-apiserver-audit.log"
      params+=" --audit-log-maxage=0"
      params+=" --audit-log-maxbackup=0"
      # Lumberjack doesn't offer any way to disable size-based rotation. It also
      # has an in-memory counter that doesn't notice if you truncate the file.
      # 2000000000 (in MiB) is a large number that fits in 31 bits. If the log
      # grows at 10MiB/s (~30K QPS), it will rotate after ~6 years if apiserver
      # never restarts. Please manually restart apiserver before this time.
      params+=" --audit-log-maxsize=2000000000"

      # Batching parameters
      if [[ -n "${ADVANCED_AUDIT_LOG_MODE:-}" ]]; then
        params+=" --audit-log-mode=${ADVANCED_AUDIT_LOG_MODE}"
      fi
      if [[ -n "${ADVANCED_AUDIT_LOG_BUFFER_SIZE:-}" ]]; then
        params+=" --audit-log-batch-buffer-size=${ADVANCED_AUDIT_LOG_BUFFER_SIZE}"
      fi
      if [[ -n "${ADVANCED_AUDIT_LOG_MAX_BATCH_SIZE:-}" ]]; then
        params+=" --audit-log-batch-max-size=${ADVANCED_AUDIT_LOG_MAX_BATCH_SIZE}"
      fi
      if [[ -n "${ADVANCED_AUDIT_LOG_MAX_BATCH_WAIT:-}" ]]; then
        params+=" --audit-log-batch-max-wait=${ADVANCED_AUDIT_LOG_MAX_BATCH_WAIT}"
      fi
      if [[ -n "${ADVANCED_AUDIT_LOG_THROTTLE_QPS:-}" ]]; then
        params+=" --audit-log-batch-throttle-qps=${ADVANCED_AUDIT_LOG_THROTTLE_QPS}"
      fi
      if [[ -n "${ADVANCED_AUDIT_LOG_THROTTLE_BURST:-}" ]]; then
        params+=" --audit-log-batch-throttle-burst=${ADVANCED_AUDIT_LOG_THROTTLE_BURST}"
      fi
      if [[ -n "${ADVANCED_AUDIT_LOG_INITIAL_BACKOFF:-}" ]]; then
        params+=" --audit-log-initial-backoff=${ADVANCED_AUDIT_LOG_INITIAL_BACKOFF}"
      fi
      # Truncating backend parameters
      if [[ -n "${ADVANCED_AUDIT_TRUNCATING_BACKEND:-}" ]]; then
        params+=" --audit-log-truncate-enabled=${ADVANCED_AUDIT_TRUNCATING_BACKEND}"
      fi
    fi
    if [[ "${ADVANCED_AUDIT_BACKEND:-}" == *"webhook"* ]]; then
      # Create the audit webhook config file, and mount it into the apiserver pod.
      local -r audit_webhook_config_file="/etc/audit_webhook.config"
      params+=" --audit-webhook-config-file=${audit_webhook_config_file}"
      create-master-audit-webhook-config "${audit_webhook_config_file}"
      audit_webhook_config_mount="{\"name\": \"auditwebhookconfigmount\",\"mountPath\": \"${audit_webhook_config_file}\", \"readOnly\": true},"
      audit_webhook_config_volume="{\"name\": \"auditwebhookconfigmount\",\"hostPath\": {\"path\": \"${audit_webhook_config_file}\", \"type\": \"FileOrCreate\"}},"

      # Batching parameters
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_MODE:-}" ]]; then
        params+=" --audit-webhook-mode=${ADVANCED_AUDIT_WEBHOOK_MODE}"
      else
        params+=" --audit-webhook-mode=batch"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_BUFFER_SIZE:-}" ]]; then
        params+=" --audit-webhook-batch-buffer-size=${ADVANCED_AUDIT_WEBHOOK_BUFFER_SIZE}"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_SIZE:-}" ]]; then
        params+=" --audit-webhook-batch-max-size=${ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_SIZE}"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_WAIT:-}" ]]; then
        params+=" --audit-webhook-batch-max-wait=${ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_WAIT}"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_THROTTLE_QPS:-}" ]]; then
        params+=" --audit-webhook-batch-throttle-qps=${ADVANCED_AUDIT_WEBHOOK_THROTTLE_QPS}"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_THROTTLE_BURST:-}" ]]; then
        params+=" --audit-webhook-batch-throttle-burst=${ADVANCED_AUDIT_WEBHOOK_THROTTLE_BURST}"
      fi
      if [[ -n "${ADVANCED_AUDIT_WEBHOOK_INITIAL_BACKOFF:-}" ]]; then
        params+=" --audit-webhook-initial-backoff=${ADVANCED_AUDIT_WEBHOOK_INITIAL_BACKOFF}"
      fi
      # Truncating backend parameters
      if [[ -n "${ADVANCED_AUDIT_TRUNCATING_BACKEND:-}" ]]; then
        params+=" --audit-webhook-truncate-enabled=${ADVANCED_AUDIT_TRUNCATING_BACKEND}"
      fi
    fi
  fi

  if [[ "${ENABLE_APISERVER_DYNAMIC_AUDIT:-}" == "true" ]]; then
    params+=" --audit-dynamic-configuration"
    RUNTIME_CONFIG="${RUNTIME_CONFIG},auditconfiguration.k8s.io/v1alpha1=true"
  fi

  if [[ "${ENABLE_APISERVER_LOGS_HANDLER:-}" == "false" ]]; then
    params+=" --enable-logs-handler=false"
  fi
  if [[ "${APISERVER_SET_KUBELET_CA:-false}" == "true" ]]; then
    params+=" --kubelet-certificate-authority=${CA_CERT_BUNDLE_PATH}"
  fi

  if [[ -n "${ADMISSION_CONTROL:-}" ]]; then
    params+=" --enable-admission-plugins=${ADMISSION_CONTROL}"
    params+=" --admission-control-config-file=/etc/srv/kubernetes/admission_controller_config.yaml"
  fi

  if [[ -n "${KUBE_APISERVER_REQUEST_TIMEOUT:-}" ]]; then
    params+=" --min-request-timeout=${KUBE_APISERVER_REQUEST_TIMEOUT}"
  fi
  if [[ -n "${RUNTIME_CONFIG:-}" ]]; then
    params+=" --runtime-config=${RUNTIME_CONFIG}"
  fi
  if [[ -n "${FEATURE_GATES:-}" ]]; then
    params+=" --feature-gates=${FEATURE_GATES}"
  fi
  if [[ -n "${MASTER_ADVERTISE_ADDRESS:-}" ]]; then
    params+=" --advertise-address=${MASTER_ADVERTISE_ADDRESS}"
  elif [[ -n "${PROJECT_ID:-}" && -n "${TOKEN_URL:-}" && -n "${TOKEN_BODY:-}" && -n "${NODE_NETWORK:-}" ]]; then
    local -r vm_external_ip=$(get-metadata-value "instance/network-interfaces/0/access-configs/0/external-ip")
    params+=" --advertise-address=${vm_external_ip}"
    if [[ -n "${KUBE_API_SERVER_RUNASUSER:-}" && -n "${KUBE_API_SERVER_RUNASGROUP:-}" ]]; then
      chown -R "${KUBE_API_SERVER_RUNASUSER}":"${KUBE_API_SERVER_RUNASGROUP}" /etc/srv/sshproxy/
    fi
  fi

  local webhook_authn_config_mount=""
  local webhook_authn_config_volume=""
  if [[ -n "${GCP_AUTHN_URL:-}" ]]; then
    params+=" --authentication-token-webhook-config-file=/etc/gcp_authn.config"
    webhook_authn_config_mount="{\"name\": \"webhookauthnconfigmount\",\"mountPath\": \"/etc/gcp_authn.config\", \"readOnly\": true},"
    webhook_authn_config_volume="{\"name\": \"webhookauthnconfigmount\",\"hostPath\": {\"path\": \"/etc/gcp_authn.config\", \"type\": \"File\"}},"
    if [[ -n "${GCP_AUTHN_CACHE_TTL:-}" ]]; then
      params+=" --authentication-token-webhook-cache-ttl=${GCP_AUTHN_CACHE_TTL}"
    fi
  fi

  local authorization_mode="RBAC"
  local -r src_dir="${KUBE_HOME}/kube-manifests/kubernetes/gci-trusty"

  # Enable ABAC mode unless the user explicitly opts out with ENABLE_LEGACY_ABAC=false
  if [[ "${ENABLE_LEGACY_ABAC:-}" != "false" ]]; then
    echo "Warning: Enabling legacy ABAC policy. All service accounts will have superuser API access. Set ENABLE_LEGACY_ABAC=false to disable this."
    # Create the ABAC file if it doesn't exist yet, or if we have a KUBE_USER set (to ensure the right user is given permissions)
    if [[ -n "${KUBE_USER:-}" || ! -e /etc/srv/kubernetes/abac-authz-policy.jsonl ]]; then
      local -r abac_policy_json="${src_dir}/abac-authz-policy.jsonl"
      if [[ -n "${KUBE_USER:-}" ]]; then
        sed -i -e "s/{{kube_user}}/${KUBE_USER}/g" "${abac_policy_json}"
      else
        sed -i -e "/{{kube_user}}/d" "${abac_policy_json}"
      fi
      cp "${abac_policy_json}" /etc/srv/kubernetes/
    fi

    params+=" --authorization-policy-file=/etc/srv/kubernetes/abac-authz-policy.jsonl"
    authorization_mode+=",ABAC"
  fi

  local webhook_config_mount=""
  local webhook_config_volume=""
  if [[ -n "${GCP_AUTHZ_URL:-}" ]]; then
    authorization_mode="${authorization_mode},Webhook"
    params+=" --authorization-webhook-config-file=/etc/gcp_authz.config"
    webhook_config_mount="{\"name\": \"webhookconfigmount\",\"mountPath\": \"/etc/gcp_authz.config\", \"readOnly\": true},"
    webhook_config_volume="{\"name\": \"webhookconfigmount\",\"hostPath\": {\"path\": \"/etc/gcp_authz.config\", \"type\": \"File\"}},"
    if [[ -n "${GCP_AUTHZ_CACHE_AUTHORIZED_TTL:-}" ]]; then
      params+=" --authorization-webhook-cache-authorized-ttl=${GCP_AUTHZ_CACHE_AUTHORIZED_TTL}"
    fi
    if [[ -n "${GCP_AUTHZ_CACHE_UNAUTHORIZED_TTL:-}" ]]; then
      params+=" --authorization-webhook-cache-unauthorized-ttl=${GCP_AUTHZ_CACHE_UNAUTHORIZED_TTL}"
    fi
  fi
  authorization_mode="Node,${authorization_mode}"
  params+=" --authorization-mode=${authorization_mode}"

  local csc_config_mount=""
  local csc_config_volume=""
  local default_konnectivity_socket_vol=""
  local default_konnectivity_socket_mnt=""
  if [[ "${PREPARE_KONNECTIVITY_SERVICE:-false}" == "true" ]]; then
    # Create the EgressSelectorConfiguration yaml file to control the Egress Selector.
    csc_config_mount="{\"name\": \"cscconfigmount\",\"mountPath\": \"/etc/srv/kubernetes/egress_selector_configuration.yaml\", \"readOnly\": false},"
    csc_config_volume="{\"name\": \"cscconfigmount\",\"hostPath\": {\"path\": \"/etc/srv/kubernetes/egress_selector_configuration.yaml\", \"type\": \"FileOrCreate\"}},"

    # UDS socket for communication between apiserver and konnectivity-server
    local default_konnectivity_socket_path="/etc/srv/kubernetes/konnectivity-server"
    default_konnectivity_socket_vol="{ \"name\": \"konnectivity-socket\", \"hostPath\": {\"path\": \"${default_konnectivity_socket_path}\", \"type\": \"DirectoryOrCreate\"}},"
    default_konnectivity_socket_mnt="{ \"name\": \"konnectivity-socket\", \"mountPath\": \"${default_konnectivity_socket_path}\", \"readOnly\": false},"
  fi
  if [[ "${EGRESS_VIA_KONNECTIVITY:-false}" == "true" ]]; then
    params+=" --egress-selector-config-file=/etc/srv/kubernetes/egress_selector_configuration.yaml"
  fi

  local container_env=""
  if [[ -n "${ENABLE_CACHE_MUTATION_DETECTOR:-}" ]]; then
    container_env+="{\"name\": \"KUBE_CACHE_MUTATION_DETECTOR\", \"value\": \"${ENABLE_CACHE_MUTATION_DETECTOR}\"}"
  fi
  if [[ -n "${ENABLE_KUBE_WATCHLIST_INCONSISTENCY_DETECTOR:-}" ]]; then
    if [[ -n "${container_env}" ]]; then
      container_env="${container_env}, "
    fi
    container_env+="{\"name\": \"KUBE_WATCHLIST_INCONSISTENCY_DETECTOR\", \"value\": \"${ENABLE_KUBE_WATCHLIST_INCONSISTENCY_DETECTOR}\"}"
  fi
  if [[ -n "${ENABLE_PATCH_CONVERSION_DETECTOR:-}" ]]; then
    if [[ -n "${container_env}" ]]; then
      container_env="${container_env}, "
    fi
    container_env+="{\"name\": \"KUBE_PATCH_CONVERSION_DETECTOR\", \"value\": \"${ENABLE_PATCH_CONVERSION_DETECTOR}\"}"
  fi
  if [[ -n "${KUBE_APISERVER_GODEBUG:-}" ]]; then
    if [[ -n "${container_env}" ]]; then
      container_env="${container_env}, "
    fi
    container_env+="{\"name\": \"GODEBUG\", \"value\": \"${KUBE_APISERVER_GODEBUG}\"}"
  fi
  if [[ -n "${container_env}" ]]; then
    container_env="\"env\":[${container_env}],"
  fi

  local -r src_file="${src_dir}/kube-apiserver.manifest"

  # params is passed by reference, so no "$"
  setup-etcd-encryption "${src_file}" params

  local healthcheck_ip="127.0.0.1"
  if [[ ${KUBE_APISERVER_HEALTHCHECK_ON_HOST_IP:-} == "true" ]]; then
    healthcheck_ip=$(hostname -i)
  fi

  params="$(convert-manifest-params "${params}")"
  # Evaluate variables.
  local -r kube_apiserver_docker_tag="${KUBE_API_SERVER_DOCKER_TAG:-$(cat /home/kubernetes/kube-docker-files/kube-apiserver.docker_tag)}"
  sed -i -e "s@{{params}}@${params}@g" "${src_file}"
  sed -i -e "s@{{container_env}}@${container_env}@g" "${src_file}"
  sed -i -e "s@{{srv_sshproxy_path}}@/etc/srv/sshproxy@g" "${src_file}"
  sed -i -e "s@{{cloud_config_mount}}@${CLOUD_CONFIG_MOUNT}@g" "${src_file}"
  sed -i -e "s@{{cloud_config_volume}}@${CLOUD_CONFIG_VOLUME}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube_docker_registry'\]}}@${DOCKER_REGISTRY}@g" "${src_file}"
  sed -i -e "s@{{pillar\['kube-apiserver_docker_tag'\]}}@${kube_apiserver_docker_tag}@g" "${src_file}"
  sed -i -e "s@{{pillar\['allow_privileged'\]}}@true@g" "${src_file}"
  sed -i -e "s@{{liveness_probe_initial_delay}}@${KUBE_APISERVER_LIVENESS_PROBE_INITIAL_DELAY_SEC:-15}@g" "${src_file}"
  sed -i -e "s@{{secure_port}}@443@g" "${src_file}"
  sed -i -e "s@{{insecure_port_mapping}}@${INSECURE_PORT_MAPPING}@g" "${src_file}"
  sed -i -e "s@{{additional_cloud_config_mount}}@@g" "${src_file}"
  sed -i -e "s@{{additional_cloud_config_volume}}@@g" "${src_file}"
  sed -i -e "s@{{webhook_authn_config_mount}}@${webhook_authn_config_mount}@g" "${src_file}"
  sed -i -e "s@{{webhook_authn_config_volume}}@${webhook_authn_config_volume}@g" "${src_file}"
  sed -i -e "s@{{webhook_config_mount}}@${webhook_config_mount}@g" "${src_file}"
  sed -i -e "s@{{webhook_config_volume}}@${webhook_config_volume}@g" "${src_file}"
  sed -i -e "s@{{csc_config_mount}}@${csc_config_mount}@g" "${src_file}"
  sed -i -e "s@{{csc_config_volume}}@${csc_config_volume}@g" "${src_file}"
  sed -i -e "s@{{audit_policy_config_mount}}@${audit_policy_config_mount}@g" "${src_file}"
  sed -i -e "s@{{audit_policy_config_volume}}@${audit_policy_config_volume}@g" "${src_file}"
  sed -i -e "s@{{audit_webhook_config_mount}}@${audit_webhook_config_mount}@g" "${src_file}"
  sed -i -e "s@{{audit_webhook_config_volume}}@${audit_webhook_config_volume}@g" "${src_file}"
  sed -i -e "s@{{konnectivity_socket_mount}}@${default_konnectivity_socket_mnt}@g" "${src_file}"
  sed -i -e "s@{{konnectivity_socket_volume}}@${default_konnectivity_socket_vol}@g" "${src_file}"
  sed -i -e "s@{{healthcheck_ip}}@${healthcheck_ip}@g" "${src_file}"

  if [[ -n "${KUBE_API_SERVER_RUNASUSER:-}" && -n "${KUBE_API_SERVER_RUNASGROUP:-}" && -n "${KUBE_PKI_READERS_GROUP:-}" ]]; then
    sed -i -e "s@{{runAsUser}}@\"runAsUser\": ${KUBE_API_SERVER_RUNASUSER},@g" "${src_file}"
    sed -i -e "s@{{runAsGroup}}@\"runAsGroup\": ${KUBE_API_SERVER_RUNASGROUP},@g" "${src_file}"
    sed -i -e "s@{{containerSecurityContext}}@\"securityContext\": { \"capabilities\": { \"drop\": [\"all\"], \"add\": [\"NET_BIND_SERVICE\"] } },@g" "${src_file}"
    local supplementalGroups="${KUBE_PKI_READERS_GROUP}"
    if [[ -n "${KMS_PLUGIN_SOCKET_WRITER_GROUP:-}" ]]; then
      supplementalGroups+=",${KMS_PLUGIN_SOCKET_WRITER_GROUP}"
    fi
    if [[ -n "${KONNECTIVITY_SERVER_SOCKET_WRITER_GROUP:-}" ]]; then
      supplementalGroups+=",${KONNECTIVITY_SERVER_SOCKET_WRITER_GROUP}"
    fi
    sed -i -e "s@{{supplementalGroups}}@\"supplementalGroups\": [ ${supplementalGroups} ],@g" "${src_file}"
  else
    sed -i -e "s@{{runAsUser}}@@g" "${src_file}"
    sed -i -e "s@{{runAsGroup}}@@g" "${src_file}"
    sed -i -e "s@{{containerSecurityContext}}@@g" "${src_file}"
    sed -i -e "s@{{supplementalGroups}}@@g" "${src_file}"
  fi

  cp "${src_file}" "${ETC_MANIFESTS:-/etc/kubernetes/manifests}"
}


# Sets-up etcd encryption.
# Configuration of etcd level encryption consists of the following steps:
# 1. Writing encryption provider config to disk
# 2. Adding encryption-provider-config flag to kube-apiserver
# 3. Add kms-socket-vol and kms-socket-vol-mnt to enable communication with kms-plugin (if requested)
#
# Expects parameters:
# $1 - path to kube-apiserver template
# $2 - kube-apiserver startup flags (must be passed by reference)
#
# Assumes vars (supplied via kube-env):
# ENCRYPTION_PROVIDER_CONFIG
# CLOUD_KMS_INTEGRATION
# ENCRYPTION_PROVIDER_CONFIG_PATH (will default to /etc/srv/kubernetes/encryption-provider-config.yml)
function setup-etcd-encryption {
  local kube_apiserver_template_path
  local -n kube_api_server_params
  local default_encryption_provider_config_vol
  local default_encryption_provider_config_vol_mnt
  local encryption_provider_config_vol_mnt
  local encryption_provider_config_vol
  local default_kms_socket_dir
  local default_kms_socket_vol_mnt
  local default_kms_socket_vol
  local kms_socket_vol_mnt
  local kms_socket_vol
  local encryption_provider_config_path

  kube_apiserver_template_path="$1"
  if [[ -z "${ENCRYPTION_PROVIDER_CONFIG:-}" ]]; then
    sed -i -e " {
      s@{{encryption_provider_mount}}@@
      s@{{encryption_provider_volume}}@@
      s@{{kms_socket_mount}}@@
      s@{{kms_socket_volume}}@@
    } " "${kube_apiserver_template_path}"
    return
  fi

  kube_api_server_params="$2"
  encryption_provider_config_path=${ENCRYPTION_PROVIDER_CONFIG_PATH:-/etc/srv/kubernetes/encryption-provider-config.yml}

  echo "${ENCRYPTION_PROVIDER_CONFIG}" | base64 --decode > "${encryption_provider_config_path}"
  kube_api_server_params+=" --encryption-provider-config=${encryption_provider_config_path}"

  default_encryption_provider_config_vol=$(echo "{ \"name\": \"encryptionconfig\", \"hostPath\": {\"path\": \"${encryption_provider_config_path}\", \"type\": \"File\"}}" | base64 | tr -d '\r\n')
  default_encryption_provider_config_vol_mnt=$(echo "{ \"name\": \"encryptionconfig\", \"mountPath\": \"${encryption_provider_config_path}\", \"readOnly\": true}" | base64 | tr -d '\r\n')

  encryption_provider_config_vol_mnt=$(echo "${ENCRYPTION_PROVIDER_CONFIG_VOL_MNT:-"${default_encryption_provider_config_vol_mnt}"}" | base64 --decode)
  encryption_provider_config_vol=$(echo "${ENCRYPTION_PROVIDER_CONFIG_VOL:-"${default_encryption_provider_config_vol}"}" | base64 --decode)
  sed -i -e " {
    s@{{encryption_provider_mount}}@${encryption_provider_config_vol_mnt},@
    s@{{encryption_provider_volume}}@${encryption_provider_config_vol},@
  } " "${kube_apiserver_template_path}"

  if [[ -n "${CLOUD_KMS_INTEGRATION:-}" ]]; then
    default_kms_socket_dir="/var/run/kmsplugin"
    default_kms_socket_vol_mnt=$(echo "{ \"name\": \"kmssocket\", \"mountPath\": \"${default_kms_socket_dir}\", \"readOnly\": false}" | base64 | tr -d '\r\n')
    default_kms_socket_vol=$(echo "{ \"name\": \"kmssocket\", \"hostPath\": {\"path\": \"${default_kms_socket_dir}\", \"type\": \"DirectoryOrCreate\"}}" | base64 | tr -d '\r\n')

    kms_socket_vol_mnt=$(echo "${KMS_PLUGIN_SOCKET_VOL_MNT:-"${default_kms_socket_vol_mnt}"}" | base64 --decode)
    kms_socket_vol=$(echo "${KMS_PLUGIN_SOCKET_VOL:-"${default_kms_socket_vol}"}" | base64 --decode)
    sed -i -e " {
      s@{{kms_socket_mount}}@${kms_socket_vol_mnt},@
      s@{{kms_socket_volume}}@${kms_socket_vol},@
    } " "${kube_apiserver_template_path}"
  else
    sed -i -e " {
      s@{{kms_socket_mount}}@@
      s@{{kms_socket_volume}}@@
    } " "${kube_apiserver_template_path}"
  fi
}
