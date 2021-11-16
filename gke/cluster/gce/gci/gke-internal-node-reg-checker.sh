#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
KUBE_HOME="/home/kubernetes"
KUBE_BIN=${KUBE_HOME}/bin
#timeout command
TIMEOUT=("timeout" "--foreground" "10s")
#curl options
CURL_TIMEOUT=("--connect-timeout" "2")
CURL_RETRY=("--retry" "2" "--retry-delay" "2")
CURL_RETRY_CONNREFUSED='--retry-connrefused'
# essential services endpoints
declare -A SERVICES_ENDPOINTS=( \
[GCS]="https://storage.googleapis.com/gke-release/" \
[GCR]="https://gke.gcr.io/" \
[LOGGING]="https://logging.googleapis.com" \
)
# store the status of the checks and are used to generate the summary-report in the end
declare -A REACHABILITY
declare -A DNS_RESOLUTION
declare -A MASTER_STATUS
SA_WORKING=""
KUBELET_LOG_LINES=0
CURL_FORMAT="time_namelookup:  %{time_namelookup}\n\
     time_connect: %{time_connect}\n\
  time_appconnect: %{time_appconnect}\n\
  tim_pretransfer: %{time_pretransfer}\n\
    time_redirect: %{time_redirect}\n\
  t_starttransfer: %{time_starttransfer}\n\
                  ----------\n\
       time_total: %{time_total}\n"
## BEGINNING of helper functions
# similar to echo but includes the timestamp in UTC
# it helps to understand how long each check took to run
# also if two params are passed to the function $1 is used to cause identation - useful to use inside the checks. ie:
# log "Checking Service Account"   -> Checking Service Account
# log 3 "Trying to get auth token" ->   Trying to get auth token -> log 3 "Trying to get auth token"
# log 5 "Trying to get project ID" ->      Trying to get project ID ->
function log {
  local date
  date=$(date --utc)
  spaces=""
  if [[ $# -eq 2 ]]; then
    spaces=$(printf "%$1s" " ")
    shift
  fi
  echo "${date} - ** ${spaces}${*} **"
}
function exists {
  command -v "$1" >/dev/null 2>&1
}
function detailed-curl {
  set +e # we need the return codes
  if time curl  -v -L "${CURL_TIMEOUT[@]}" --silent --show-error -o /dev/null --ipv4  "${1}"; then
    log "Connection succeeded to ${1}"
    return 0
  else
    log "Failed connecting to ${1}. Here is a breakdown of the connection time:"
    curl -w "${CURL_FORMAT}" -L "${CURL_TIMEOUT[@]}"  --silent --show-error -o /dev/null --ipv4 "${1}"
    return $?
  fi
  set -e # re-enable errexit
}
# curls an URL and extracts a JSON field(if provided)
# $1 = URL
# $2 = field to extract from response
function simple-curl {
  local v
  set +e # we need the return codes
  v=$(time curl --fail -v "${CURL_TIMEOUT[@]}" "${CURL_RETRY[@]}" ${CURL_RETRY_CONNREFUSED} --silent -k --show-error  -H "Metadata-Flavor: Google" -s "${1}")
  if [[ -z "${v}" ]]; then
    set -e # re-enable errexit
    return 1
  fi
  # if number of arguments is 1 we just return the curl output
  if [[ $# -eq 1 ]]; then
    echo "${v}"
  # if number of arguments is 2 we need to extract a field from a JSON response
  else
    echo "${v}" | python -c "import sys; import json; print(json.loads(sys.stdin.read())[\"${2}\"])"
  fi
  set -e # re-enable errexit
}
# Get default service account credentials of the VM
function get-credentials {
  simple-curl "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" "access_token"
}
# returns 000 if it fails to connect
function get-http-response {
  set +e # we need the return codes
  time curl --silent -v "${CURL_TIMEOUT[@]}" "${CURL_RETRY[@]}" ${CURL_RETRY_CONNREFUSED} --ipv4 -o /dev/null -w '%{http_code}' -H 'Metadata-Flavor: Google' "${@}"
  set -e # re-enable errexit
}
# Gets a value from the metadata server
#
# $1 entry to retrieve.
function retrieve-metadata-entry {
  simple-curl "http://metadata.google.internal/computeMetadata/v1/${1}?alt=text"
}
## END of helper functions
### BEGINNING of Checks
# each check is a separate function that will test something and print the results
# information can be stored in global variables to be used later in the summary-report()
# It's suggested to be verbose about the steps in the check so if a check fails the output
# generated should have enough info to help understand what could be missing or wrong with the node
# Tries to connect to all the endpoints and records their statuses
function check-service-endpoint {
  for K in "${!SERVICES_ENDPOINTS[@]}"; do
    log "Trying to connect to ${SERVICES_ENDPOINTS[$K]}"
    if detailed-curl "${SERVICES_ENDPOINTS[$K]}"; then
      # if detailed-curl returned 0 it was able to connect to the endpoint
      # therefore name was resolved and the endpoint is reachable
      REACHABILITY[$K]="true"
      DNS_RESOLUTION[$K]="true"
    elif  [ $? -eq 6 ]; then
      # if curl returned 6 it could not resolve the name
      # therefore can't reach the endpoint
      DNS_RESOLUTION[$K]="false"
      REACHABILITY[$K]="false"
    else
      # if curl returned anything else it was able to resolve the name
      # but something went wrong with the connection
      DNS_RESOLUTION[$K]="true"
      REACHABILITY[$K]="false"
    fi
  done
}
function check-containers {
  log "Checking for containers running..."
  # there is chance that crictl isn't available so we check first
  if exists "${KUBE_BIN}/crictl" ; then
    "${TIMEOUT[@]}" "${KUBE_BIN}/crictl" ps
  else
    log "crictl not found"
  fi
}
function check-systemd-services {
  log "Checking for kubernetes systemd services..."
  log 2 "systemctl list-units -all -t service 'kube*'"
  "${TIMEOUT[@]}" systemctl list-units -all -t service 'kube*'
}
function check-kubelet-logs {
  local JOURNAL_LINES=50
  log "Checking for Kubelet Logs(last ${JOURNAL_LINES} lines)"
  "${TIMEOUT[@]}" journalctl -u kubelet.service -n ${JOURNAL_LINES} --no-pager --utc
  KUBELET_LOG_LINES=$("${TIMEOUT[@]}" journalctl -u kubelet.service -n ${JOURNAL_LINES} --no-pager --utc | wc -l)
}
# Tries to connect to the master and records its status
function check-master-reachability {
  MASTER_STATUS["version"]="Unknown"
  MASTER_STATUS["healthz"]="Unknown"
  REACHABILITY[MASTER]="not tested"
  # check if the master address is available
  if [[ -z ${KUBERNETES_MASTER_NAME:-} ]]; then
     log "Can't connect to the master as the address is unknown. issues reading kube-env?"
     return 0
  fi
  # check the master version to include in the summary
  log "Trying to connect to the master: https://${KUBERNETES_MASTER_NAME}/version"
  if ! MASTER_STATUS["version"]=$(simple-curl "https://${KUBERNETES_MASTER_NAME}/version" "gitVersion"); then
    REACHABILITY[MASTER]="false"
    log "Failed connecting to ${KUBERNETES_MASTER_NAME}. Here is a breakdown of the connection time:"
    set +e # it's expected that curl will return something different than zero
    curl -w "${CURL_FORMAT}" -o /dev/null  --silent "${CURL_TIMEOUT[@]}" -v -k "https://${KUBERNETES_MASTER_NAME}/version"
    set -e # re-enable errexit
    return 0
  fi
  log "Connected successfully to ${KUBERNETES_MASTER_NAME} and retrieved master version"
  REACHABILITY[MASTER]="true"
  #since we were able to get the version try to read the healthz endpoint
  log "Trying to connect to the master: https://${KUBERNETES_MASTER_NAME}/healthz"
  if ! MASTER_STATUS["healthz"]=$(simple-curl "https://${KUBERNETES_MASTER_NAME}/healthz"); then
    log "Failed connecting to ${KUBERNETES_MASTER_NAME} to retrieve healthz status"
    return 0
  fi
  log "Connected successfully to ${KUBERNETES_MASTER_NAME} and retrieved master healthz status"
}
# Check if SA is enabled and working
function check-service-account {
  local HTTP_TOKEN_RESPONSE_CODE
  local HTTP_MONITORING_RESPONSE_CODE
  local curl_headers
  local testing_url
  log "Checking status of the service account"
  HTTP_TOKEN_RESPONSE_CODE=$(get-http-response 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token')
  # 000 or simply 0 means that curl could not reach the endpoint
  if [[ ${HTTP_TOKEN_RESPONSE_CODE} -eq 0 ]]; then
    log 2 "Could not reach metadata server to get the token."
    SA_WORKING="false"
    return 0
  fi
  # A 404 means there is no token and the SA is either disabled or deleted
  if [[ ${HTTP_TOKEN_RESPONSE_CODE} -eq 404 ]]; then
    SA_WORKING="false"
    log 2 "Could not get a token. SA likely either disabled or deleted "
    return 0
  fi
  # we expect a 200 for a valid token
  if [[ ! ${HTTP_TOKEN_RESPONSE_CODE} -eq 200 ]]; then
    log 2 "Failure getting token - skipping check."
    return 0
  fi
  # if a token is available we still need to test it in case the SA was recently disabled or deleted or simply doesn't have the permissions
  log 2 "Token available need to test it..."
  local numeric_project_id
  local random_string
  log 2 "Retrieving project number..."
  numeric_project_id=$(retrieve-metadata-entry "project/numeric-project-id")
  if [[ -z "${numeric_project_id}"  ]]; then
    log 4 "Could not retrieve project number from metadata server. Can't check service account."
    return 0
  fi
  random_string="rQvQYQ26o9aLbAQbkijuhy"
  # There is still a possibility that a token is returned but not usable, ie SA has been disabled recently.
  # The only way to check that is by trying to use it and check if it returns a 401 - Forbidden.
  log 2 "Retrieving token"
  curl_headers="Authorization: Bearer $(get-credentials)"
  log 2 "Testing token against https://monitoring.googleapis.com"
  testing_url="https://monitoring.googleapis.com/v1/projects/${numeric_project_id}/dashboards/gke-node-registration-checker-${random_string}"
  # we use get-http-response() but it would output the bearer token and we don't want that
  set +e
  HTTP_MONITORING_RESPONSE_CODE=$(time curl --silent ${curl_headers:+-H "${curl_headers}"} "${CURL_TIMEOUT[@]}" "${CURL_RETRY[@]}" ${CURL_RETRY_CONNREFUSED} --ipv4 -o /dev/null -w '%{http_code}' -H 'Metadata-Flavor: Google' "${testing_url}")
  set -e
  if [[ ${HTTP_MONITORING_RESPONSE_CODE} -eq 0 ]]; then
    log 2 "Failed connecting to monitoring.googleapis.com"
    return 0
  fi
  if [[ ${HTTP_MONITORING_RESPONSE_CODE} -eq 401 ]]; then
    log 2 "Permission denied to monitoring.googleapis.com when using the token"
    SA_WORKING="false"
    return 0
  fi
  # we don't expect the project to have a monitoring dashaboard named gke-node-registration-checker-${random_string} so a 404 is a good sign that the token is valid
  if [[ ${HTTP_MONITORING_RESPONSE_CODE} -eq 404 ]]; then
    log 2 "Valid Token"
    SA_WORKING="true"
    return 0
  fi
 log 2 "Could not confirm if the service account has the right permissions."
 log 2 "https://monitoring.googleapis.com returned HTTP CODE: ${HTTP_MONITORING_RESPONSE_CODE}"
}
### END of Checks
function summary-report {
  #printf formats
  local row_format="%-10s %-8s %-10s\n"
  local divider_format="%30s\n"
  local SA_NAME
  log "Retrieving service account e-mail..."
  SA_NAME=$(retrieve-metadata-entry  "instance/service-accounts/default/email")
  log 'Here is a summary of the checks performed:'
  # print summary table
  printf "${divider_format}" " " | tr ' ' '-'
  printf "${row_format}" "Service" "DNS" "Reachable"
  printf "${divider_format}" " " | tr ' ' '-'
  # print rows
  for K in "${!SERVICES_ENDPOINTS[@]}"; do
    printf "${row_format}" "${K}" "${DNS_RESOLUTION[$K]:-}" "${REACHABILITY[$K]:-}"
  done
  printf "${row_format}" "Master" "N/A" "${REACHABILITY[MASTER]:-Unknown}"
  printf "${divider_format}" " " | tr ' ' '-'
  echo
  printf "${row_format}" "Master" "Healthz" "Version"
  printf "${divider_format}" " " | tr ' ' '-'
  printf "${row_format}" "" "${MASTER_STATUS["healthz"]:-Unknown}" "${MASTER_STATUS["version"]:-Unknown}"
  printf "${divider_format}" " " | tr ' ' '-'
  echo "Service Account: ${SA_NAME:-Unknown} - enabled: ${SA_WORKING:-Not tested}"
  # journalctl always returns at least two lines, so 3 or more means logs are available for the unit
  if [[ ${KUBELET_LOG_LINES} -gt 2 ]]; then
    echo "Kubelet logs available: Yes(see above)"
  else
    echo "Kubelet logs available: No"
  fi
}
# Test if the node has registered with the API server based on kubelet healthz AND
# the node ready status reported by the K8s API server
function node-registered {
  set +e # we need the return codes
  local -r max_attempts=6
  local attempt=1
  log "Checking if the node is registered..."
  log 2 "Checking if Kubelet is healthy..."
  until curl -f --retry 0 "${CURL_TIMEOUT[@]}" --silent -o /dev/null http://localhost:10255/healthz?timeout=2s ; do
    if (( attempt == max_attempts )); then
      log 4 "Node not registered after ${attempt} attempts - Collecting information..."
      set -e # re-enable errexit
      return 1
    fi
    sleep "$(( 2 ** attempt++ ))"
  done
  log 4 "Kubelet is healthy."
  set -e # re-enable errexit
  if exists "${KUBE_BIN}/kubectl" ; then
    log 2 "Checking node status with the K8s API Server..."
    if [[  $(KUBECONFIG=/var/lib/kubelet/kubeconfig "${KUBE_BIN}/kubectl" --request-timeout=10s get node "$(hostname)" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}') == "True" ]]; then
      log 4 "Node ready and registered."
      return 0
    else
      log 4 "Not able to confirm if the node is ready - Collecting information..."
      return 1
    fi
  else
    log "kubectl not found"
    return 1
  fi
}
function main {
  # this script is installed and started as part of the kube-node-configuration.service(configure-helper.sh) which is preloaded in COS images
  # by the time this script is started kube-node-installation.service(configure.sh) has already ran but we will wait a bit before executing this checks.
  # 2 minutes is usually enough but shielded nodes seem to take a bit longer. 4 minutes is enough.
  # we don't want to check too early(things might not be ready) neither too late(auto-repair might kick in)
  readonly SLEEP_TIME="4m"
  log "Starting Node Registration Checker"
  log "Loading variables from kube-env"
  if [[ ! -e "${KUBE_HOME}/kube-env" ]]; then
    log "The ${KUBE_HOME}/kube-env file does not exist!! Exiting."
    exit 1
  fi
  source "${KUBE_HOME}/kube-env"
  log "Sleeping for ${SLEEP_TIME} to allow registration to complete "
  sleep "${SLEEP_TIME}"
  if ! node-registered ; then
    check-containers
    check-service-endpoint
    check-systemd-services
    check-kubelet-logs
    check-master-reachability
    check-service-account
    summary-report
  fi
  log "Completed running Node Registration Checker"
}

main
