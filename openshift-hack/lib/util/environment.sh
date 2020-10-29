#!/usr/bin/env bash

# This script holds library functions for setting up the shell environment for OpenShift scripts

# os::util::environment::use_sudo updates $USE_SUDO to be 'true', so that later scripts choosing between
# execution using 'sudo' and execution without it chose to use 'sudo'
#
# Globals:
#  None
# Arguments:
#  None
# Returns:
#  - export USE_SUDO
function os::util::environment::use_sudo() {
    USE_SUDO=true
    export USE_SUDO
}
readonly -f os::util::environment::use_sudo

# os::util::environment::setup_time_vars sets up environment variables that describe durations of time
# These variables can be used to specify times for other utility functions
#
# Globals:
#  None
# Arguments:
#  None
# Returns:
#  - export TIME_MS
#  - export TIME_SEC
#  - export TIME_MIN
function os::util::environment::setup_time_vars() {
    TIME_MS=1
    export TIME_MS
    TIME_SEC="$(( 1000  * TIME_MS ))"
    export TIME_SEC
    TIME_MIN="$(( 60 * TIME_SEC ))"
    export TIME_MIN
}
readonly -f os::util::environment::setup_time_vars

# os::util::environment::setup_all_server_vars sets up all environment variables necessary to configure and start an OpenShift server
#
# Globals:
#  - OS_ROOT
#  - PATH
#  - TMPDIR
#  - LOG_DIR
#  - ARTIFACT_DIR
#  - KUBELET_SCHEME
#  - KUBELET_BIND_HOST
#  - KUBELET_HOST
#  - KUBELET_PORT
#  - BASETMPDIR
#  - ETCD_PORT
#  - ETCD_PEER_PORT
#  - API_BIND_HOST
#  - API_HOST
#  - API_PORT
#  - API_SCHEME
#  - PUBLIC_MASTER_HOST
#  - USE_IMAGES
# Arguments:
#  - 1: the path under the root temporary directory for OpenShift where these subdirectories should be made
# Returns:
#  - export PATH
#  - export BASETMPDIR
#  - export LOG_DIR
#  - export VOLUME_DIR
#  - export ARTIFACT_DIR
#  - export FAKE_HOME_DIR
#  - export KUBELET_SCHEME
#  - export KUBELET_BIND_HOST
#  - export KUBELET_HOST
#  - export KUBELET_PORT
#  - export ETCD_PORT
#  - export ETCD_PEER_PORT
#  - export ETCD_DATA_DIR
#  - export API_BIND_HOST
#  - export API_HOST
#  - export API_PORT
#  - export API_SCHEME
#  - export SERVER_CONFIG_DIR
#  - export MASTER_CONFIG_DIR
#  - export NODE_CONFIG_DIR
#  - export USE_IMAGES
#  - export TAG
function os::util::environment::setup_all_server_vars() {
    os::util::environment::setup_kubelet_vars
    os::util::environment::setup_etcd_vars
    os::util::environment::setup_server_vars
    os::util::environment::setup_images_vars
}
readonly -f os::util::environment::setup_all_server_vars

# os::util::environment::update_path_var updates $PATH so that OpenShift binaries are available
#
# Globals:
#  - OS_ROOT
#  - PATH
# Arguments:
#  None
# Returns:
#  - export PATH
function os::util::environment::update_path_var() {
    local prefix
    if os::util::find::system_binary 'go' >/dev/null 2>&1; then
        prefix+="${OS_OUTPUT_BINPATH}/$(os::build::host_platform):"
    fi
    if [[ -n "${GOPATH:-}" ]]; then
        prefix+="${GOPATH}/bin:"
    fi

    PATH="${prefix:-}${PATH}"
    export PATH
}
readonly -f os::util::environment::update_path_var

# os::util::environment::setup_tmpdir_vars sets up temporary directory path variables
#
# Globals:
#  - TMPDIR
# Arguments:
#  - 1: the path under the root temporary directory for OpenShift where these subdirectories should be made
# Returns:
#  - export BASETMPDIR
#  - export BASEOUTDIR
#  - export LOG_DIR
#  - export VOLUME_DIR
#  - export ARTIFACT_DIR
#  - export FAKE_HOME_DIR
#  - export OS_TMP_ENV_SET
function os::util::environment::setup_tmpdir_vars() {
    local sub_dir=$1

    BASETMPDIR="${TMPDIR:-/tmp}/openshift/${sub_dir}"
    export BASETMPDIR
    VOLUME_DIR="${BASETMPDIR}/volumes"
    export VOLUME_DIR

    BASEOUTDIR="${OS_OUTPUT_SCRIPTPATH}/${sub_dir}"
    export BASEOUTDIR
    LOG_DIR="${ARTIFACT_DIR:-${BASEOUTDIR}}/logs"
    export LOG_DIR
    ARTIFACT_DIR="${ARTIFACT_DIR:-${BASEOUTDIR}/artifacts}"
    export ARTIFACT_DIR
    FAKE_HOME_DIR="${BASEOUTDIR}/openshift.local.home"
    export FAKE_HOME_DIR

    mkdir -p "${LOG_DIR}" "${VOLUME_DIR}" "${ARTIFACT_DIR}" "${FAKE_HOME_DIR}"

    export OS_TMP_ENV_SET="${sub_dir}"
}
readonly -f os::util::environment::setup_tmpdir_vars

# os::util::environment::setup_kubelet_vars sets up environment variables necessary for interacting with the kubelet
#
# Globals:
#  - KUBELET_SCHEME
#  - KUBELET_BIND_HOST
#  - KUBELET_HOST
#  - KUBELET_PORT
# Arguments:
#  None
# Returns:
#  - export KUBELET_SCHEME
#  - export KUBELET_BIND_HOST
#  - export KUBELET_HOST
#  - export KUBELET_PORT
function os::util::environment::setup_kubelet_vars() {
    KUBELET_SCHEME="${KUBELET_SCHEME:-https}"
    export KUBELET_SCHEME
    KUBELET_BIND_HOST="${KUBELET_BIND_HOST:-127.0.0.1}"
    export KUBELET_BIND_HOST
    KUBELET_HOST="${KUBELET_HOST:-${KUBELET_BIND_HOST}}"
    export KUBELET_HOST
    KUBELET_PORT="${KUBELET_PORT:-10250}"
    export KUBELET_PORT
}
readonly -f os::util::environment::setup_kubelet_vars

# os::util::environment::setup_etcd_vars sets up environment variables necessary for interacting with etcd
#
# Globals:
#  - BASETMPDIR
#  - ETCD_HOST
#  - ETCD_PORT
#  - ETCD_PEER_PORT
# Arguments:
#  None
# Returns:
#  - export ETCD_HOST
#  - export ETCD_PORT
#  - export ETCD_PEER_PORT
#  - export ETCD_DATA_DIR
function os::util::environment::setup_etcd_vars() {
    ETCD_HOST="${ETCD_HOST:-127.0.0.1}"
    export ETCD_HOST
    ETCD_PORT="${ETCD_PORT:-4001}"
    export ETCD_PORT
    ETCD_PEER_PORT="${ETCD_PEER_PORT:-7001}"
    export ETCD_PEER_PORT

    ETCD_DATA_DIR="${BASETMPDIR}/etcd"
    export ETCD_DATA_DIR

    mkdir -p "${ETCD_DATA_DIR}"
}
readonly -f os::util::environment::setup_etcd_vars

# os::util::environment::setup_server_vars sets up environment variables necessary for interacting with the server
#
# Globals:
#  - BASETMPDIR
#  - KUBELET_HOST
#  - API_BIND_HOST
#  - API_HOST
#  - API_PORT
#  - API_SCHEME
#  - PUBLIC_MASTER_HOST
# Arguments:
#  None
# Returns:
#  - export API_BIND_HOST
#  - export API_HOST
#  - export API_PORT
#  - export API_SCHEME
#  - export SERVER_CONFIG_DIR
#  - export MASTER_CONFIG_DIR
#  - export NODE_CONFIG_DIR
function os::util::environment::setup_server_vars() {
    # turn on cache mutation detector every time we start a server
    KUBE_CACHE_MUTATION_DETECTOR="${KUBE_CACHE_MUTATION_DETECTOR:-true}"
    export KUBE_CACHE_MUTATION_DETECTOR

    API_BIND_HOST="${API_BIND_HOST:-127.0.0.1}"
    export API_BIND_HOST
    API_HOST="${API_HOST:-${API_BIND_HOST}}"
    export API_HOST
    API_PORT="${API_PORT:-8443}"
    export API_PORT
    API_SCHEME="${API_SCHEME:-https}"
    export API_SCHEME

    MASTER_ADDR="${API_SCHEME}://${API_HOST}:${API_PORT}"
    export MASTER_ADDR
    PUBLIC_MASTER_HOST="${PUBLIC_MASTER_HOST:-${API_HOST}}"
    export PUBLIC_MASTER_HOST

    SERVER_CONFIG_DIR="${BASETMPDIR}/openshift.local.config"
    export SERVER_CONFIG_DIR
    MASTER_CONFIG_DIR="${SERVER_CONFIG_DIR}/master"
    export MASTER_CONFIG_DIR
    NODE_CONFIG_DIR="${SERVER_CONFIG_DIR}/node-${KUBELET_HOST}"
    export NODE_CONFIG_DIR

    ETCD_CLIENT_CERT="${MASTER_CONFIG_DIR}/master.etcd-client.crt"
    export ETCD_CLIENT_CERT
    ETCD_CLIENT_KEY="${MASTER_CONFIG_DIR}/master.etcd-client.key"
    export ETCD_CLIENT_KEY
    ETCD_CA_BUNDLE="${MASTER_CONFIG_DIR}/ca-bundle.crt"
    export ETCD_CA_BUNDLE

    mkdir -p "${SERVER_CONFIG_DIR}" "${MASTER_CONFIG_DIR}" "${NODE_CONFIG_DIR}"
}
readonly -f os::util::environment::setup_server_vars

# os::util::environment::setup_images_vars sets up environment variables necessary for interacting with release images
#
# Globals:
#  - OS_ROOT
#  - USE_IMAGES
# Arguments:
#  None
# Returns:
#  - export USE_IMAGES
#  - export TAG
#  - export MAX_IMAGES_BULK_IMPORTED_PER_REPOSITORY
function os::util::environment::setup_images_vars() {
    # Use either the latest release built images, or latest.
    IMAGE_PREFIX="${OS_IMAGE_PREFIX:-"openshift/origin"}"
    if [[ -z "${USE_IMAGES-}" ]]; then
        TAG='latest'
        export TAG
        USE_IMAGES="${IMAGE_PREFIX}-\${component}:latest"
        export USE_IMAGES

        if [[ -e "${OS_ROOT}/_output/local/releases/.commit" ]]; then
            TAG="$(cat "${OS_ROOT}/_output/local/releases/.commit")"
            export TAG
            USE_IMAGES="${IMAGE_PREFIX}-\${component}:${TAG}"
            export USE_IMAGES
        fi
    fi
	export MAX_IMAGES_BULK_IMPORTED_PER_REPOSITORY="${MAX_IMAGES_BULK_IMPORTED_PER_REPOSITORY:-3}"
}
readonly -f os::util::environment::setup_images_vars
