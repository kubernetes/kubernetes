#!/usr/bin/env bash
#
# This library holds miscellaneous utility functions. If there begin to be groups of functions in this
# file that share intent or are thematically similar, they should be split into their own files.

# os::util::describe_return_code describes an exit code
#
# Globals:
#  - OS_SCRIPT_START_TIME
# Arguments:
#  - 1: exit code to describe
# Returns:
#  None
function os::util::describe_return_code() {
	local return_code=$1
	local message
    message="$( os::util::repository_relative_path "$0" ) exited with code ${return_code} "

	if [[ -n "${OS_SCRIPT_START_TIME:-}" ]]; then
		local end_time
        end_time="$(date +%s)"
		local elapsed_time
        elapsed_time="$(( end_time - OS_SCRIPT_START_TIME ))"
		local formatted_time
        formatted_time="$( os::util::format_seconds "${elapsed_time}" )"
		message+="after ${formatted_time}"
	fi

	if [[ "${return_code}" = "0" ]]; then
		os::log::info "${message}"
	else
		os::log::error "${message}"
	fi
}
readonly -f os::util::describe_return_code

# os::util::install_describe_return_code installs the return code describer for the EXIT trap
# If the EXIT trap is not initialized, installing this plugin will initialize it.
#
# Globals:
#  None
# Arguments:
#  None
# Returns:
#  - export OS_DESCRIBE_RETURN_CODE
#  - export OS_SCRIPT_START_TIME
function os::util::install_describe_return_code() {
	export OS_DESCRIBE_RETURN_CODE="true"
	OS_SCRIPT_START_TIME="$( date +%s )"; export OS_SCRIPT_START_TIME
	os::util::trap::init_exit
}
readonly -f os::util::install_describe_return_code

# OS_ORIGINAL_WD is the original working directory the script sourcing this utility file was called
# from. This is an important directory as if $0 is a relative path, we cannot use the following path
# utility without knowing from where $0 is relative.
if [[ -z "${OS_ORIGINAL_WD:-}" ]]; then
	# since this could be sourced in a context where the utilities are already loaded,
	# we want to ensure that this is re-entrant, so we only set $OS_ORIGINAL_WD if it
	# is not set already
	OS_ORIGINAL_WD="$( pwd )"
	readonly OS_ORIGINAL_WD
	export OS_ORIGINAL_WD
fi

# os::util::repository_relative_path returns the relative path from the $OS_ROOT directory to the
# given file, if the file is inside of the $OS_ROOT directory. If the file is outside of $OS_ROOT,
# this function will return the absolute path to the file
#
# Globals:
#  - OS_ROOT
# Arguments:
#  - 1: the path to relativize
# Returns:
#  None
function os::util::repository_relative_path() {
	local filename=$1
	local directory; directory="$( dirname "${filename}" )"
	filename="$( basename "${filename}" )"

	if [[ "${directory}" != "${OS_ROOT}"* ]]; then
		pushd "${OS_ORIGINAL_WD}" >/dev/null 2>&1 || exit 1
		directory="$( os::util::absolute_path "${directory}" )"
		popd >/dev/null 2>&1 || exit 1
	fi

	directory="${directory##*${OS_ROOT}/}"

	echo "${directory}/${filename}"
}
readonly -f os::util::repository_relative_path

# os::util::format_seconds formats a duration of time in seconds to print in HHh MMm SSs
#
# Globals:
#  None
# Arguments:
#  - 1: time in seconds to format
# Return:
#  None
function os::util::format_seconds() {
	local raw_seconds=$1

	local hours minutes seconds
	(( hours=raw_seconds/3600 ))
	(( minutes=(raw_seconds%3600)/60 ))
	(( seconds=raw_seconds%60 ))

	printf '%02dh %02dm %02ds' "${hours}" "${minutes}" "${seconds}"
}
readonly -f os::util::format_seconds

# os::util::sed attempts to make our Bash scripts agnostic to the platform
# on which they run `sed` by glossing over a discrepancy in flag use in GNU.
#
# Globals:
#  None
# Arguments:
#  - all: arguments to pass to `sed -i`
# Return:
#  None
function os::util::sed() {
	local sudo="${USE_SUDO:+sudo}"
	if LANG=C sed --help 2>&1 | grep -q "GNU sed"; then
		${sudo} sed -i'' "$@"
	else
		${sudo} sed -i '' "$@"
	fi
}
readonly -f os::util::sed

# os::util::base64decode attempts to make our Bash scripts agnostic to the platform
# on which they run `base64decode` by glossing over a discrepancy in flag use in GNU.
#
# Globals:
#  None
# Arguments:
#  - all: arguments to pass to `base64decode`
# Return:
#  None
function os::util::base64decode() {
	if [[ "$(go env GOHOSTOS)" == "darwin" ]]; then
		base64 -D "$@"
	else
		base64 -d "$@"
	fi
}
readonly -f os::util::base64decode

# os::util::curl_etcd sends a request to the backing etcd store for the master.
# We use the administrative client cert and key for access and re-encode them
# as necessary for OSX clients.
#
# Globals:
#  MASTER_CONFIG_DIR
#  API_SCHEME
#  API_HOST
#  ETCD_PORT
# Arguments:
#  - 1: etcd-relative URL to curl, with leading slash
# Returns:
#  None
function os::util::curl_etcd() {
	local url="$1"
	local full_url="${API_SCHEME}://${API_HOST}:${ETCD_PORT}${url}"

	local etcd_client_cert="${MASTER_CONFIG_DIR}/master.etcd-client.crt"
	local etcd_client_key="${MASTER_CONFIG_DIR}/master.etcd-client.key"
	local ca_bundle="${MASTER_CONFIG_DIR}/ca-bundle.crt"

	if curl -V | grep -q 'SecureTransport'; then
		# on newer OSX `curl` implementations, SSL is not used and client certs
		# and keys are expected to be encoded in P12 format instead of PEM format,
		# so we need to convert the secrets that the server wrote if we haven't
		# already done so
		local etcd_client_cert_p12="${MASTER_CONFIG_DIR}/master.etcd-client.crt.p12"
		local etcd_client_cert_p12_password="${CURL_CERT_P12_PASSWORD:-'password'}"
		if [[ ! -f "${etcd_client_cert_p12}" ]]; then
			openssl pkcs12 -export                        \
			               -in "${etcd_client_cert}"      \
			               -inkey "${etcd_client_key}"    \
			               -out "${etcd_client_cert_p12}" \
			               -password "pass:${etcd_client_cert_p12_password}"
		fi

		curl --fail --silent --cacert "${ca_bundle}" \
		     --cert "${etcd_client_cert_p12}:${etcd_client_cert_p12_password}" "${full_url}"
	else
		curl --fail --silent --cacert "${ca_bundle}" \
		     --cert "${etcd_client_cert}" --key "${etcd_client_key}" "${full_url}"
	fi
}

# os::util::ensure_tmpfs ensures that the target dir is mounted on tmpfs
#
# Globals:
#  OS_TMPFS_REQUIRED
# Arguments:
#  - 1: target to check
# Returns:
#  None
function os::util::ensure_tmpfs() {
	if [[ -z "${OS_TMPFS_REQUIRED:-}" ]]; then
		return 0
	fi

	local target="$1"
	if [[ ! -d "${target}" ]]; then
		os::log::fatal "Target dir ${target} does not exist, cannot perform fstype check."
	fi

	os::log::debug "Filesystem information:
$( df -h -T )"

	os::log::debug "Mount information:
$( findmnt --all )"

	local fstype
	fstype="$( df --output=fstype "${target}" | tail -n 1 )"
	if [[ "${fstype}" != "tmpfs" ]]; then
		local message="Expected \`${target}\` to be mounted on \`tmpfs\` but found \`${fstype}\` instead."
		os::log::fatal "${message}"
	fi
}
