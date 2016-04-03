#!/bin/sh

# This is a convenience script for reporting issues that include a base
# template of information. See https://github.com/docker/docker/pull/8845

set -e

DOCKER_ISSUE_URL=${DOCKER_ISSUE_URL:-"https://github.com/docker/docker/issues/new"}
DOCKER_ISSUE_NAME_PREFIX=${DOCKER_ISSUE_NAME_PREFIX:-"Report: "}
DOCKER=${DOCKER:-"docker"}
DOCKER_COMMAND="${DOCKER}"
export DOCKER_COMMAND

# pulled from https://gist.github.com/cdown/1163649
function urlencode() {
	# urlencode <string>

	local length="${#1}"
	for (( i = 0; i < length; i++ )); do
			local c="${1:i:1}"
			case $c in
					[a-zA-Z0-9.~_-]) printf "$c" ;;
					*) printf '%%%02X' "'$c"
			esac
	done
}

function template() {
# this should always match the template from CONTRIBUTING.md
	cat <<- EOM
	Description of problem:


	\`docker version\`:
	`${DOCKER_COMMAND} -D version`


	\`docker info\`:
	`${DOCKER_COMMAND} -D info`


	\`uname -a\`:
	`uname -a`


	Environment details (AWS, VirtualBox, physical, etc.):


	How reproducible:


	Steps to Reproduce:
	1.
	2.
	3.


	Actual Results:


	Expected Results:


	Additional info:


	EOM
}

function format_issue_url() {
	if [ ${#@} -ne 2 ] ; then
		return 1
	fi
	local issue_name=$(urlencode "${DOCKER_ISSUE_NAME_PREFIX}${1}")
	local issue_body=$(urlencode "${2}")
	echo "${DOCKER_ISSUE_URL}?title=${issue_name}&body=${issue_body}"
}


echo -ne "Do you use \`sudo\` to call docker? [y|N]: "
read -r -n 1 use_sudo
echo ""

if [ "x${use_sudo}" = "xy" -o "x${use_sudo}" = "xY" ]; then
	export DOCKER_COMMAND="sudo ${DOCKER}"
fi

echo -ne "Title of new issue?: "
read -r issue_title
echo ""

issue_url=$(format_issue_url "${issue_title}" "$(template)")

if which xdg-open 2>/dev/null >/dev/null ; then
	echo -ne "Would like to launch this report in your browser? [Y|n]: "
	read -r -n 1 launch_now
	echo ""

	if [ "${launch_now}" != "n" -a "${launch_now}" != "N" ]; then
		xdg-open "${issue_url}"
	fi
fi

echo "If you would like to manually open the url, you can open this link if your browser: ${issue_url}"

