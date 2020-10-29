#!/usr/bin/env bash

# This file contains functions used for writing log messages
# to stdout and stderr from scripts while they run.

# os::log::info writes the message to stdout.
#
# Arguments:
#  - all: message to write
function os::log::info() {
	local message; message="$( os::log::internal::prefix_lines "[INFO]" "$*" )"
	os::log::internal::to_logfile "${message}"
	echo "${message}"
}
readonly -f os::log::info

# os::log::warning writes the message to stderr.
# A warning indicates something went wrong but
# not so wrong that we cannot recover.
#
# Arguments:
#  - all: message to write
function os::log::warning() {
	local message; message="$( os::log::internal::prefix_lines "[WARNING]" "$*" )"
	os::log::internal::to_logfile "${message}"
	os::text::print_yellow "${message}" 1>&2
}
readonly -f os::log::warning

# os::log::error writes the message to stderr.
# An error indicates that something went wrong
# and we will most likely fail after this.
#
# Arguments:
#  - all: message to write
function os::log::error() {
	local message; message="$( os::log::internal::prefix_lines "[ERROR]" "$*" )"
	os::log::internal::to_logfile "${message}"
	os::text::print_red "${message}" 1>&2
}
readonly -f os::log::error

# os::log::fatal writes the message to stderr and
# returns a non-zero code to force a process exit.
# A fatal error indicates that there is no chance
# of recovery.
#
# Arguments:
#  - all: message to write
function os::log::fatal() {
	local message; message="$( os::log::internal::prefix_lines "[FATAL]" "$*" )"
	os::log::internal::to_logfile "${message}"
	os::text::print_red "${message}" 1>&2
	exit 1
}
readonly -f os::log::fatal

# os::log::debug writes the message to stderr if
# the ${OS_DEBUG} variable is set.
#
# Globals:
#  - OS_DEBUG
# Arguments:
#  - all: message to write
function os::log::debug() {
	local message; message="$( os::log::internal::prefix_lines "[DEBUG]" "$*" )"
	os::log::internal::to_logfile "${message}"
	if [[ -n "${OS_DEBUG:-}" ]]; then
		os::text::print_blue "${message}" 1>&2
	fi
}
readonly -f os::log::debug

# os::log::internal::to_logfile makes a best-effort
# attempt to write the message to the script logfile
#
# Globals:
#  - LOG_DIR
# Arguments:
#  - all: message to write
function os::log::internal::to_logfile() {
	if [[ -n "${LOG_DIR:-}" && -d "${LOG_DIR-}" ]]; then
		echo "$*" >>"${LOG_DIR}/scripts.log"
	fi
}

# os::log::internal::prefix_lines prints out the
# original content with the given prefix at the
# start of every line.
#
# Arguments:
#  - 1: prefix for lines
#  - 2: content to prefix
function os::log::internal::prefix_lines() {
	local prefix="$1"
	local content="$2"

	local old_ifs="${IFS}"
	IFS=$'\n'
	for line in ${content}; do
		echo "${prefix} ${line}"
	done
	IFS="${old_ifs}"
}