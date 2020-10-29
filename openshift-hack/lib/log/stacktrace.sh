#!/usr/bin/env bash
#
# This library contains an implementation of a stack trace for Bash scripts.

# os::log::stacktrace::install installs the stacktrace as a handler for the ERR signal if one
# has not already been installed and sets `set -o errtrace` in order to propagate the handler
# If the ERR trap is not initialized, installing this plugin will initialize it.
#
# Globals:
#  None
# Arguments:
#  None
# Returns:
#  - export OS_USE_STACKTRACE
function os::log::stacktrace::install() {
    # setting 'errtrace' propagates our ERR handler to functions, expansions and subshells
    set -o errtrace

    # OS_USE_STACKTRACE is read by os::util::trap at runtime to request a stacktrace
    export OS_USE_STACKTRACE=true

    os::util::trap::init_err
}
readonly -f os::log::stacktrace::install

# os::log::stacktrace::print prints the stacktrace and exits with the return code from the script that
# called for a stack trace. This function will always return 0 if it is not handling the signal, and if it
# is handling the signal, this function will always `exit`, not return, the return code it receives as
# its first argument.
#
# Globals:
#  - BASH_SOURCE
#  - BASH_LINENO
#  - FUNCNAME
# Arguments:
#  - 1: the return code of the command in the script that generated the ERR signal
#  - 2: the last command that ran before handlers were invoked
#  - 3: whether or not `set -o errexit` was set in the script that generated the ERR signal
# Returns:
#  None
function os::log::stacktrace::print() {
    local return_code=$1
    local last_command=$2
    local errexit_set=${3:-}

    if [[ "${return_code}" = "0" ]]; then
        # we're not supposed to respond when no error has occurred
        return 0
    fi

    if [[ -z "${errexit_set}" ]]; then
        # if errexit wasn't set in the shell when the ERR signal was issued, then we can ignore the signal
        # as this is not cause for failure
        return 0
    fi

    # dump the entire stack for debugging purposes
    os::log::debug "$( os::util::repository_relative_path "${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" )"
    for (( i = 0; i < ${#BASH_LINENO[@]}; i++ )); do
        os::log::debug "$( os::util::repository_relative_path "${BASH_SOURCE[$i+1]:-"$( os::util::repository_relative_path "$0" )"}" ):${BASH_LINENO[$i]}: ${FUNCNAME[$i]}"
    done

    # iterate backwards through the stack until we leave library files, so we can be sure we start logging
    # actual script code and not this handler's call
    local stack_begin_index
    for (( stack_begin_index = 0; stack_begin_index < ${#BASH_SOURCE[@]}; stack_begin_index++ )); do
        if [[ ! "${BASH_SOURCE[${stack_begin_index}]}" =~ hack/lib/(log/stacktrace|util/trap)\.sh ]]; then
            break
        fi
    done

    local preamble_finished
    local stack_index=1
    local i
    for (( i = stack_begin_index; i < ${#BASH_SOURCE[@]}; i++ )); do
        local bash_source
        bash_source="$( os::util::repository_relative_path "${BASH_SOURCE[$i]}" )"
        if [[ -z "${preamble_finished:-}" ]]; then
            preamble_finished=true
            os::log::error "${bash_source}:${BASH_LINENO[$i-1]}: \`${last_command}\` exited with status ${return_code}." >&2
            exit "${return_code}"
        fi
        stack_index=$(( stack_index + 1 ))
    done

    # we know we're the privileged handler in this chain, so we can safely exit the shell without
    # starving another handler of the privilege of reacting to this signal
    os::log::info "  Exiting with code ${return_code}." >&2
    exit "${return_code}"
}
readonly -f os::log::stacktrace::print
