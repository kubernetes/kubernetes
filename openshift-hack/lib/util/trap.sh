#!/usr/bin/env bash
#
# This library defines the trap handlers for the ERR and EXIT signals. Any new handler for these signals
# must be added to these handlers and activated by the environment variable mechanism that the rest use.
# These functions ensure that no handler can ever alter the exit code that was emitted by a command
# in a test script.

# os::util::trap::init_err initializes the privileged handler for the ERR signal if it hasn't
# been registered already. This will overwrite any other handlers registered on the signal.
#
# Globals:
#  None
# Arguments:
#  None
# Returns:
#  None
function os::util::trap::init_err() {
    if ! trap -p ERR | grep -q 'os::util::trap::err_handler'; then
        trap 'os::util::trap::err_handler;' ERR
    fi
}
readonly -f os::util::trap::init_err

# os::util::trap::init_exit initializes the privileged handler for the EXIT signal if it hasn't
# been registered already. This will overwrite any other handlers registered on the signal.
#
# Globals:
#  None
# Arguments:
#  None
# Returns:
#  None
function os::util::trap::init_exit() {
    if ! trap -p EXIT | grep -q 'os::util::trap::exit_handler'; then
        trap 'os::util::trap::exit_handler;' EXIT
    fi
}
readonly -f os::util::trap::init_exit

# os::util::trap::err_handler is the handler for the ERR signal.
#
# Globals:
#  - OS_TRAP_DEBUG
#  - OS_USE_STACKTRACE
# Arguments:
#  None
# Returns:
#  - returns original return code, allows privileged handler to exit if necessary
function os::util::trap::err_handler() {
    local -r return_code=$?
    local -r last_command="${BASH_COMMAND}"

    if set +o | grep -q '\-o errexit'; then
        local -r errexit_set=true
    fi

    if [[ "${OS_TRAP_DEBUG:-}" = "true" ]]; then
        echo "[DEBUG] Error handler executing with return code \`${return_code}\`, last command \`${last_command}\`, and errexit set \`${errexit_set:-}\`"
    fi

    if [[ "${OS_USE_STACKTRACE:-}" = "true" ]]; then
        # the OpenShift stacktrace function is treated as a privileged handler for this signal
        # and is therefore allowed to run outside of a subshell in order to allow it to `exit`
        # if necessary
        os::log::stacktrace::print "${return_code}" "${last_command}" "${errexit_set:-}"
    fi

    return "${return_code}"
}
readonly -f os::util::trap::err_handler

# os::util::trap::exit_handler is the handler for the EXIT signal.
#
# Globals:
#  - OS_TRAP_DEBUG
#  - OS_DESCRIBE_RETURN_CODE
# Arguments:
#  None
# Returns:
#  - original exit code of the script that exited
function os::util::trap::exit_handler() {
    local -r return_code=$?

    # we do not want these traps to be able to trigger more errors, we can let them fail silently
    set +o errexit

    if [[ "${OS_TRAP_DEBUG:-}" = "true" ]]; then
        echo "[DEBUG] Exit handler executing with return code \`${return_code}\`"
    fi

    # the following envars selectively enable optional exit traps, all of which are run inside of
    # a subshell in order to sandbox them and not allow them to influence how this script will exit
    if [[ "${OS_DESCRIBE_RETURN_CODE:-}" = "true" ]]; then
        ( os::util::describe_return_code "${return_code}" )
    fi

    exit "${return_code}"
}
readonly -f os::util::trap::exit_handler
