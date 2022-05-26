#!/usr/bin/env bash
# plsdo.sh, version 0.1.1
set -o errexit -o nounset -o pipefail

_plsdo_run() {
    case "${1-}" in
    ""|help)
        _plsdo_help "${2-}" ;;
    *)
        "$@" ;;
    esac
}

declare -A help

help[list]="Print the list of tasks"
list() {
    declare -F | awk '{print $3}' | grep -v '^_'
}

_plsdo_help_task_name_width="${_plsdo_help_task_name_width:-12}"

_plsdo_help() {
    local topic="${1-}"
    # print help for the topic
    if [ -n "$topic" ]; then
        if ! command -v "$topic" > /dev/null ; then
            _plsdo_error "No such task: $topic"
            return 1
        fi

        printf "\nUsage:\n  %s %s\n\n%s\n" "$0" "$topic" "${help[$topic]-}"
        return 0
    fi

    # print list of tasks and their help line.
    [ -n "${banner-}" ] && echo "$banner" && echo
    for i in $(list); do
        printf "%-${_plsdo_help_task_name_width}s\t%s\n" "$i" "${help[$i]-}" | head -1
    done
}

_plsdo_error() {
    >&2 echo "$@"
}

# shellcheck disable=SC2016
help[_plsdo_completion]='Print tab completion for $SHELL.

Redirect the output to a file that will be run when the shell starts,
such as ~/.bashrc.

    $ ./do _pldsdo_completion >> ~/.bash_complete/do
'
_plsdo_completion() {
    local shell; shell="$(basename "$SHELL" 2> /dev/null)"
    case "$shell" in
    bash)
        cat <<+++
_dotslashdo_completions() {
    if ! command -v $0 > /dev/null; then return; fi
    if [ "\${#COMP_WORDS[@]}" != "2" ]; then return; fi
    COMPREPLY=(\$(compgen -W "\$($0 list)" "\${COMP_WORDS[1]}"))
}
complete -F _dotslashdo_completions $0
+++
        ;;
    "")
        _plsdo_error "Set \$SHELL to select tab completion."
        return 1 ;;
    *)
        _plsdo_error "No completetion for shell: $shell"
        return 1 ;;
    esac
}
