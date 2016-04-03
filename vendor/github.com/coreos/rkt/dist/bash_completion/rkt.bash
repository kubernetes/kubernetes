#!/bin/bash

__debug()
{
    if [[ -n ${BASH_COMP_DEBUG_FILE} ]]; then
        echo "$*" >> "${BASH_COMP_DEBUG_FILE}"
    fi
}

# Homebrew on Macs have version 1.3 of bash-completion which doesn't include
# _init_completion. This is a very minimal version of that function.
__my_init_completion()
{
    COMPREPLY=()
    _get_comp_words_by_ref cur prev words cword
}

__index_of_word()
{
    local w word=$1
    shift
    index=0
    for w in "$@"; do
        [[ $w = "$word" ]] && return
        index=$((index+1))
    done
    index=-1
}

__contains_word()
{
    local w word=$1; shift
    for w in "$@"; do
        [[ $w = "$word" ]] && return
    done
    return 1
}

__handle_reply()
{
    __debug "${FUNCNAME}"
    case $cur in
        -*)
            if [[ $(type -t compopt) = "builtin" ]]; then
                compopt -o nospace
            fi
            local allflags
            if [ ${#must_have_one_flag[@]} -ne 0 ]; then
                allflags=("${must_have_one_flag[@]}")
            else
                allflags=("${flags[*]} ${two_word_flags[*]}")
            fi
            COMPREPLY=( $(compgen -W "${allflags[*]}" -- "$cur") )
            if [[ $(type -t compopt) = "builtin" ]]; then
                [[ $COMPREPLY == *= ]] || compopt +o nospace
            fi
            return 0;
            ;;
    esac

    # check if we are handling a flag with special work handling
    local index
    __index_of_word "${prev}" "${flags_with_completion[@]}"
    if [[ ${index} -ge 0 ]]; then
        ${flags_completion[${index}]}
        return
    fi

    # we are parsing a flag and don't have a special handler, no completion
    if [[ ${cur} != "${words[cword]}" ]]; then
        return
    fi

    local completions
    if [[ ${#must_have_one_flag[@]} -ne 0 ]]; then
        completions=("${must_have_one_flag[@]}")
    elif [[ ${#must_have_one_noun[@]} -ne 0 ]]; then
        completions=("${must_have_one_noun[@]}")
    else
        completions=("${commands[@]}")
    fi
    COMPREPLY=( $(compgen -W "${completions[*]}" -- "$cur") )

    if [[ ${#COMPREPLY[@]} -eq 0 ]]; then
        declare -F __custom_func >/dev/null && __custom_func
    fi
}

# The arguments should be in the form "ext1|ext2|extn"
__handle_filename_extension_flag()
{
    local ext="$1"
    _filedir "@(${ext})"
}

__handle_subdirs_in_dir_flag()
{
    local dir="$1"
    pushd "${dir}" >/dev/null 2>&1 && _filedir -d && popd >/dev/null 2>&1
}

__handle_flag()
{
    __debug "${FUNCNAME}: c is $c words[c] is ${words[c]}"

    # if a command required a flag, and we found it, unset must_have_one_flag()
    local flagname=${words[c]}
    # if the word contained an =
    if [[ ${words[c]} == *"="* ]]; then
        flagname=${flagname%=*} # strip everything after the =
        flagname="${flagname}=" # but put the = back
    fi
    __debug "${FUNCNAME}: looking for ${flagname}"
    if __contains_word "${flagname}" "${must_have_one_flag[@]}"; then
        must_have_one_flag=()
    fi

    # skip the argument to a two word flag
    if __contains_word "${words[c]}" "${two_word_flags[@]}"; then
        c=$((c+1))
        # if we are looking for a flags value, don't show commands
        if [[ $c -eq $cword ]]; then
            commands=()
        fi
    fi

    # skip the flag itself
    c=$((c+1))

}

__handle_noun()
{
    __debug "${FUNCNAME}: c is $c words[c] is ${words[c]}"

    if __contains_word "${words[c]}" "${must_have_one_noun[@]}"; then
        must_have_one_noun=()
    fi

    nouns+=("${words[c]}")
    c=$((c+1))
}

__handle_command()
{
    __debug "${FUNCNAME}: c is $c words[c] is ${words[c]}"

    local next_command
    if [[ -n ${last_command} ]]; then
        next_command="_${last_command}_${words[c]}"
    else
        next_command="_${words[c]}"
    fi
    c=$((c+1))
    __debug "${FUNCNAME}: looking for ${next_command}"
    declare -F $next_command >/dev/null && $next_command
}

__handle_word()
{
    if [[ $c -ge $cword ]]; then
        __handle_reply
        return
    fi
    __debug "${FUNCNAME}: c is $c words[c] is ${words[c]}"
    if [[ "${words[c]}" == -* ]]; then
        __handle_flag
    elif __contains_word "${words[c]}" "${commands[@]}"; then
        __handle_command
    else
        __handle_noun
    fi
    __handle_word
}

__rkt_parse_image()
{
    local rkt_output
    if rkt_output=$(rkt image list --no-legend 2>/dev/null); then
        out=($(echo "${rkt_output}" | awk '{print $1}'))
        COMPREPLY=( $( compgen -W "${out[*]}" -- "$cur" ) )
    fi
}

__rkt_parse_list()
{
    local rkt_output
    if rkt_output=$(rkt list --no-legend 2>/dev/null); then
        if [[ -n "$1" ]]; then
            out=($(echo "${rkt_output}" | grep ${1} | awk '{print $1}'))
        else
            out=($(echo "${rkt_output}" | awk '{print $1}'))
        fi
        COMPREPLY=( $( compgen -W "${out[*]}" -- "$cur" ) )
    fi
}

__custom_func() {
    case ${last_command} in
        rkt_image_export | \
        rkt_image_extract | \
        rkt_image_cat-manifest | \
        rkt_image_render | \
        rkt_image_rm | \
        rkt_run | \
        rkt_prepare)
            __rkt_parse_image
            return
            ;;
        rkt_run-prepared)
            __rkt_parse_list prepared
            return
            ;;
        rkt_enter)
            __rkt_parse_list running
            return
            ;;
        rkt_rm)
            __rkt_parse_list "prepare\|exited"
            return
            ;;
        rkt_status)
            __rkt_parse_list
            return
            ;;
        *)
            ;;
    esac
}

_rkt_api-service()
{
    last_command="rkt_api-service"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--listen=")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_cat-manifest()
{
    last_command="rkt_cat-manifest"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--pretty-print")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_enter()
{
    last_command="rkt_enter"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--app=")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_fetch()
{
    last_command="rkt_fetch"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--no-store")
    flags+=("--signature=")
    flags+=("--store-only")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_gc()
{
    last_command="rkt_gc"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--expire-prepared=")
    flags+=("--grace-period=")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_image_cat-manifest()
{
    last_command="rkt_image_cat-manifest"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--pretty-print")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_image_export()
{
    last_command="rkt_image_export"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--overwrite")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_image_extract()
{
    last_command="rkt_image_extract"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--overwrite")
    flags+=("--rootfs-only")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_image_gc()
{
    last_command="rkt_image_gc"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--grace-period=")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_image_list()
{
    last_command="rkt_image_list"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--fields=")
    flags+=("--full")
    flags+=("--no-legend")
    flags+=("--order=")
    flags+=("--sort=")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_image_render()
{
    last_command="rkt_image_render"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--overwrite")
    flags+=("--rootfs-only")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_image_rm()
{
    last_command="rkt_image_rm"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_image()
{
    last_command="rkt_image"
    commands=()
    commands+=("cat-manifest")
    commands+=("export")
    commands+=("extract")
    commands+=("gc")
    commands+=("list")
    commands+=("render")
    commands+=("rm")

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_list()
{
    last_command="rkt_list"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--full")
    flags+=("--no-legend")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_metadata-service()
{
    last_command="rkt_metadata-service"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--listen-port=")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_prepare()
{
    last_command="rkt_prepare"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--exec=")
    flags+=("--inherit-env")
    flags+=("--mount=")
    flags+=("--no-overlay")
    flags+=("--no-store")
    flags+=("--pod-manifest=")
    flags+=("--port=")
    flags+=("--private-users")
    flags+=("--quiet")
    flags+=("--set-env=")
    flags+=("--stage1-path=")
    flags+=("--stage1-url=")
    flags+=("--stage1-name=")
    flags+=("--stage1-hash=")
    flags+=("--stage1-from-dir=")
    flags+=("--store-only")
    flags+=("--volume=")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_rm()
{
    last_command="rkt_rm"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--uuid-file=")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_run()
{
    last_command="rkt_run"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cpu=")
    flags+=("--exec=")
    flags+=("--inherit-env")
    flags+=("--interactive")
    flags+=("--mds-register")
    flags+=("--memory=")
    flags+=("--mount=")
    flags+=("--net=")
    flags+=("--no-overlay")
    flags+=("--no-store")
    flags+=("--pod-manifest=")
    flags+=("--port=")
    flags+=("--private-users")
    flags+=("--set-env=")
    flags+=("--signature=")
    flags+=("--stage1-path=")
    flags+=("--stage1-url=")
    flags+=("--stage1-name=")
    flags+=("--stage1-hash=")
    flags+=("--stage1-from-dir=")
    flags+=("--store-only")
    flags+=("--uuid-file-save=")
    flags+=("--volume=")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_run-prepared()
{
    last_command="rkt_run-prepared"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--interactive")
    flags+=("--mds-register")
    flags+=("--net=")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_status()
{
    last_command="rkt_status"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--wait")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_trust()
{
    last_command="rkt_trust"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--insecure-allow-http")
    flags+=("--prefix=")
    flags+=("--root")
    flags+=("--skip-fingerprint-review")
    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt_version()
{
    last_command="rkt_version"
    commands=()

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

_rkt()
{
    last_command="rkt"
    commands=()
    commands+=("api-service")
    commands+=("cat-manifest")
    commands+=("enter")
    commands+=("fetch")
    commands+=("gc")
    commands+=("image")
    commands+=("list")
    commands+=("metadata-service")
    commands+=("prepare")
    commands+=("rm")
    commands+=("run")
    commands+=("run-prepared")
    commands+=("status")
    commands+=("trust")
    commands+=("version")

    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--debug")
    flags+=("--dir=")
    flags+=("--insecure-options=")
    flags+=("--user-config=")
    flags+=("--local-config=")
    flags+=("--system-config=")
    flags+=("--trust-keys-from-https")

    must_have_one_flag=()
    must_have_one_noun=()
}

__start_rkt()
{
    local cur prev words cword
    if declare -F _init_completion >/dev/null 2>&1; then
        _init_completion -s || return
    else
        __my_init_completion || return
    fi

    local c=0
    local flags=()
    local two_word_flags=()
    local flags_with_completion=()
    local flags_completion=()
    local commands=("rkt")
    local must_have_one_flag=()
    local must_have_one_noun=()
    local last_command
    local nouns=()

    __handle_word
}

if [[ $(type -t compopt) = "builtin" ]]; then
    complete -F __start_rkt rkt
else
    complete -o nospace -F __start_rkt rkt
fi

# ex: ts=4 sw=4 et filetype=sh
