# bash completion for heketi-cli                           -*- shell-script -*-

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
    _get_comp_words_by_ref "$@" cur prev words cword
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
    __debug "${FUNCNAME[0]}"
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
                [[ "${COMPREPLY[0]}" == *= ]] || compopt +o nospace
            fi

            # complete after --flag=abc
            if [[ $cur == *=* ]]; then
                if [[ $(type -t compopt) = "builtin" ]]; then
                    compopt +o nospace
                fi

                local index flag
                flag="${cur%%=*}"
                __index_of_word "${flag}" "${flags_with_completion[@]}"
                if [[ ${index} -ge 0 ]]; then
                    COMPREPLY=()
                    PREFIX=""
                    cur="${cur#*=}"
                    ${flags_completion[${index}]}
                    if [ -n "${ZSH_VERSION}" ]; then
                        # zfs completion needs --flag= prefix
                        eval "COMPREPLY=( \"\${COMPREPLY[@]/#/${flag}=}\" )"
                    fi
                fi
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
    completions=("${commands[@]}")
    if [[ ${#must_have_one_noun[@]} -ne 0 ]]; then
        completions=("${must_have_one_noun[@]}")
    fi
    if [[ ${#must_have_one_flag[@]} -ne 0 ]]; then
        completions+=("${must_have_one_flag[@]}")
    fi
    COMPREPLY=( $(compgen -W "${completions[*]}" -- "$cur") )

    if [[ ${#COMPREPLY[@]} -eq 0 && ${#noun_aliases[@]} -gt 0 && ${#must_have_one_noun[@]} -ne 0 ]]; then
        COMPREPLY=( $(compgen -W "${noun_aliases[*]}" -- "$cur") )
    fi

    if [[ ${#COMPREPLY[@]} -eq 0 ]]; then
        declare -F __custom_func >/dev/null && __custom_func
    fi

    __ltrim_colon_completions "$cur"
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
    __debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"

    # if a command required a flag, and we found it, unset must_have_one_flag()
    local flagname=${words[c]}
    local flagvalue
    # if the word contained an =
    if [[ ${words[c]} == *"="* ]]; then
        flagvalue=${flagname#*=} # take in as flagvalue after the =
        flagname=${flagname%%=*} # strip everything after the =
        flagname="${flagname}=" # but put the = back
    fi
    __debug "${FUNCNAME[0]}: looking for ${flagname}"
    if __contains_word "${flagname}" "${must_have_one_flag[@]}"; then
        must_have_one_flag=()
    fi

    # if you set a flag which only applies to this command, don't show subcommands
    if __contains_word "${flagname}" "${local_nonpersistent_flags[@]}"; then
      commands=()
    fi

    # keep flag value with flagname as flaghash
    if [ -n "${flagvalue}" ] ; then
        flaghash[${flagname}]=${flagvalue}
    elif [ -n "${words[ $((c+1)) ]}" ] ; then
        flaghash[${flagname}]=${words[ $((c+1)) ]}
    else
        flaghash[${flagname}]="true" # pad "true" for bool flag
    fi

    # skip the argument to a two word flag
    if __contains_word "${words[c]}" "${two_word_flags[@]}"; then
        c=$((c+1))
        # if we are looking for a flags value, don't show commands
        if [[ $c -eq $cword ]]; then
            commands=()
        fi
    fi

    c=$((c+1))

}

__handle_noun()
{
    __debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"

    if __contains_word "${words[c]}" "${must_have_one_noun[@]}"; then
        must_have_one_noun=()
    elif __contains_word "${words[c]}" "${noun_aliases[@]}"; then
        must_have_one_noun=()
    fi

    nouns+=("${words[c]}")
    c=$((c+1))
}

__handle_command()
{
    __debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"

    local next_command
    if [[ -n ${last_command} ]]; then
        next_command="_${last_command}_${words[c]//:/__}"
    else
        if [[ $c -eq 0 ]]; then
            next_command="_$(basename "${words[c]//:/__}")"
        else
            next_command="_${words[c]//:/__}"
        fi
    fi
    c=$((c+1))
    __debug "${FUNCNAME[0]}: looking for ${next_command}"
    declare -F $next_command >/dev/null && $next_command
}

__handle_word()
{
    if [[ $c -ge $cword ]]; then
        __handle_reply
        return
    fi
    __debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"
    if [[ "${words[c]}" == -* ]]; then
        __handle_flag
    elif __contains_word "${words[c]}" "${commands[@]}"; then
        __handle_command
    elif [[ $c -eq 0 ]] && __contains_word "$(basename "${words[c]}")" "${commands[@]}"; then
        __handle_command
    else
        __handle_noun
    fi
    __handle_word
}

_heketi-cli_cluster_create()
{
    last_command="heketi-cli_cluster_create"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_cluster_delete()
{
    last_command="heketi-cli_cluster_delete"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_cluster_info()
{
    last_command="heketi-cli_cluster_info"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_cluster_list()
{
    last_command="heketi-cli_cluster_list"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_cluster()
{
    last_command="heketi-cli_cluster"
    commands=()
    commands+=("create")
    commands+=("delete")
    commands+=("info")
    commands+=("list")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_device_add()
{
    last_command="heketi-cli_device_add"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--name=")
    local_nonpersistent_flags+=("--name=")
    flags+=("--node=")
    local_nonpersistent_flags+=("--node=")
    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_device_delete()
{
    last_command="heketi-cli_device_delete"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_device_disable()
{
    last_command="heketi-cli_device_disable"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_device_enable()
{
    last_command="heketi-cli_device_enable"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_device_info()
{
    last_command="heketi-cli_device_info"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_device()
{
    last_command="heketi-cli_device"
    commands=()
    commands+=("add")
    commands+=("delete")
    commands+=("disable")
    commands+=("enable")
    commands+=("info")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_node_add()
{
    last_command="heketi-cli_node_add"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--cluster=")
    local_nonpersistent_flags+=("--cluster=")
    flags+=("--management-host-name=")
    local_nonpersistent_flags+=("--management-host-name=")
    flags+=("--storage-host-name=")
    local_nonpersistent_flags+=("--storage-host-name=")
    flags+=("--zone=")
    local_nonpersistent_flags+=("--zone=")
    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_node_delete()
{
    last_command="heketi-cli_node_delete"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_node_disable()
{
    last_command="heketi-cli_node_disable"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_node_enable()
{
    last_command="heketi-cli_node_enable"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_node_info()
{
    last_command="heketi-cli_node_info"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_node()
{
    last_command="heketi-cli_node"
    commands=()
    commands+=("add")
    commands+=("delete")
    commands+=("disable")
    commands+=("enable")
    commands+=("info")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_setup-openshift-heketi-storage()
{
    last_command="heketi-cli_setup-openshift-heketi-storage"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--listfile=")
    local_nonpersistent_flags+=("--listfile=")
    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_topology_info()
{
    last_command="heketi-cli_topology_info"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_topology_load()
{
    last_command="heketi-cli_topology_load"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json=")
    two_word_flags+=("-j")
    local_nonpersistent_flags+=("--json=")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_topology()
{
    last_command="heketi-cli_topology"
    commands=()
    commands+=("info")
    commands+=("load")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_volume_create()
{
    last_command="heketi-cli_volume_create"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--clusters=")
    local_nonpersistent_flags+=("--clusters=")
    flags+=("--disperse-data=")
    local_nonpersistent_flags+=("--disperse-data=")
    flags+=("--durability=")
    local_nonpersistent_flags+=("--durability=")
    flags+=("--name=")
    local_nonpersistent_flags+=("--name=")
    flags+=("--persistent-volume")
    local_nonpersistent_flags+=("--persistent-volume")
    flags+=("--persistent-volume-endpoint=")
    local_nonpersistent_flags+=("--persistent-volume-endpoint=")
    flags+=("--persistent-volume-file=")
    local_nonpersistent_flags+=("--persistent-volume-file=")
    flags+=("--redundancy=")
    local_nonpersistent_flags+=("--redundancy=")
    flags+=("--replica=")
    local_nonpersistent_flags+=("--replica=")
    flags+=("--size=")
    local_nonpersistent_flags+=("--size=")
    flags+=("--snapshot-factor=")
    local_nonpersistent_flags+=("--snapshot-factor=")
    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_volume_delete()
{
    last_command="heketi-cli_volume_delete"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_volume_expand()
{
    last_command="heketi-cli_volume_expand"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--expand-size=")
    local_nonpersistent_flags+=("--expand-size=")
    flags+=("--volume=")
    local_nonpersistent_flags+=("--volume=")
    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_volume_info()
{
    last_command="heketi-cli_volume_info"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_volume_list()
{
    last_command="heketi-cli_volume_list"
    commands=()

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli_volume()
{
    last_command="heketi-cli_volume"
    commands=()
    commands+=("create")
    commands+=("delete")
    commands+=("expand")
    commands+=("info")
    commands+=("list")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

_heketi-cli()
{
    last_command="heketi-cli"
    commands=()
    commands+=("cluster")
    commands+=("device")
    commands+=("node")
    commands+=("setup-openshift-heketi-storage")
    commands+=("topology")
    commands+=("volume")

    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

    flags+=("--json")
    flags+=("--log-flush-frequency=")
    flags+=("--secret=")
    flags+=("--server=")
    two_word_flags+=("-s")
    flags+=("--user=")
    flags+=("--version")
    flags+=("-v")
    local_nonpersistent_flags+=("--version")

    must_have_one_flag=()
    must_have_one_noun=()
    noun_aliases=()
}

__start_heketi-cli()
{
    local cur prev words cword
    declare -A flaghash 2>/dev/null || :
    if declare -F _init_completion >/dev/null 2>&1; then
        _init_completion -s || return
    else
        __my_init_completion -n "=" || return
    fi

    local c=0
    local flags=()
    local two_word_flags=()
    local local_nonpersistent_flags=()
    local flags_with_completion=()
    local flags_completion=()
    local commands=("heketi-cli")
    local must_have_one_flag=()
    local must_have_one_noun=()
    local last_command
    local nouns=()

    __handle_word
}

if [[ $(type -t compopt) = "builtin" ]]; then
    complete -o default -F __start_heketi-cli heketi-cli
else
    complete -o default -o nospace -F __start_heketi-cli heketi-cli
fi

# ex: ts=4 sw=4 et filetype=sh
