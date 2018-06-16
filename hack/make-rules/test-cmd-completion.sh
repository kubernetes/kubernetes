#!/bin/bash

# Copyright 2018 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This command checks that the built commands can function together for
# simple scenarios.  It does not require Docker.

check_completion_result(){
    local completion COMP_CWORD COMP_LINE COMP_POINT COMP_WORDS COMPREPLY=()

    # load bash-completion if necessary
    declare -F _completion_loader &>/dev/null || {
        source /usr/share/bash-completion/bash_completion
    }

    RESULT="$1"
    shift

    COMP_LINE=$@
    COMP_POINT=${#COMP_LINE}
    eval set -- "${@}"
    COMP_WORDS=("${@}")

    # add '' to COMP_WORDS if the last character of the command line is a space
    [[ ${COMP_LINE[@]: -1} = ' ' ]] && COMP_WORDS+=('')

    # index of the last word
    COMP_CWORD=$(( ${#COMP_WORDS[@]} - 1 ))

    # determine completion function
    completion=$(complete -p "$1" 2>/dev/null | awk '{print $(NF-1)}')

    # run _completion_loader only if necessary
    [[ -n $completion ]] || {
        # load completion
        _completion_loader "$1"
        # detect completion
        completion=$(complete -p "$1" 2>/dev/null | awk '{print $(NF-1)}')
    }

    # ensure completion was detected
    [[ -n $completion ]] || return 1

    # execute completion function
    "$completion"


    # print completions to stdout sort and assign to compare var
    result=$(printf '%s ' "${COMPREPLY[@]}" | sort)
    expected=$(printf '%s ' "$RESULT" | sort)

    if [ "$result" == "$expected" ]
    then
        echo "kubectl completion test success: '$@'"
    else
        echo "kubectl completion test failed: '$@'"
        exit 1
    fi

}

completion_root_command() {
    check_completion_result 'annotate api-versions apply attach auth autoscale' 'kubectl a'
    check_completion_result 'certificate cluster-info completion config convert cordon cp create' 'kubectl c'
    check_completion_result 'delete describe drain' 'kubectl d'
    check_completion_result 'edit exec explain expose' 'kubectl e'
    check_completion_result 'get' 'kubectl g'
    check_completion_result 'label logs' 'kubectl l'
    check_completion_result 'options' 'kubectl o'
    check_completion_result 'patch plugin port-forward proxy' 'kubectl p'
    check_completion_result 'replace rolling-update rollout run' 'kubectl r'
    check_completion_result 'scale set' 'kubectl s'
    check_completion_result 'taint top' 'kubectl t'
    check_completion_result 'uncordon' 'kubectl u'
    check_completion_result 'version' 'kubectl v'
}

# Runs all kubectl completion tests.
runCompletionTests() {
    # setup completion
    source <(kubectl completion bash)

    # check kubectl root options
    completion_root_command
}