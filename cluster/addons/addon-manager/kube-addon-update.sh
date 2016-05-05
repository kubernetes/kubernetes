#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# The business logic for whether a given object should be created
# was already enforced by salt, and /etc/kubernetes/addons is the
# managed result is of that. Start everything below that directory.

# Parameters
# $1 path to add-ons


# LIMITATIONS
# 1. controllers are not updated unless their name is changed
# 3. Services will not be updated unless their name is changed,
#    but for services we actually want updates without name change.
# 4. Json files are not handled at all. Currently addons must be
#    in yaml files
# 5. exit code is probably not always correct (I haven't checked
#    carefully if it works in 100% cases)
# 6. There are no unittests
# 8. Will not work if the total length of paths to addons is greater than
#    bash can handle. Probably it is not a problem: ARG_MAX=2097152 on GCE.
# 9. Performance issue: yaml files are read many times in a single execution.

# cosmetic improvements to be done
# 1. improve the log function; add timestamp, file name, etc.
# 2. logging doesn't work from files that print things out.
# 3. kubectl prints the output to stderr (the output should be captured and then
#    logged)



# global config
KUBECTL=${TEST_KUBECTL:-}   # substitute for tests
KUBECTL=${KUBECTL:-${KUBECTL_BIN:-}}
KUBECTL=${KUBECTL:-/usr/local/bin/kubectl}
if [[ ! -x ${KUBECTL} ]]; then
    echo "ERROR: kubectl command (${KUBECTL}) not found or is not executable" 1>&2
    exit 1
fi

# If an add-on definition is incorrect, or a definition has just disappeared
# from the local directory, the script will still keep on retrying.
# The script does not end until all retries are done, so
# one invalid manifest may block updates of other add-ons.
# Be careful how you set these parameters
NUM_TRIES=1    # will be updated based on input parameters
DELAY_AFTER_ERROR_SEC=${TEST_DELAY_AFTER_ERROR_SEC:=10}


# remember that you can't log from functions that print some output (because
# logs are also printed on stdout)
# $1 level
# $2 message
function log() {
  # manage log levels manually here

  # add the timestamp if you find it useful
  case $1 in
    DB3 )
#        echo "$1: $2"
        ;;
    DB2 )
#        echo "$1: $2"
        ;;
    DBG )
#        echo "$1: $2"
        ;;
    INFO )
        echo "$1: $2"
        ;;
    WRN )
        echo "$1: $2"
        ;;
    ERR )
        echo "$1: $2"
        ;;
    * )
        echo "INVALID_LOG_LEVEL $1: $2"
        ;;
  esac
}

#$1 yaml file path
function get-object-kind-from-file() {
    # prints to stdout, so log cannot be used
    #WARNING: only yaml is supported
    cat $1 | python -c '''
try:
        import pipes,sys,yaml
        y = yaml.load(sys.stdin)
        labels = y["metadata"]["labels"]
        if ("kubernetes.io/cluster-service", "true") not in labels.iteritems():
            # all add-ons must have the label "kubernetes.io/cluster-service".
            # Otherwise we are ignoring them (the update will not work anyway)
            print "ERROR"
        else:
            print y["kind"]
except Exception, ex:
        print "ERROR"
    '''
}

# $1 yaml file path
# returns a string of the form <namespace>/<name> (we call it nsnames)
function get-object-nsname-from-file() {
    # prints to stdout, so log cannot be used
    #WARNING: only yaml is supported
    #addons that do not specify a namespace are assumed to be in "default".
    cat $1 | python -c '''
try:
        import pipes,sys,yaml
        y = yaml.load(sys.stdin)
        labels = y["metadata"]["labels"]
        if ("kubernetes.io/cluster-service", "true") not in labels.iteritems():
            # all add-ons must have the label "kubernetes.io/cluster-service".
            # Otherwise we are ignoring them (the update will not work anyway)
            print "ERROR"
        else:
            try:
                print "%s/%s" % (y["metadata"]["namespace"], y["metadata"]["name"])
            except Exception, ex:
                print "default/%s" % y["metadata"]["name"]
except Exception, ex:
        print "ERROR"
    '''
}

# $1 addon directory path
# $2 addon type (e.g. ReplicationController)
# echoes the string with paths to files containing addon for the given type
# works only for yaml files (!) (ignores json files)
function get-addon-paths-from-disk() {
    # prints to stdout, so log cannot be used
    local -r addon_dir=$1
    local -r obj_type=$2
    local kind
    local file_path
    for file_path in $(find ${addon_dir} -name \*.yaml); do
        kind=$(get-object-kind-from-file ${file_path})
        # WARNING: assumption that the topmost indentation is zero (I'm not sure yaml allows for topmost indentation)
        if [[ "${kind}" == "${obj_type}" ]]; then
            echo ${file_path}
        fi
    done
}

# waits for all subprocesses
# returns 0 if all of them were successful and 1 otherwise
function wait-for-jobs() {
    local rv=0
    local pid
    for pid in $(jobs -p); do
        wait ${pid}
        if [[ $? -ne 0 ]]; then
            rv=1;
            log ERR "error in pid ${pid}"
        fi
        log DB2 "pid ${pid} completed, current error code: ${rv}"
    done
    return ${rv}
}


function run-until-success() {
    local -r command=$1
    local tries=$2
    local -r delay=$3
    local -r command_name=$1
    while [ ${tries} -gt 0 ]; do
        log DBG "executing: '$command'"
        # let's give the command as an argument to bash -c, so that we can use
        # && and || inside the command itself
        /bin/bash -c "${command}" && \
            log DB3 "== Successfully executed ${command_name} at $(date -Is) ==" && \
            return 0
        let tries=tries-1
        log INFO "== Failed to execute ${command_name} at $(date -Is). ${tries} tries remaining. =="
        sleep ${delay}
    done
    return 1
}

# $1 object type
# returns a list of <namespace>/<name> pairs (nsnames)
function get-addon-nsnames-from-server() {
    local -r obj_type=$1
    "${KUBECTL}" get "${obj_type}" --all-namespaces -o go-template="{{range.items}}{{.metadata.namespace}}/{{.metadata.name}} {{end}}" --api-version=v1 -l kubernetes.io/cluster-service=true
}

# returns the characters after the last separator (including)
# If the separator is empty or if it doesn't appear in the string,
# an empty string is printed
# $1 input string
# $2 separator (must be single character, or empty)
function get-suffix() {
    # prints to stdout, so log cannot be used
    local -r input_string=$1
    local -r separator=$2
    local suffix

    if [[ "${separator}" == "" ]]; then
        echo ""
        return
    fi

    if  [[ "${input_string}" == *"${separator}"* ]]; then
        suffix=$(echo "${input_string}" | rev | cut -d "${separator}" -f1 | rev)
        echo "${separator}${suffix}"
    else
        echo ""
    fi
}

# returns the characters up to the last '-' (without it)
# $1 input string
# $2 separator
function get-basename() {
    # prints to stdout, so log cannot be used
    local -r input_string=$1
    local -r separator=$2
    local suffix
    suffix="$(get-suffix ${input_string} ${separator})"
    # this will strip the suffix (if matches)
    echo ${input_string%$suffix}
}

function delete-object() {
    local -r obj_type=$1
    local -r namespace=$2
    local -r obj_name=$3
    log INFO "Deleting ${obj_type} ${namespace}/${obj_name}"

    run-until-success "${KUBECTL} delete --namespace=${namespace} ${obj_type} ${obj_name}" ${NUM_TRIES} ${DELAY_AFTER_ERROR_SEC}
}

function create-object() {
    local -r obj_type=$1
    local -r file_path=$2

    local nsname_from_file
    nsname_from_file=$(get-object-nsname-from-file ${file_path})
    if [[ "${nsname_from_file}" == "ERROR" ]]; then
       log INFO "Cannot read object name from ${file_path}. Ignoring"
       return 1
    fi
    IFS='/' read namespace obj_name <<< "${nsname_from_file}"

    log INFO "Creating new ${obj_type} from file ${file_path} in namespace ${namespace}, name: ${obj_name}"
    # this will keep on failing if the ${file_path} disappeared in the meantime.
    # Do not use too many retries.
    run-until-success "${KUBECTL} create --namespace=${namespace} -f ${file_path}" ${NUM_TRIES} ${DELAY_AFTER_ERROR_SEC}
}

function update-object() {
    local -r obj_type=$1
    local -r namespace=$2
    local -r obj_name=$3
    local -r file_path=$4
    log INFO "updating the ${obj_type} ${namespace}/${obj_name} with the new definition ${file_path}"
    delete-object ${obj_type} ${namespace} ${obj_name}
    create-object ${obj_type} ${file_path}
}

# deletes the objects from the server
# $1 object type
# $2 a list of object nsnames
function delete-objects() {
    local -r obj_type=$1
    local -r obj_nsnames=$2
    local namespace
    local obj_name
    for nsname in ${obj_nsnames}; do
        IFS='/' read namespace obj_name <<< "${nsname}"
        delete-object ${obj_type} ${namespace} ${obj_name} &
    done
}

# creates objects from the given files
# $1 object type
# $2 a list of paths to definition files
function create-objects() {
    local -r obj_type=$1
    local -r file_paths=$2
    local file_path
    for file_path in ${file_paths}; do
        # Remember that the file may have disappear by now
        # But we don't want to check it here because
        # such race condition may always happen after
        # we check it. Let's have the race
        # condition happen a bit more often so that
        # we see that our tests pass anyway.
        create-object ${obj_type} ${file_path} &
    done
}

# updates objects
# $1 object type
# $2 a list of update specifications
# each update specification is a ';' separated pair: <nsname>;<file path>
function update-objects() {
    local -r obj_type=$1      # ignored
    local -r update_spec=$2
    local objdesc
    local nsname
    local obj_name
    local namespace

    for objdesc in ${update_spec}; do
        IFS=';' read nsname file_path <<< "${objdesc}"
        IFS='/' read namespace obj_name <<< "${nsname}"

        update-object ${obj_type} ${namespace} ${obj_name} ${file_path} &
    done
}

# Global variables set by function match-objects.
nsnames_for_delete=""   # a list of object nsnames to be deleted
for_update=""           # a list of pairs <nsname>;<filePath> for objects that should be updated
nsnames_for_ignore=""   # a list of object nsnames that will be ignored
new_files=""            # a list of file paths that weren't matched by any existing objects (these objects must be created now)


# $1 path to files with objects
# $2 object type in the API (ReplicationController or Service)
# $3 name separator (single character or empty)
function match-objects() {
    local -r addon_dir=$1
    local -r obj_type=$2
    local -r separator=$3

    # output variables (globals)
    nsnames_for_delete=""
    for_update=""
    nsnames_for_ignore=""
    new_files=""

    addon_nsnames_on_server=$(get-addon-nsnames-from-server "${obj_type}")
    # if the api server is unavailable then abandon the update for this cycle 
    if [[ $? -ne 0 ]]; then
        log ERR "unable to query ${obj_type} - exiting"
        exit 1
    fi

    addon_paths_in_files=$(get-addon-paths-from-disk "${addon_dir}" "${obj_type}")

    log DB2 "addon_nsnames_on_server=${addon_nsnames_on_server}"
    log DB2 "addon_paths_in_files=${addon_paths_in_files}"

    local matched_files=""

    local basensname_on_server=""
    local nsname_on_server=""
    local suffix_on_server=""
    local nsname_from_file=""
    local suffix_from_file=""
    local found=0
    local addon_path=""

    # objects that were moved between namespaces will have different nsname
    # because the namespace is included. So they will be treated
    # like different objects and not updated but deleted and created again
    # (in the current version update is also delete+create, so it does not matter)
    for nsname_on_server in ${addon_nsnames_on_server}; do
        basensname_on_server=$(get-basename ${nsname_on_server} ${separator})
        suffix_on_server="$(get-suffix ${nsname_on_server} ${separator})"

        log DB3 "Found existing addon ${nsname_on_server}, basename=${basensname_on_server}"

        # check if the addon is present in the directory and decide
        # what to do with it
        # this is not optimal because we're reading the files over and over
        # again. But for small number of addons it doesn't matter so much.
        found=0
        for addon_path in ${addon_paths_in_files}; do
            nsname_from_file=$(get-object-nsname-from-file ${addon_path})
            if [[ "${nsname_from_file}" == "ERROR" ]]; then
                log INFO "Cannot read object name from ${addon_path}. Ignoring"
                continue
            else
                log DB2 "Found object name '${nsname_from_file}' in file ${addon_path}"
            fi
            suffix_from_file="$(get-suffix ${nsname_from_file} ${separator})"

            log DB3 "matching: ${basensname_on_server}${suffix_from_file} == ${nsname_from_file}"
            if [[ "${basensname_on_server}${suffix_from_file}" == "${nsname_from_file}" ]]; then
                log DB3 "matched existing ${obj_type} ${nsname_on_server} to file ${addon_path}; suffix_on_server=${suffix_on_server}, suffix_from_file=${suffix_from_file}"
                found=1
                matched_files="${matched_files} ${addon_path}"
                if [[ "${suffix_on_server}" == "${suffix_from_file}" ]]; then
                    nsnames_for_ignore="${nsnames_for_ignore} ${nsname_from_file}"
                else
                    for_update="${for_update} ${nsname_on_server};${addon_path}"
                fi
                break
            fi
        done
        if [[ ${found} -eq 0 ]]; then
            log DB2 "No definition file found for replication controller ${nsname_on_server}. Scheduling for deletion"
            nsnames_for_delete="${nsnames_for_delete} ${nsname_on_server}"
        fi
    done

    log DB3 "matched_files=${matched_files}"


    # note that if the addon file is invalid (or got removed after listing files
    # but before we managed to match it) it will not be matched to any
    # of the existing objects. So we will treat it as a new file
    # and try to create its object.
    for addon_path in ${addon_paths_in_files}; do
        echo ${matched_files} | grep "${addon_path}" >/dev/null
        if [[ $? -ne 0 ]]; then
            new_files="${new_files} ${addon_path}"
        fi
    done
}



function reconcile-objects() {
    local -r addon_path=$1
    local -r obj_type=$2
    local -r separator=$3    # name separator
    match-objects ${addon_path} ${obj_type} ${separator}

    log DBG "${obj_type}: nsnames_for_delete=${nsnames_for_delete}"
    log DBG "${obj_type}: for_update=${for_update}"
    log DBG "${obj_type}: nsnames_for_ignore=${nsnames_for_ignore}"
    log DBG "${obj_type}: new_files=${new_files}"

    delete-objects "${obj_type}" "${nsnames_for_delete}"
    # wait for jobs below is a protection against changing the basename
    # of a replication controllerm without changing the selector.
    # If we don't wait, the new rc may be created before the old one is deleted
    # In such case the old one will wait for all its pods to be gone, but the pods
    # are created by the new replication controller.
    # passing --cascade=false could solve the problem, but we want
    # all orphan pods to be deleted.
    wait-for-jobs
    deleteResult=$?

    create-objects "${obj_type}" "${new_files}"
    update-objects "${obj_type}" "${for_update}"

    local nsname
    for nsname in ${nsnames_for_ignore}; do
        log DB2 "The ${obj_type} ${nsname} is already up to date"
    done

    wait-for-jobs
    createUpdateResult=$?

    if [[ ${deleteResult} -eq 0 ]] && [[ ${createUpdateResult} -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}

function update-addons() {
    local -r addon_path=$1
    # be careful, reconcile-objects uses global variables
    reconcile-objects ${addon_path} ReplicationController "-" &
    reconcile-objects ${addon_path} Deployment "-" &

    # We don't expect names to be versioned for the following kinds, so
    # we match the entire name, ignoring version suffix.
    # That's why we pass an empty string as the version separator.
    # If the description differs on disk, the object should be recreated.
    # This is not implemented in this version.
    reconcile-objects ${addon_path} Service "" &
    reconcile-objects ${addon_path} PersistentVolume "" &
    reconcile-objects ${addon_path} PersistentVolumeClaim "" &

    wait-for-jobs
    if [[ $? -eq 0 ]]; then
        log INFO "== Kubernetes addon update completed successfully at $(date -Is) =="
    else
        log WRN "== Kubernetes addon update completed with errors at $(date -Is) =="
    fi
}

# input parameters:
# $1 input directory
# $2 retry period in seconds - the script will retry api-server errors for approximately
#     this amound of time (it is not very precise), at interval equal $DELAY_AFTER_ERROR_SEC.
#

if [[ $# -ne 2 ]]; then
    echo "Illegal number of parameters. Usage $0 addon-dir [retry-period]" 1>&2
    exit 1
fi

NUM_TRIES=$(($2 / ${DELAY_AFTER_ERROR_SEC}))
if [[ ${NUM_TRIES} -le 0 ]]; then
    NUM_TRIES=1
fi

addon_path=$1
update-addons ${addon_path}
