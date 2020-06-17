#!/usr/bin/env bash
# This utility file contains functions that wrap commands to be tested. All wrapper functions run commands
# in a sub-shell and redirect all output. Tests in test-cmd *must* use these functions for testing.

# expect_success runs the cmd and expects an exit code of 0
function os::cmd::expect_success() {
	if [[ $# -ne 1 ]]; then echo "os::cmd::expect_success expects only one argument, got $#"; return 1; fi
	local cmd=$1

	os::cmd::internal::expect_exit_code_run_grep "${cmd}"
}
readonly -f os::cmd::expect_success

# expect_failure runs the cmd and expects a non-zero exit code
function os::cmd::expect_failure() {
	if [[ $# -ne 1 ]]; then echo "os::cmd::expect_failure expects only one argument, got $#"; return 1; fi
	local cmd=$1

	os::cmd::internal::expect_exit_code_run_grep "${cmd}" "os::cmd::internal::failure_func"
}
readonly -f os::cmd::expect_failure

# expect_success_and_text runs the cmd and expects an exit code of 0
# as well as running a grep test to find the given string in the output
function os::cmd::expect_success_and_text() {
	if [[ $# -ne 2 ]]; then echo "os::cmd::expect_success_and_text expects two arguments, got $#"; return 1; fi
	local cmd=$1
	local expected_text=$2

	os::cmd::internal::expect_exit_code_run_grep "${cmd}" "os::cmd::internal::success_func" "${expected_text}"
}
readonly -f os::cmd::expect_success_and_text

# expect_failure_and_text runs the cmd and expects a non-zero exit code
# as well as running a grep test to find the given string in the output
function os::cmd::expect_failure_and_text() {
	if [[ $# -ne 2 ]]; then echo "os::cmd::expect_failure_and_text expects two arguments, got $#"; return 1; fi
	local cmd=$1
	local expected_text=$2

	os::cmd::internal::expect_exit_code_run_grep "${cmd}" "os::cmd::internal::failure_func" "${expected_text}"
}
readonly -f os::cmd::expect_failure_and_text

# expect_success_and_not_text runs the cmd and expects an exit code of 0
# as well as running a grep test to ensure the given string is not in the output
function os::cmd::expect_success_and_not_text() {
	if [[ $# -ne 2 ]]; then echo "os::cmd::expect_success_and_not_text expects two arguments, got $#"; return 1; fi
	local cmd=$1
	local expected_text=$2

	os::cmd::internal::expect_exit_code_run_grep "${cmd}" "os::cmd::internal::success_func" "${expected_text}" "os::cmd::internal::failure_func"
}
readonly -f os::cmd::expect_success_and_not_text

# expect_failure_and_not_text runs the cmd and expects a non-zero exit code
# as well as running a grep test to ensure the given string is not in the output
function os::cmd::expect_failure_and_not_text() {
	if [[ $# -ne 2 ]]; then echo "os::cmd::expect_failure_and_not_text expects two arguments, got $#"; return 1; fi
	local cmd=$1
	local expected_text=$2

	os::cmd::internal::expect_exit_code_run_grep "${cmd}" "os::cmd::internal::failure_func" "${expected_text}" "os::cmd::internal::failure_func"
}
readonly -f os::cmd::expect_failure_and_not_text

# expect_code runs the cmd and expects a given exit code
function os::cmd::expect_code() {
	if [[ $# -ne 2 ]]; then echo "os::cmd::expect_code expects two arguments, got $#"; return 1; fi
	local cmd=$1
	local expected_cmd_code=$2

	os::cmd::internal::expect_exit_code_run_grep "${cmd}" "os::cmd::internal::specific_code_func ${expected_cmd_code}"
}
readonly -f os::cmd::expect_code

# expect_code_and_text runs the cmd and expects the given exit code
# as well as running a grep test to find the given string in the output
function os::cmd::expect_code_and_text() {
	if [[ $# -ne 3 ]]; then echo "os::cmd::expect_code_and_text expects three arguments, got $#"; return 1; fi
	local cmd=$1
	local expected_cmd_code=$2
	local expected_text=$3

	os::cmd::internal::expect_exit_code_run_grep "${cmd}" "os::cmd::internal::specific_code_func ${expected_cmd_code}" "${expected_text}"
}
readonly -f os::cmd::expect_code_and_text

# expect_code_and_not_text runs the cmd and expects the given exit code
# as well as running a grep test to ensure the given string is not in the output
function os::cmd::expect_code_and_not_text() {
	if [[ $# -ne 3 ]]; then echo "os::cmd::expect_code_and_not_text expects three arguments, got $#"; return 1; fi
	local cmd=$1
	local expected_cmd_code=$2
	local expected_text=$3

	os::cmd::internal::expect_exit_code_run_grep "${cmd}" "os::cmd::internal::specific_code_func ${expected_cmd_code}" "${expected_text}" "os::cmd::internal::failure_func"
}
readonly -f os::cmd::expect_code_and_not_text

millisecond=1
second=$(( 1000 * millisecond ))
minute=$(( 60 * second ))

# os::cmd::try_until_success runs the cmd in a small interval until either the command succeeds or times out
# the default time-out for os::cmd::try_until_success is 60 seconds.
# the default interval for os::cmd::try_until_success is 200ms
function os::cmd::try_until_success() {
	if [[ $# -lt 1 ]]; then echo "os::cmd::try_until_success expects at least one arguments, got $#"; return 1; fi
	local cmd=$1
	local duration=${2:-$minute}
	local interval=${3:-0.2}

	os::cmd::internal::run_until_exit_code "${cmd}" "os::cmd::internal::success_func" "${duration}" "${interval}"
}
readonly -f os::cmd::try_until_success

# os::cmd::try_until_failure runs the cmd until either the command fails or times out
# the default time-out for os::cmd::try_until_failure is 60 seconds.
function os::cmd::try_until_failure() {
	if [[ $# -lt 1 ]]; then echo "os::cmd::try_until_failure expects at least one argument, got $#"; return 1; fi
	local cmd=$1
	local duration=${2:-$minute}
	local interval=${3:-0.2}

	os::cmd::internal::run_until_exit_code "${cmd}" "os::cmd::internal::failure_func" "${duration}" "${interval}"
}
readonly -f os::cmd::try_until_failure

# os::cmd::try_until_text runs the cmd until either the command outputs the desired text or times out
# the default time-out for os::cmd::try_until_text is 60 seconds.
function os::cmd::try_until_text() {
	if [[ $# -lt 2 ]]; then echo "os::cmd::try_until_text expects at least two arguments, got $#"; return 1; fi
	local cmd=$1
	local text=$2
	local duration=${3:-$minute}
	local interval=${4:-0.2}

	os::cmd::internal::run_until_text "${cmd}" "${text}" "os::cmd::internal::success_func" "${duration}" "${interval}"
}
readonly -f os::cmd::try_until_text

# os::cmd::try_until_not_text runs the cmd until either the command doesnot output the text or times out
# the default time-out for os::cmd::try_until_not_text is 60 seconds.
function os::cmd::try_until_not_text() {
	if [[ $# -lt 2 ]]; then echo "os::cmd::try_until_not_text expects at least two arguments, got $#"; return 1; fi
	local cmd=$1
	local text=$2
	local duration=${3:-$minute}
	local interval=${4:-0.2}

	os::cmd::internal::run_until_text "${cmd}" "${text}" "os::cmd::internal::failure_func" "${duration}" "${interval}"
}
readonly -f os::cmd::try_until_text

# Functions in the os::cmd::internal namespace are discouraged from being used outside of os::cmd

# In order to harvest stderr and stdout at the same time into different buckets, we need to stick them into files
# in an intermediate step
os_cmd_internal_tmpdir="${TMPDIR:-"/tmp"}/cmd"
os_cmd_internal_tmpout="${os_cmd_internal_tmpdir}/tmp_stdout.log"
os_cmd_internal_tmperr="${os_cmd_internal_tmpdir}/tmp_stderr.log"

# os::cmd::internal::expect_exit_code_run_grep runs the provided test command and expects a specific
# exit code from that command as well as the success of a specified `grep` invocation. Output from the
# command to be tested is suppressed unless either `VERBOSE=1` or the test fails. This function bypasses
# any error exiting settings or traps set by upstream callers by masking the return code of the command
# with the return code of setting the result variable on failure.
#
# Globals:
#  - JUNIT_REPORT_OUTPUT
#  - VERBOSE
# Arguments:
#  - 1: the command to run
#  - 2: command evaluation assertion to use
#  - 3: text to test for
#  - 4: text assertion to use
# Returns:
#  - 0: if all assertions met
#  - 1: if any assertions fail
function os::cmd::internal::expect_exit_code_run_grep() {
	local cmd=$1
	# default expected cmd code to 0 for success
	local cmd_eval_func=${2:-os::cmd::internal::success_func}
	# default to nothing
	local grep_args=${3:-}
	# default expected test code to 0 for success
	local test_eval_func=${4:-os::cmd::internal::success_func}

	local -a junit_log

	os::cmd::internal::init_tempdir
	os::test::junit::declare_test_start

	local name
    name=$(os::cmd::internal::describe_call "${cmd}" "${cmd_eval_func}" "${grep_args}" "${test_eval_func}")
	local preamble="Running ${name}..."
	echo "${preamble}"
	# for ease of parsing, we want the entire declaration on one line, so we replace '\n' with ';'
	junit_log+=( "${name//$'\n'/;}" )

	local start_time
    start_time=$(os::cmd::internal::seconds_since_epoch)

	local cmd_result
    cmd_result=$( os::cmd::internal::run_collecting_output "${cmd}"; echo $? )
	local cmd_succeeded
    cmd_succeeded=$( ${cmd_eval_func} "${cmd_result}"; echo $? )

	local test_result=0
	if [[ -n "${grep_args}" ]]; then
		test_result=$( os::cmd::internal::run_collecting_output 'grep -Eq "'"${grep_args}"'" <(os::cmd::internal::get_results)'; echo $? )
	fi
	local test_succeeded
    test_succeeded=$( ${test_eval_func} "${test_result}"; echo $? )

	local end_time
    end_time=$(os::cmd::internal::seconds_since_epoch)
	local time_elapsed
    time_elapsed=$(echo "scale=3; ${end_time} - ${start_time}" | bc | xargs printf '%5.3f') # in decimal seconds, we need leading zeroes for parsing later

	# clear the preamble so we can print out the success or error message
	os::text::clear_string "${preamble}"

	local return_code
	if (( cmd_succeeded && test_succeeded )); then
		os::text::print_green "SUCCESS after ${time_elapsed}s: ${name}"
		junit_log+=( "SUCCESS after ${time_elapsed}s: ${name//$'\n'/;}" )

		if [[ -n ${VERBOSE-} ]]; then
			os::cmd::internal::print_results
		fi
		return_code=0
	else
	    local cause
        cause=$(os::cmd::internal::assemble_causes "${cmd_succeeded}" "${test_succeeded}")

		os::text::print_red_bold "FAILURE after ${time_elapsed}s: ${name}: ${cause}"
		junit_log+=( "FAILURE after ${time_elapsed}s: ${name//$'\n'/;}: ${cause}" )

		os::text::print_red "$(os::cmd::internal::print_results)"
		return_code=1
	fi

	junit_log+=( "$(os::cmd::internal::print_results)" )
	# append inside of a subshell so that IFS doesn't get propagated out
	( IFS=$'\n'; echo "${junit_log[*]}" >> "${JUNIT_REPORT_OUTPUT:-/dev/null}" )
	os::test::junit::declare_test_end
	return "${return_code}"
}
readonly -f os::cmd::internal::expect_exit_code_run_grep

# os::cmd::internal::init_tempdir initializes the temporary directory
function os::cmd::internal::init_tempdir() {
	mkdir -p "${os_cmd_internal_tmpdir}"
	rm -f "${os_cmd_internal_tmpdir}"/tmp_std{out,err}.log
}
readonly -f os::cmd::internal::init_tempdir

# os::cmd::internal::describe_call determines the file:line of the latest function call made
# from outside of this file in the call stack, and the name of the function being called from
# that line, returning a string describing the call
function os::cmd::internal::describe_call() {
	local cmd=$1
	local cmd_eval_func=$2
	local grep_args=${3:-}
	local test_eval_func=${4:-}

	local caller_id
    caller_id=$(os::cmd::internal::determine_caller)
	local full_name="${caller_id}: executing '${cmd}'"

	local cmd_expectation
    cmd_expectation=$(os::cmd::internal::describe_expectation "${cmd_eval_func}")
	local full_name="${full_name} expecting ${cmd_expectation}"

	if [[ -n "${grep_args}" ]]; then
		local text_expecting=
		case "${test_eval_func}" in
		"os::cmd::internal::success_func")
			text_expecting="text" ;;
		"os::cmd::internal::failure_func")
			text_expecting="not text" ;;
		esac
		full_name="${full_name} and ${text_expecting} '${grep_args}'"
	fi

	echo "${full_name}"
}
readonly -f os::cmd::internal::describe_call

# os::cmd::internal::determine_caller determines the file relative to the OpenShift Origin root directory
# and line number of the function call to the outer os::cmd wrapper function
function os::cmd::internal::determine_caller() {
	local call_depth=
	local len_sources="${#BASH_SOURCE[@]}"
	for (( i=0; i<len_sources; i++ )); do
		if grep -q "hack/lib/cmd\.sh$" "${BASH_SOURCE[i]}"; then
			call_depth=i
			break
		fi
	done

	local caller_file="${BASH_SOURCE[${call_depth}]}"
    caller_file="$( os::util::repository_relative_path "${caller_file}" )"
	local caller_line="${BASH_LINENO[${call_depth}-1]}"
	echo "${caller_file}:${caller_line}"
}
readonly -f os::cmd::internal::determine_caller

# os::cmd::internal::describe_expectation describes a command return code evaluation function
function os::cmd::internal::describe_expectation() {
	local func=$1
	case "${func}" in
	"os::cmd::internal::success_func")
		echo "success" ;;
	"os::cmd::internal::failure_func")
		echo "failure" ;;
	"os::cmd::internal::specific_code_func"*[0-9])
	    local code
        code=$(echo "${func}" | grep -Eo "[0-9]+$")
		echo "exit code ${code}" ;;
	"")
		echo "any result"
	esac
}
readonly -f os::cmd::internal::describe_expectation

# os::cmd::internal::seconds_since_epoch returns the number of seconds elapsed since the epoch
# with milli-second precision
function os::cmd::internal::seconds_since_epoch() {
    local ns
    ns=$(date +%s%N)
	# if `date` doesn't support nanoseconds, return second precision
	if [[ "$ns" == *N ]]; then
		date "+%s.000"
		return
	fi
	bc <<< "scale=3; ${ns}/1000000000"
}
readonly -f os::cmd::internal::seconds_since_epoch

# os::cmd::internal::run_collecting_output runs the command given, piping stdout and stderr into
# the given files, and returning the exit code of the command
function os::cmd::internal::run_collecting_output() {
	local cmd=$1

	local result=
	eval "${cmd}" 1>>"${os_cmd_internal_tmpout}" 2>>"${os_cmd_internal_tmperr}" || result=$?
	local result=${result:-0} # if we haven't set result yet, the command succeeded

	return "${result}"
}
readonly -f os::cmd::internal::run_collecting_output

# os::cmd::internal::success_func determines if the input exit code denotes success
# this function returns 0 for false and 1 for true to be compatible with arithmetic tests
function os::cmd::internal::success_func() {
	local exit_code=$1

	# use a negated test to get output correct for (( ))
	[[ "${exit_code}" -ne "0" ]]
	return $?
}
readonly -f os::cmd::internal::success_func

# os::cmd::internal::failure_func determines if the input exit code denotes failure
# this function returns 0 for false and 1 for true to be compatible with arithmetic tests
function os::cmd::internal::failure_func() {
	local exit_code=$1

	# use a negated test to get output correct for (( ))
	[[ "${exit_code}" -eq "0" ]]
	return $?
}
readonly -f os::cmd::internal::failure_func

# os::cmd::internal::specific_code_func determines if the input exit code matches the given code
# this function returns 0 for false and 1 for true to be compatible with arithmetic tests
function os::cmd::internal::specific_code_func() {
	local expected_code=$1
	local exit_code=$2

	# use a negated test to get output correct for (( ))
	[[ "${exit_code}" -ne "${expected_code}" ]]
	return $?
}
readonly -f os::cmd::internal::specific_code_func

# os::cmd::internal::get_results prints the stderr and stdout files
function os::cmd::internal::get_results() {
	cat "${os_cmd_internal_tmpout}" "${os_cmd_internal_tmperr}"
}
readonly -f os::cmd::internal::get_results

# os::cmd::internal::get_last_results prints the stderr and stdout from the last attempt
function os::cmd::internal::get_last_results() {
	awk 'BEGIN { RS = "\x1e" } END { print $0 }' "${os_cmd_internal_tmpout}"
	awk 'BEGIN { RS = "\x1e" } END { print $0 }' "${os_cmd_internal_tmperr}"
}
readonly -f os::cmd::internal::get_last_results

# os::cmd::internal::mark_attempt marks the end of an attempt in the stdout and stderr log files
# this is used to make the try_until_* output more concise
function os::cmd::internal::mark_attempt() {
	echo -e '\x1e' >> "${os_cmd_internal_tmpout}"
	echo -e '\x1e' >> "${os_cmd_internal_tmperr}"
}
readonly -f os::cmd::internal::mark_attempt

# os::cmd::internal::compress_output compresses an output file into timeline representation
function os::cmd::internal::compress_output() {
	local logfile=$1

	awk -f "${OS_ROOT}/hack/lib/compress.awk" "${logfile}"
}
readonly -f os::cmd::internal::compress_output

# os::cmd::internal::print_results pretty-prints the stderr and stdout files. If attempt separators
# are present, this function returns a concise view of the stdout and stderr output files using a
# timeline format, where consecutive output lines that are the same are condensed into one line
# with a counter
function os::cmd::internal::print_results() {
	if [[ -s "${os_cmd_internal_tmpout}" ]]; then
		echo "Standard output from the command:"
		if grep -q $'\x1e' "${os_cmd_internal_tmpout}"; then
			os::cmd::internal::compress_output "${os_cmd_internal_tmpout}"
		else
			cat "${os_cmd_internal_tmpout}"; echo
		fi
	else
		echo "There was no output from the command."
	fi

	if [[ -s "${os_cmd_internal_tmperr}" ]]; then
		echo "Standard error from the command:"
		if grep -q $'\x1e' "${os_cmd_internal_tmperr}"; then
			os::cmd::internal::compress_output "${os_cmd_internal_tmperr}"
		else
			cat "${os_cmd_internal_tmperr}"; echo
		fi
	else
		echo "There was no error output from the command."
	fi
}
readonly -f os::cmd::internal::print_results

# os::cmd::internal::assemble_causes determines from the two input booleans which part of the test
# failed and generates a nice delimited list of failure causes
function os::cmd::internal::assemble_causes() {
	local cmd_succeeded=$1
	local test_succeeded=$2

	local causes=()
	if (( ! cmd_succeeded )); then
		causes+=("the command returned the wrong error code")
	fi
	if (( ! test_succeeded )); then
		causes+=("the output content test failed")
	fi

	local list
    list=$(printf '; %s' "${causes[@]}")
	echo "${list:2}"
}
readonly -f os::cmd::internal::assemble_causes

# os::cmd::internal::run_until_exit_code runs the provided command until the exit code test given
# succeeds or the timeout given runs out. Output from the command to be tested is suppressed unless
# either `VERBOSE=1` or the test fails. This function bypasses any error exiting settings or traps
# set by upstream callers by masking the return code of the command with the return code of setting
# the result variable on failure.
#
# Globals:
#  - JUNIT_REPORT_OUTPUT
#  - VERBOSE
# Arguments:
#  - 1: the command to run
#  - 2: command evaluation assertion to use
#  - 3: timeout duration
#  - 4: interval duration
# Returns:
#  - 0: if all assertions met before timeout
#  - 1: if timeout occurs
function os::cmd::internal::run_until_exit_code() {
	local cmd=$1
	local cmd_eval_func=$2
	local duration=$3
	local interval=$4

	local -a junit_log

	os::cmd::internal::init_tempdir
	os::test::junit::declare_test_start

	local description
    description=$(os::cmd::internal::describe_call "${cmd}" "${cmd_eval_func}")
	local duration_seconds
    duration_seconds=$(echo "scale=3; $(( duration )) / 1000" | bc | xargs printf '%5.3f')
	local description="${description}; re-trying every ${interval}s until completion or ${duration_seconds}s"
	local preamble="Running ${description}..."
	echo "${preamble}"
	# for ease of parsing, we want the entire declaration on one line, so we replace '\n' with ';'
	junit_log+=( "${description//$'\n'/;}" )

	local start_time
    start_time=$(os::cmd::internal::seconds_since_epoch)

	local deadline=$(( $(date +%s000) + duration ))
	local cmd_succeeded=0
	while [ "$(date +%s000)" -lt $deadline ]; do
	    local cmd_result
        cmd_result=$( os::cmd::internal::run_collecting_output "${cmd}"; echo $? )
		cmd_succeeded=$( ${cmd_eval_func} "${cmd_result}"; echo $? )
		if (( cmd_succeeded )); then
			break
		fi
		sleep "${interval}"
		os::cmd::internal::mark_attempt
	done

	local end_time
    end_time=$(os::cmd::internal::seconds_since_epoch)
	local time_elapsed
    time_elapsed=$(echo "scale=9; ${end_time} - ${start_time}" | bc | xargs printf '%5.3f') # in decimal seconds, we need leading zeroes for parsing later

	# clear the preamble so we can print out the success or error message
	os::text::clear_string "${preamble}"

	local return_code
	if (( cmd_succeeded )); then
		os::text::print_green "SUCCESS after ${time_elapsed}s: ${description}"
		junit_log+=( "SUCCESS after ${time_elapsed}s: ${description//$'\n'/;}" )

		if [[ -n ${VERBOSE-} ]]; then
			os::cmd::internal::print_results
		fi
		return_code=0
	else
		os::text::print_red_bold "FAILURE after ${time_elapsed}s: ${description}: the command timed out"
		junit_log+=( "FAILURE after ${time_elapsed}s: ${description//$'\n'/;}: the command timed out" )

		os::text::print_red "$(os::cmd::internal::print_results)"
		return_code=1
	fi

	junit_log+=( "$(os::cmd::internal::print_results)" )
	( IFS=$'\n'; echo "${junit_log[*]}" >> "${JUNIT_REPORT_OUTPUT:-/dev/null}" )
	os::test::junit::declare_test_end
	return "${return_code}"
}
readonly -f os::cmd::internal::run_until_exit_code

# os::cmd::internal::run_until_text runs the provided command until the assertion function succeeds with
# the given text on the command output or the timeout given runs out. This can be used to run until the
# output does or does not contain some text. Output from the command to be tested is suppressed unless
# either `VERBOSE=1` or the test fails. This function bypasses any error exiting settings or traps
# set by upstream callers by masking the return code of the command with the return code of setting
# the result variable on failure.
#
# Globals:
#  - JUNIT_REPORT_OUTPUT
#  - VERBOSE
# Arguments:
#  - 1: the command to run
#  - 2: text to test for
#  - 3: text assertion to use
#  - 4: timeout duration
#  - 5: interval duration
# Returns:
#  - 0: if all assertions met before timeout
#  - 1: if timeout occurs
function os::cmd::internal::run_until_text() {
	local cmd=$1
	local text=$2
	local test_eval_func=${3:-os::cmd::internal::success_func}
	local duration=$4
	local interval=$5

	local -a junit_log

	os::cmd::internal::init_tempdir
	os::test::junit::declare_test_start

	local description
    description=$(os::cmd::internal::describe_call "${cmd}" "" "${text}" "${test_eval_func}")
	local duration_seconds
    duration_seconds=$(echo "scale=3; $(( duration )) / 1000" | bc | xargs printf '%5.3f')
	local description="${description}; re-trying every ${interval}s until completion or ${duration_seconds}s"
	local preamble="Running ${description}..."
	echo "${preamble}"
	# for ease of parsing, we want the entire declaration on one line, so we replace '\n' with ';'
	junit_log+=( "${description//$'\n'/;}" )

	local start_time
    start_time=$(os::cmd::internal::seconds_since_epoch)

	local deadline
    deadline=$(( $(date +%s000) + duration ))
	local test_succeeded=0
	while [ "$(date +%s000)" -lt $deadline ]; do
	    local cmd_result=
        cmd_result=$( os::cmd::internal::run_collecting_output "${cmd}"; echo $? )
		local test_result
		test_result=$( os::cmd::internal::run_collecting_output 'grep -Eq "'"${text}"'" <(os::cmd::internal::get_last_results)'; echo $? )
		test_succeeded=$( ${test_eval_func} "${test_result}"; echo $? )

		if (( test_succeeded )); then
			break
		fi
		sleep "${interval}"
		os::cmd::internal::mark_attempt
	done

	local end_time
    end_time=$(os::cmd::internal::seconds_since_epoch)
	local time_elapsed
    time_elapsed=$(echo "scale=9; ${end_time} - ${start_time}" | bc | xargs printf '%5.3f') # in decimal seconds, we need leading zeroes for parsing later

    # clear the preamble so we can print out the success or error message
    os::text::clear_string "${preamble}"

	local return_code
	if (( test_succeeded )); then
		os::text::print_green "SUCCESS after ${time_elapsed}s: ${description}"
		junit_log+=( "SUCCESS after ${time_elapsed}s: ${description//$'\n'/;}" )

		if [[ -n ${VERBOSE-} ]]; then
			os::cmd::internal::print_results
		fi
		return_code=0
	else
		os::text::print_red_bold "FAILURE after ${time_elapsed}s: ${description}: the command timed out"
		junit_log+=( "FAILURE after ${time_elapsed}s: ${description//$'\n'/;}: the command timed out" )

		os::text::print_red "$(os::cmd::internal::print_results)"
		return_code=1
	fi

	junit_log+=( "$(os::cmd::internal::print_results)" )
	( IFS=$'\n'; echo "${junit_log[*]}" >> "${JUNIT_REPORT_OUTPUT:-/dev/null}" )
	os::test::junit::declare_test_end
	return "${return_code}"
}
readonly -f os::cmd::internal::run_until_text
