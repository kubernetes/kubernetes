#!/usr/bin/env bash

# Be strict.
set -o errexit
set -o nounset
set -o pipefail

# Create aliases for simpler utility function invocations.
shopt -s expand_aliases

alias log.debug="log \"DEBG\" \${FUNCNAME[0]:-N/A} \${LINENO} "
alias  log.info="log \"INFO\" \${FUNCNAME[0]:-N/A} \${LINENO} "
alias  log.warn="log \"WARN\" \${FUNCNAME[0]:-N/A} \${LINENO} "
alias  log.fail="log \"FAIL\" \${FUNCNAME[0]:-N/A} \${LINENO} "

alias log.debugvar="log_variable DEBG \${FUNCNAME[0]:-N/A} \${LINENO} "
alias log.assertvar="__assert_variable DEBG \${FUNCNAME[0]:-N/A} \${LINENO} "

# Wrapper around log_variable that also makes sure that the variable itself is
# not set to an empty value.
__assert_variable()
{
  if [[ -z "${!4}" ]]; then
    log "$@"
    log.fail "assertion failure: variable \$${4} cannot be empty"
  fi
}

log_header()
{
  local pad_left
  local pad_right
  local msg
  local msglen
  local i

  msg="${1}"
  msglen="${#1}"

  echo >&2 "##################################################"
  if (( msglen <= 46 )); then
    pad_left=$(( (50 - msglen) / 2 - 1 ))
    pad_right=$(( 50 - msglen - pad_left - 2 ))
    for ((i=0; i < pad_left; ++i)); do
      echo >&2 -n "#"
    done
    echo >&2 -n " ${msg} "
    for ((i=0; i < pad_right; ++i)); do
      echo >&2 -n "#"
    done
    echo >&2
  else
    echo >&2 "${msg}"
  fi
  echo >&2 "##################################################"
}

log()
{
  local level
  local funcname
  local line

  level="${1}"
  funcname="${2}"
  line="${3}"

  shift 3

  echo -e >&2 "${level}: $(date -Iseconds): function \`${funcname}', line ${line}:" "$@"

  if [[ "${level}" == FAIL ]]; then
    exit 1
  fi
}

# Print a variable's value.
log_variable()
{
  local level
  local funcname
  local line
  local variable_name
  local variable_type
  local variable_value

  level="${1}"
  funcname="${2}"
  line="${3}"
  variable_name="${4}"

  shift 4

  # If variable is not set (essentially, it's undefined), then log a warning instead.
  if [[ ! -v ${variable_name} ]]; then
    log WARN "${funcname}" "${line}" "variable \$${variable_name} is not set"
    return 0
  fi

  variable_type=$(declare -p "${variable_name}")
  case "${variable_type#declare -}" in
    [aA]*)
      # Strip 11 characters for "declare -a ", and 1 character for "="
      # after the variable name.
      variable_value="${variable_type:$((11 + ${#variable_name} + 1))}"
      log "${level}" "${funcname}" "${line}" "variable \$${variable_name} is ${variable_value}"
      ;;
    *)
      if [[ "${!variable_name}" =~ .*$'\n'.* ]]; then
        log "${level}" "${funcname}" "${line}" "variable \$${variable_name} is (multiline):
----------8<----------
${!variable_name}
---------->8----------"
      else
        log "${level}" "${funcname}" "${line}" "variable \$${variable_name} is \`${!variable_name}'"
      fi
      ;;
  esac
}
