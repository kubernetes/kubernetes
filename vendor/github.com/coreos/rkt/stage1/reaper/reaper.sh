#!/usr/bin/bash
shopt -s nullglob

SYSCTL=/usr/bin/systemctl
DIAG=/diagnostic

if [ $# -eq 3 ]; then
    app=$1
    root_dir=$2
    target=$3
    status=$(${SYSCTL} show --property ExecMainStatus "${app}.service")
    status="${status#*=}"
    echo "${status}" > "/rkt/status/$app"
    if [ "${status}" != 0 ] ; then
        # The command "systemctl exit $status" sets the return value that will
        # be used when the pod exits (via shutdown.service).
        # This command is available since systemd v227. On older versions, the
        # command will fail and rkt will just exit with return code 0.
        if [ -n "$EXIT_POD" ] ; then
                ${SYSCTL} exit ${status#*=} 2>/dev/null
        fi
        # systemd EXIT_EXEC is 203 (https://github.com/systemd/systemd/blob/v230/src/basic/exit-status.h#L47)
        if [ "${status}" == 203 ]; then
            "${DIAG}" "${root_dir}" "${target}" 2>&1
        fi
    fi
    exit 0
fi
