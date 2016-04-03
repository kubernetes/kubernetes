#!/usr/bin/bash
shopt -s nullglob

SYSCTL=/usr/bin/systemctl

if [ $# -eq 1 ]; then
    app=$1
    status=$(${SYSCTL} show --property ExecMainStatus "${app}.service")
    echo "${status#*=}" > "/rkt/status/$app"
    if [ "${status#*=}" != 0 ] ; then
        # The command "systemctl exit $status" sets the return value that will
        # be used when the pod exits (via shutdown.service).
        # This command is available since systemd v227. On older versions, the
        # command will fail and rkt will just exit with return code 0.
        ${SYSCTL} exit ${status#*=} 2>/dev/null
    fi
    exit 0
fi
