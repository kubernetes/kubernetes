#!/bin/bash

# Root directory of the repository.
MACHINE_ROOT=/usr/bin

PLATFORM=`uname -s | tr '[:upper:]' '[:lower:]'`
ARCH=`uname -m`

if [ "$ARCH" = "x86_64" ]; then
    ARCH="amd64"
else
    ARCH="386"
fi
MACHINE_BIN_NAME=docker-machine_$PLATFORM-$ARCH
BATS_LOG=/tmp/bats.log

touch ${BATS_LOG}
rm ${BATS_LOG}

teardown() {
  echo "$BATS_TEST_NAME
----------
$output
----------

" >> ${BATS_LOG}
}

EXTRA_ARGS_CFG='EXTRA_ARGS'
EXTRA_ARGS='--tlsverify --tlscacert=/var/lib/boot2docker/ca.pem --tlskey=/var/lib/boot2docker/server-key.pem --tlscert=/var/lib/boot2docker/server.pem --label=provider=virtualbox -H tcp://0.0.0.0:2376'
TMP_EXTRA_ARGS_FILE=/tmp/tmp_extra_args
DAEMON_CFG_FILE=${BATS_TEST_DIRNAME}/daemon.cfg
set_extra_config() {
  if [ -f ${TMP_EXTRA_ARGS_FILE} ];
  then
    rm ${TMP_EXTRA_ARGS_FILE}
  fi
  echo -n "${EXTRA_ARGS_CFG}='"  > ${TMP_EXTRA_ARGS_FILE}
  echo -n "$1 "  >> ${TMP_EXTRA_ARGS_FILE}
  echo "${EXTRA_ARGS}'"  >> ${TMP_EXTRA_ARGS_FILE}
}

if [ ! -e $MACHINE_ROOT/$MACHINE_BIN_NAME ]; then
  echo "${MACHINE_ROOT}/${MACHINE_BIN_NAME} not found"
  exit 1
fi

function machine() {
    ${MACHINE_ROOT}/$MACHINE_BIN_NAME "$@"
}
