#!/bin/sh

# Copyright 2017 The Kubernetes Authors.
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

# This driver is especially designed to test a long mounting scenario
# which can cause a volume to be detached while mount is in progress.


FLEX_DUMMY_LOG=${FLEX_DUMMY_LOG:-"/tmp/flex-dummy.log"}

VALID_MNTDEVICE=foo

# attach always returns one valid mount device so a different device
# showing up in a subsequent driver call implies a bug
validateMountDeviceOrDie() {
        MNTDEVICE=$1
        CALL=$2
        if [ "$MNTDEVICE" != "$VALID_MNTDEVICE" ]; then
                log "{\"status\":\"Failure\",\"message\":\"call "${CALL}" expected device "${VALID_MNTDEVICE}", got device "${MNTDEVICE}"\"}"
                exit 0
        fi
}

log() {
        printf "$*" >&1
}

debug() {
        echo "$(date) $*" >> "${FLEX_DUMMY_LOG}"
}

attach() {
        debug "attach $@"
        log "{\"status\":\"Success\",\"device\":\""${VALID_MNTDEVICE}"\"}"
        exit 0
}

detach() {
        debug "detach $@"
        # TODO issue 44737 detach is passed PV name, not mount device
        log "{\"status\":\"Success\"}"
        exit 0
}

waitforattach() {
        debug "waitforattach $@"
        MNTDEVICE=$1
        validateMountDeviceOrDie "$MNTDEVICE" "waitforattach"
        log "{\"status\":\"Success\",\"device\":\""${MNTDEVICE}"\"}"
        exit 0
}

isattached() {
        debug "isattached $@"
        log "{\"status\":\"Success\",\"attached\":true}"
        exit 0
}

domountdevice() {
        debug "domountdevice $@"
        MNTDEVICE=$2
        validateMountDeviceOrDie "$MNTDEVICE" "domountdevice"
        MNTPATH=$1
        mkdir -p ${MNTPATH} >/dev/null 2>&1
        mount -t tmpfs none ${MNTPATH} >/dev/null 2>&1
        sleep 120
        echo "Hello from flexvolume!" >> "${MNTPATH}/index.html"
        log "{\"status\":\"Success\"}"
        exit 0
}

unmountdevice() {
        debug "unmountdevice $@"
        MNTPATH=$1
        rm "${MNTPATH}/index.html" >/dev/null 2>&1
        umount ${MNTPATH} >/dev/null 2>&1
        log "{\"status\":\"Success\"}"
        exit 0
}

expandvolume() {
        debug "expandvolume $@"
        log "{\"status\":\"Success\"}"
        exit 0
}

expandfs() {
        debug "expandfs $@"
        log "{\"status\":\"Success\"}"
        exit 0
}

op=$1

if [ "$op" = "init" ]; then
        debug "init $@"
        log "{\"status\":\"Success\",\"capabilities\":{\"attach\":true, \"requiresFSResize\":true}}"
        exit 0
fi

shift

case "$op" in
        attach)
                attach $*
                ;;
        detach)
                detach $*
                ;;
        waitforattach)
                waitforattach $*
                ;;
        isattached)
                isattached $*
                ;;
        mountdevice)
                domountdevice $*
                ;;
        unmountdevice)
                unmountdevice $*
                ;;
        expandvolume)
                expandvolume $*
                ;;
        expandfs)
                expandfs $*
                ;;
        *)
                log "{\"status\":\"Not supported\"}"
                exit 0
esac

exit 1
