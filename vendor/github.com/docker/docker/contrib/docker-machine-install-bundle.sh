#!/usr/bin/env bash
#
# This script installs the bundle to Docker Machine instances, for the purpose
# of testing the latest Docker with Swarm mode enabled.
# Do not use in production.
#
# Requirements (on host to run this script)
#  - bash is installed
#  - Docker Machine is installed
#  - GNU tar is installed
#
# Requirements (on Docker machine instances)
#  - Docker can be managed via one of  `systemctl`, `service`, or `/etc/init.d/docker`
#
set -e
set -o pipefail

errexit() {
    echo "$1"
    exit 1
}

BUNDLE="bundles/$(cat VERSION)"

bundle_files(){
    # prefer dynbinary if exists
    for f in dockerd docker-proxy; do
	if [ -d $BUNDLE/dynbinary-daemon ]; then
	    echo $BUNDLE/dynbinary-daemon/$f
	else
	    echo $BUNDLE/binary-daemon/$f
	fi
    done
    for f in docker-containerd docker-containerd-ctr docker-containerd-shim docker-init docker-runc; do
	echo $BUNDLE/binary-daemon/$f
    done
    if [ -d $BUNDLE/dynbinary-client ]; then
	echo $BUNDLE/dynbinary-client/docker
    else
	echo $BUNDLE/binary-client/docker
    fi
}

control_docker(){
    m=$1; op=$2
    # NOTE: `docker-machine ssh $m sh -c "foo bar"` does not work
    #       (but `docker-machine ssh $m sh -c "foo\ bar"` works)
    #       Anyway we avoid using `sh -c` here for avoiding confusion
    cat <<EOF | docker-machine ssh $m sudo sh
if command -v systemctl > /dev/null; then
  systemctl $op docker
elif command -v service > /dev/null; then
  service docker $op
elif [ -x /etc/init.d/docker ]; then
  /etc/init.d/docker $op
else
  echo "not sure how to control the docker daemon"
  exit 1
fi
EOF
}

detect_prefix(){
    m=$1
    script='dirname $(dirname $(which dockerd))'
    echo $script | docker-machine ssh $m sh
}

install_to(){
    m=$1; shift; files=$@
    echo "$m: detecting docker"
    prefix=$(detect_prefix $m)
    echo "$m: detected docker on $prefix"
    echo "$m: stopping docker"
    control_docker $m stop
    echo "$m: installing docker"
    # NOTE: GNU tar is required because we use --transform here
    # TODO: compression (should not be default)
    tar ch --transform 's/.*\///' $files | docker-machine ssh $m sudo tar Cx $prefix/bin
    echo "$m: starting docker"
    control_docker $m start
    echo "$m: done"
}

check_prereq(){
    command -v docker-machine > /dev/null || errexit "docker-machine not installed"
    ( tar --version | grep GNU > /dev/null ) || errexit "GNU tar not installed"
}

case "$1" in
    "install")
	shift; machines=$@
	check_prereq
	files=$(bundle_files)
	echo "Files to be installed:"
	for f in $files; do echo $f; done
	pids=()
	for m in $machines; do
	    install_to $m $files &
	    pids+=($!)
	done
	status=0
	for pid in ${pids[@]}; do
	    wait $pid || { status=$?; echo "background process $pid failed with exit status $status"; }
	done
	exit $status
	;;
    *)
	errexit "Usage: $0 install MACHINES"
	;;
esac
