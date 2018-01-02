# Jenkins CI script for Windows to Linux CI.
# Heavily modified by John Howard (@jhowardmsft) December 2015 to try to make it more reliable.
set +xe
SCRIPT_VER="Wed Apr 20 18:30:19 UTC 2016"

# TODO to make (even) more resilient: 
#  - Wait for daemon to be running before executing docker commands
#  - Check if jq is installed
#  - Make sure bash is v4.3 or later. Can't do until all Azure nodes on the latest version
#  - Make sure we are not running as local system. Can't do until all Azure nodes are updated.
#  - Error if docker versions are not equal. Can't do until all Azure nodes are updated
#  - Error if go versions are not equal. Can't do until all Azure nodes are updated.
#  - Error if running 32-bit posix tools. Probably can take from bash --version and check contains "x86_64"
#  - Warn if the CI directory cannot be deleted afterwards. Otherwise turdlets are left behind
#  - Use %systemdrive% ($SYSTEMDRIVE) rather than hard code to c: for TEMP
#  - Consider cross building the Windows binary and copy across. That's a bit of a heavy lift. Only reason
#    for doing that is that it mirrors the actual release process for docker.exe which is cross-built.
#    However, should absolutely not be a problem if built natively, so nit-picking.
#  - Tidy up of images and containers. Either here, or in the teardown script.

ec=0
uniques=1
echo INFO: Started at `date`. Script version $SCRIPT_VER


# !README!
# There are two daemons running on the remote Linux host:
# 	- outer: specified by DOCKER_HOST, this is the daemon that will build and run the inner docker daemon
#			from the sources matching the PR.
#	- inner: runs on the host network, on a port number similar to that of DOCKER_HOST but the last two digits are inverted
#			(2357 if DOCKER_HOST had port 2375; and 2367 if DOCKER_HOST had port 2376).
#			The windows integration tests are run against this inner daemon.

# get the ip, inner and outer ports.
ip="${DOCKER_HOST#*://}"
port_outer="${ip#*:}"
# inner port is like outer port with last two digits inverted.
port_inner=$(echo "$port_outer" | sed -E 's/(.)(.)$/\2\1/')
ip="${ip%%:*}"

echo "INFO: IP=$ip PORT_OUTER=$port_outer PORT_INNER=$port_inner"

# If TLS is enabled
if [ -n "$DOCKER_TLS_VERIFY" ]; then
	protocol=https
	if [ -z "$DOCKER_MACHINE_NAME" ]; then
		ec=1
		echo "ERROR: DOCKER_MACHINE_NAME is undefined"
	fi
	certs=$(echo ~/.docker/machine/machines/$DOCKER_MACHINE_NAME)
	curlopts="--cacert $certs/ca.pem --cert $certs/cert.pem --key $certs/key.pem"
	run_extra_args="-v tlscerts:/etc/docker"
	daemon_extra_args="--tlsverify --tlscacert /etc/docker/ca.pem --tlscert /etc/docker/server.pem --tlskey /etc/docker/server-key.pem"
else
	protocol=http
fi

# Save for use by make.sh and scripts it invokes
export MAIN_DOCKER_HOST="tcp://$ip:$port_inner"

# Verify we can get the remote node to respond to _ping
if [ $ec -eq 0 ]; then
	reply=`curl -s $curlopts $protocol://$ip:$port_outer/_ping`
	if [ "$reply" != "OK" ]; then
		ec=1
		echo "ERROR: Failed to get an 'OK' response from the docker daemon on the Linux node"
		echo "       at $ip:$port_outer when called with an http request for '_ping'. This implies that"
		echo "       either the daemon has crashed/is not running, or the Linux node is unavailable."
		echo
		echo "       A regular ping to the remote Linux node is below. It should reply. If not, the"
		echo "       machine cannot be reached at all and may have crashed. If it does reply, it is"
		echo "       likely a case of the Linux daemon not running or having crashed, which requires"
		echo "       further investigation."
		echo
		echo "       Try re-running this CI job, or ask on #docker-dev or #docker-maintainers"
		echo "       for someone to perform further diagnostics, or take this node out of rotation."
		echo
		ping $ip
	else
		echo "INFO: The Linux nodes outer daemon replied to a ping. Good!"
	fi 
fi

# Get the version from the remote node. Note this may fail if jq is not installed.
# That's probably worth checking to make sure, just in case.
if [ $ec -eq 0 ]; then
	remoteVersion=`curl -s $curlopts $protocol://$ip:$port_outer/version | jq -c '.Version'`
	echo "INFO: Remote daemon is running docker version $remoteVersion"
fi

# Compare versions. We should really fail if result is no 1. Output at end of script.
if [ $ec -eq 0 ]; then
	uniques=`docker version | grep Version | /usr/bin/sort -u | wc -l`
fi

# Make sure we are in repo
if [ $ec -eq 0 ]; then
	if [ ! -d hack ]; then
		echo "ERROR: Are you sure this is being launched from a the root of docker repository?"
		echo "       If this is a Windows CI machine, it should be c:\jenkins\gopath\src\github.com\docker\docker."
                echo "       Current directory is `pwd`"
		ec=1
	fi
fi

# Are we in split binary mode?
if [ `grep DOCKER_CLIENTONLY Makefile | wc -l` -gt 0 ]; then
    splitBinary=0
	echo "INFO: Running in single binary mode"
else
    splitBinary=1
	echo "INFO: Running in split binary mode"
fi


# Get the commit has and verify we have something
if [ $ec -eq 0 ]; then
	export COMMITHASH=$(git rev-parse --short HEAD)
	echo INFO: Commit hash is $COMMITHASH
	if [ -z $COMMITHASH ]; then
		echo "ERROR: Failed to get commit hash. Are you sure this is a docker repository?"
		ec=1
	fi
fi

# Redirect to a temporary location. Check is here for local runs from Jenkins machines just in case not
# in the right directory where the repo is cloned. We also redirect TEMP to not use the environment
# TEMP as when running as a standard user (not local system), it otherwise exposes a bug in posix tar which
# will cause CI to fail from Windows to Linux. Obviously it's not best practice to ever run as local system...
if [ $ec -eq 0 ]; then
	export TEMP=/c/CI/CI-$COMMITHASH
	export TMP=$TEMP
	/usr/bin/mkdir -p $TEMP  # Make sure Linux mkdir for -p
fi

# Tidy up time
if [ $ec -eq 0 ]; then
	echo INFO: Deleting pre-existing containers and images...
    
	# Force remove all containers based on a previously built image with this commit
	! docker rm -f $(docker ps -aq --filter "ancestor=docker:$COMMITHASH") &>/dev/null
    
	# Force remove any container with this commithash as a name
	! docker rm -f $(docker ps -aq --filter "name=docker-$COMMITHASH") &>/dev/null

	# This SHOULD never happen, but just in case, also blow away any containers
	# that might be around.
	! if [ ! $(docker ps -aq | wc -l) -eq 0 ]; then
		echo WARN: There were some leftover containers. Cleaning them up.
		! docker rm -f $(docker ps -aq)
	fi
	
    # Force remove the image if it exists
	! docker rmi -f "docker-$COMMITHASH" &>/dev/null
fi

# Provide the docker version for debugging purposes. If these fail, game over. 
# as the Linux box isn't responding for some reason.
if [ $ec -eq 0 ]; then
	echo INFO: Docker version and info of the outer daemon on the Linux node
	echo
	docker version
	ec=$?
	if [ 0 -ne $ec ]; then
		echo "ERROR: The main linux daemon does not appear to be running. Has the Linux node crashed?"
	fi
	echo
fi

# Same as above, but docker info
if [ $ec -eq 0 ]; then
	echo
	docker info
	ec=$?
	if [ 0 -ne $ec ]; then
		echo "ERROR: The main linux daemon does not appear to be running. Has the Linux node crashed?"
	fi
	echo
fi

# build the daemon image
if [ $ec -eq 0 ]; then
	echo "INFO: Running docker build on Linux host at $DOCKER_HOST"
	if [ $splitBinary -eq 0 ]; then
		set -x
		docker build --rm --force-rm --build-arg APT_MIRROR=cdn-fastly.deb.debian.org -t "docker:$COMMITHASH" .
    cat <<EOF | docker build --rm --force-rm -t "docker:$COMMITHASH" -
FROM docker:$COMMITHASH
RUN hack/make.sh binary
RUN cp bundles/latest/binary/docker /bin/docker 
CMD dockerd -D -H tcp://0.0.0.0:$port_inner $daemon_extra_args
EOF
	else
		set -x
		docker build --rm --force-rm --build-arg APT_MIRROR=cdn-fastly.deb.debian.org -t "docker:$COMMITHASH" .
    cat <<EOF | docker build --rm --force-rm -t "docker:$COMMITHASH" -
FROM docker:$COMMITHASH
RUN hack/make.sh binary
RUN cp bundles/latest/binary-daemon/dockerd /bin/dockerd 
CMD dockerd -D -H tcp://0.0.0.0:$port_inner $daemon_extra_args
EOF

	fi
	ec=$?
	set +x
	if [ 0 -ne $ec ]; then
		echo "ERROR: docker build failed"
	fi
fi

# Start the docker-in-docker daemon from the image we just built
if [ $ec -eq 0 ]; then
	echo "INFO: Starting build of a Linux daemon to test against, and starting it..."
	set -x
	# aufs in aufs is faster than vfs in aufs
	docker run -d $run_extra_args -e DOCKER_GRAPHDRIVER=aufs --pid host --privileged --name "docker-$COMMITHASH" --net host "docker:$COMMITHASH"
	ec=$?
	set +x
	if [ 0 -ne $ec ]; then
	    	echo "ERROR: Failed to compile and start the linux daemon"
	fi
fi

# Build locally.
if [ $ec -eq 0 ]; then
	echo "INFO: Starting local build of Windows binary..."
	set -x
	export TIMEOUT="120m"
	export DOCKER_HOST="tcp://$ip:$port_inner"
    # This can be removed
	export DOCKER_TEST_HOST="tcp://$ip:$port_inner"
	unset DOCKER_CLIENTONLY
	export DOCKER_REMOTE_DAEMON=1
	hack/make.sh binary 
	ec=$?
	set +x
	if [ 0 -ne $ec ]; then
	    echo "ERROR: Build of binary on Windows failed"
	fi
fi

# Make a local copy of the built binary and ensure that is first in our path
if [ $ec -eq 0 ]; then
	VERSION=$(< ./VERSION)
	if [ $splitBinary -eq 0 ]; then
		cp bundles/$VERSION/binary/docker.exe $TEMP
	else
		cp bundles/$VERSION/binary-client/docker.exe $TEMP
	fi
	ec=$?
	if [ 0 -ne $ec ]; then
		echo "ERROR: Failed to copy built binary to $TEMP"
	fi
	export PATH=$TEMP:$PATH
fi

# Run the integration tests
if [ $ec -eq 0 ]; then	
	echo "INFO: Running Integration tests..."
	set -x
	export DOCKER_TEST_TLS_VERIFY="$DOCKER_TLS_VERIFY"
	export DOCKER_TEST_CERT_PATH="$DOCKER_CERT_PATH"
	#export TESTFLAGS='-check.vv'
	hack/make.sh test-integration-cli
	ec=$?
	set +x
	if [ 0 -ne $ec ]; then
		echo "ERROR: CLI test failed."
		# Next line is useful, but very long winded if included
		docker -H=$MAIN_DOCKER_HOST logs --tail 100 "docker-$COMMITHASH"
    fi
fi

# Tidy up any temporary files from the CI run
if [ ! -z $COMMITHASH ]; then
	rm -rf $TEMP
fi

# CI Integrity check - ensure we are using the same version of go as present in the Dockerfile
GOVER_DOCKERFILE=`grep 'ENV GO_VERSION' Dockerfile | awk '{print $3}'`
GOVER_INSTALLED=`go version | awk '{print $3}'`
if [ "${GOVER_INSTALLED:2}" != "$GOVER_DOCKERFILE" ]; then
	#ec=1  # Uncomment to make CI fail once all nodes are updated.
	echo
	echo "---------------------------------------------------------------------------"
	echo "WARN: CI should be using go version $GOVER_DOCKERFILE, but is using ${GOVER_INSTALLED:2}"
	echo "      Please ping #docker-maintainers on IRC to get this CI server updated."
	echo "---------------------------------------------------------------------------"
	echo
fi

# Check the Linux box is running a matching version of docker
if [ "$uniques" -ne 1 ]; then
    ec=0  # Uncomment to make CI fail once all nodes are updated.
	echo
	echo "---------------------------------------------------------------------------"
	echo "ERROR: This CI node is not running the same version of docker as the daemon."
	echo "       This is a CI configuration issue."
	echo "---------------------------------------------------------------------------"
	echo
fi

# Tell the user how we did.
if [ $ec -eq 0 ]; then
	echo INFO: Completed successfully at `date`. 
else
	echo ERROR: Failed with exitcode $ec at `date`.
fi
exit $ec
