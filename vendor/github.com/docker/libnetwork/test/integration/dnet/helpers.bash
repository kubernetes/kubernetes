function get_docker_bridge_ip() {
    echo $(docker run --rm -it busybox ip route show | grep default | cut -d" " -f3)
}

function inst_id2port() {
    echo $((41000+${1}-1))
}

function dnet_container_name() {
    echo dnet-$1-$2
}

function dnet_container_ip() {
    docker inspect --format '{{.NetworkSettings.IPAddress}}' dnet-$1-$2
}

function get_sbox_id() {
    local line

    line=$(dnet_cmd $(inst_id2port ${1}) service ls | grep ${2})
    echo ${line} | cut -d" " -f5
}

function net_connect() {
	local al gl
	if [ -n "$4" ]; then
	    if [ "${4}" != ":" ]; then
		al="--alias=${4}"
	    fi
	fi
	if [ -n "$5" ]; then
	    gl="--alias=${5}"
	fi
	dnet_cmd $(inst_id2port ${1}) service publish $gl ${2}.${3}
	dnet_cmd $(inst_id2port ${1}) service attach $al ${2} ${2}.${3}
}

function net_disconnect() {
	dnet_cmd $(inst_id2port ${1}) service detach ${2} ${2}.${3}
	dnet_cmd $(inst_id2port ${1}) service unpublish ${2}.${3}
}

function start_consul() {
    stop_consul
    docker run -d \
	   --name=pr_consul \
	   -p 8500:8500 \
	   -p 8300-8302:8300-8302/tcp \
	   -p 8300-8302:8300-8302/udp \
	   -h consul \
	   progrium/consul -server -bootstrap
    sleep 2
}

function stop_consul() {
    echo "consul started"
    docker stop pr_consul || true
    # You cannot destroy a container in Circle CI. So do not attempt destroy in circleci
    if [ -z "$CIRCLECI" ]; then
	docker rm -f pr_consul || true
    fi
}

hrun() {
    local e E T oldIFS
    [[ ! "$-" =~ e ]] || e=1
    [[ ! "$-" =~ E ]] || E=1
    [[ ! "$-" =~ T ]] || T=1
    set +e
    set +E
    set +T
    output="$("$@" 2>&1)"
    status="$?"
    oldIFS=$IFS
    IFS=$'\n' lines=($output)
    [ -z "$e" ] || set -e
    [ -z "$E" ] || set -E
    [ -z "$T" ] || set -T
    IFS=$oldIFS
}

function wait_for_dnet() {
    local hport

    hport=$1
    echo "waiting on dnet to come up ..."
    for i in `seq 1 10`;
    do
	hrun ./bin/dnet -H tcp://127.0.0.1:${hport} network ls
	echo ${output}
	if [ "$status" -eq 0 ]; then
	    return
	fi

	if [[ "${lines[1]}" =~ .*EOF.* ]]
	then
	    docker logs ${2}
	fi
	echo "still waiting after ${i} seconds"
	sleep 1
    done
}

function parse_discovery_str() {
    local d provider address
    discovery=$1
    provider=$(echo ${discovery} | cut -d":" -f1)
    address=$(echo ${discovery} | cut -d":" -f2):$(echo ${discovery} | cut -d":" -f3)
    address=${address:2}
    echo "${discovery} ${provider} ${address}"
}

function start_dnet() {
    local inst suffix name hport cport hopt store bridge_ip labels tomlfile nip
    local discovery provider address

    inst=$1
    shift
    suffix=$1
    shift

    store=$(echo $suffix | cut -d":" -f1)
    nip=$(echo $suffix | cut -s -d":" -f2)


    stop_dnet ${inst} ${store}
    name=$(dnet_container_name ${inst} ${store})

    hport=$((41000+${inst}-1))
    cport=2385
    hopt=""

    while [ -n "$1" ]
    do
	if [[ "$1" =~ ^[0-9]+$ ]]
	then
	    hport=$1
	    cport=$1
	    hopt="-H tcp://0.0.0.0:${cport}"
	else
	    store=$1
	fi
	shift
    done

    bridge_ip=$(get_docker_bridge_ip)

    echo "start_dnet parsed values: " ${inst} ${suffix} ${name} ${hport} ${cport} ${hopt} ${store}

    mkdir -p /tmp/dnet/${name}
    tomlfile="/tmp/dnet/${name}/libnetwork.toml"

    # Try discovery URLs with or without path
    neigh_ip=""
    neighbors=""
    if [ "$store" = "zookeeper" ]; then
	read discovery provider address < <(parse_discovery_str zk://${bridge_ip}:2182)
    elif [ "$store" = "etcd" ]; then
	read discovery provider address < <(parse_discovery_str etcd://${bridge_ip}:42000/custom_prefix)
    elif [ "$store" = "consul" ]; then
	read discovery provider address < <(parse_discovery_str consul://${bridge_ip}:8500/custom_prefix)
    else
	if [ "$nip" != "" ]; then
	    neighbors=${nip}
	fi

	discovery=""
	provider=""
	address=""
    fi

    if [ "$discovery" != "" ]; then
	cat > ${tomlfile} <<EOF
title = "LibNetwork Configuration file for ${name}"

[daemon]
  debug = false
[cluster]
  discovery = "${discovery}"
  Heartbeat = 10
[scopes]
  [scopes.global]
    [scopes.global.client]
      provider = "${provider}"
      address = "${address}"
EOF
    else
	cat > ${tomlfile} <<EOF
title = "LibNetwork Configuration file for ${name}"

[daemon]
  debug = false
[orchestration]
  agent = true
  bind = "eth0"
  peer = "${neighbors}"
EOF
    fi

    cat ${tomlfile}
    docker run \
	   -d \
	   --hostname=$(echo ${name} | sed s/_/-/g) \
	   --name=${name}  \
	   --privileged \
	   -p ${hport}:${cport} \
	   -e _OVERLAY_HOST_MODE \
	   -v $(pwd)/:/go/src/github.com/docker/libnetwork \
	   -v /tmp:/tmp \
	   -v $(pwd)/${TMPC_ROOT}:/scratch \
	   -v /usr/local/bin/runc:/usr/local/bin/runc \
	   -w /go/src/github.com/docker/libnetwork \
	   mrjana/golang ./bin/dnet -d -D ${hopt} -c ${tomlfile}

    wait_for_dnet $(inst_id2port ${inst}) ${name}
}

function start_ovrouter() {
    local name=${1}
    local parent=${2}

    docker run \
	   -d \
	   --name=${name} \
	   --net=container:${parent} \
	   --volumes-from ${parent} \
	   -w /go/src/github.com/docker/libnetwork \
	   mrjana/golang ./cmd/ovrouter/ovrouter eth0
}

function skip_for_circleci() {
    if [ -n "$CIRCLECI" ]; then
	skip
    fi
}

function stop_dnet() {
    local name

    name=$(dnet_container_name $1 $2)
    rm -rf /tmp/dnet/${name} || true
    docker stop ${name} || true
    # You cannot destroy a container in Circle CI. So do not attempt destroy in circleci
    if [ -z "$CIRCLECI" ]; then
	docker rm -f ${name} || true
    fi
}

function dnet_cmd() {
    local hport

    hport=$1
    shift
    ./bin/dnet -H tcp://127.0.0.1:${hport} $*
}

function dnet_exec() {
    docker exec -it ${1} bash -c "trap \"echo SIGHUP\" SIGHUP; $2"
}

function runc() {
    local dnet

    dnet=${1}
    shift
    dnet_exec ${dnet} "cp /var/lib/docker/network/files/${1}*/* /scratch/rootfs/etc"
    dnet_exec ${dnet} "mkdir -p /var/run/netns"
    dnet_exec ${dnet} "touch /var/run/netns/c && mount -o bind /var/run/docker/netns/${1} /var/run/netns/c"
    dnet_exec ${dnet} "ip netns exec c unshare -fmuip --mount-proc chroot \"/scratch/rootfs\" /bin/sh -c \"/bin/mount -t proc proc /proc && ${2}\""
    dnet_exec ${dnet} "umount /var/run/netns/c && rm /var/run/netns/c"
}

function runc_nofail() {
    local dnet

    dnet=${1}
    shift
    dnet_exec ${dnet} "cp /var/lib/docker/network/files/${1}*/* /scratch/rootfs/etc"
    dnet_exec ${dnet} "mkdir -p /var/run/netns"
    dnet_exec ${dnet} "touch /var/run/netns/c && mount -o bind /var/run/docker/netns/${1} /var/run/netns/c"
    set +e
    dnet_exec ${dnet} "ip netns exec c unshare -fmuip --mount-proc chroot \"/scratch/rootfs\" /bin/sh -c \"/bin/mount -t proc proc /proc && ${2}\""
    status="$?"
    set -e
    dnet_exec ${dnet} "umount /var/run/netns/c && rm /var/run/netns/c"
}

function start_etcd() {
    local bridge_ip
    stop_etcd

    bridge_ip=$(get_docker_bridge_ip)
    docker run -d \
	   --net=host \
	   --name=dn_etcd \
	   mrjana/etcd --listen-client-urls http://0.0.0.0:42000 \
	   --advertise-client-urls http://${bridge_ip}:42000
    sleep 2
}

function stop_etcd() {
    docker stop dn_etcd || true
    # You cannot destroy a container in Circle CI. So do not attempt destroy in circleci
    if [ -z "$CIRCLECI" ]; then
	docker rm -f dn_etcd || true
    fi
}

function start_zookeeper() {
    stop_zookeeper
    docker run -d \
	   --name=zookeeper_server \
	   -p 2182:2181 \
	   -h zookeeper \
	   dnephin/docker-zookeeper:3.4.6
    sleep 2
}

function stop_zookeeper() {
    echo "zookeeper started"
    docker stop zookeeper_server || true
    # You cannot destroy a container in Circle CI. So do not attempt destroy in circleci
    if [ -z "$CIRCLECI" ]; then
	docker rm -f zookeeper_server || true
    fi
}

function test_overlay() {
    dnet_suffix=$1

    echo $(docker ps)

    start=1
    end=3
    # Setup overlay network and connect containers ot it
    if [ -z "${2}" -o "${2}" != "skip_add" ]; then
	if [ -z "${2}" -o "${2}" != "internal" ]; then
	    dnet_cmd $(inst_id2port 1) network create -d overlay multihost
	else
	    dnet_cmd $(inst_id2port 1) network create -d overlay --internal multihost
	fi
    fi

    for i in `seq ${start} ${end}`;
    do
	dnet_cmd $(inst_id2port $i) container create container_${i}
	net_connect ${i} container_${i} multihost
    done

    # Now test connectivity between all the containers using service names
    for i in `seq ${start} ${end}`;
    do
	if [ -z "${2}" -o "${2}" != "internal" ]; then
	    runc $(dnet_container_name $i $dnet_suffix) $(get_sbox_id ${i} container_${i}) \
		"ping -c 1 www.google.com"
	else
	    default_route=`runc $(dnet_container_name $i $dnet_suffix) $(get_sbox_id ${i} container_${i}) "ip route | grep default"`
	    [ "$default_route" = "" ]
	fi
	for j in `seq ${start} ${end}`;
	do
	    if [ "$i" -eq "$j" ]; then
		continue
	    fi
	    runc $(dnet_container_name $i $dnet_suffix) $(get_sbox_id ${i} container_${i}) \
		 "ping -c 1 container_$j"
	done
    done

    # Setup bridge network and connect containers ot it
    if [ -z "${2}" -o "${2}" != "skip_add" ]; then
	if [ -z "${2}" -o "${2}" != "internal" ]; then
	    dnet_cmd $(inst_id2port 1) network create -d bridge br1
	    dnet_cmd $(inst_id2port 1) network create -d bridge br2
	    net_connect ${start} container_${start} br1
	    net_connect ${start} container_${start} br2

	    # Make sure external connectivity works
	    runc $(dnet_container_name ${start} $dnet_suffix) $(get_sbox_id ${start} container_${start}) \
		"ping -c 1 www.google.com"
	    net_disconnect ${start} container_${start} br1
	    net_disconnect ${start} container_${start} br2

	    # Make sure external connectivity works
	    runc $(dnet_container_name ${start} $dnet_suffix) $(get_sbox_id ${start} container_${start}) \
		"ping -c 1 www.google.com"
	    dnet_cmd $(inst_id2port 1) network rm br1
	    dnet_cmd $(inst_id2port 1) network rm br2

	    # Disconnect from overlay network
	    net_disconnect ${start} container_${start} multihost

	    # Connect to overlay network again
	    net_connect ${start} container_${start} multihost

	    # Make sure external connectivity still works
	    runc $(dnet_container_name ${start} $dnet_suffix) $(get_sbox_id ${start} container_${start}) \
		"ping -c 1 www.google.com"
	fi
    fi

    # Teardown the container connections and the network
    for i in `seq ${start} ${end}`;
    do
	net_disconnect ${i} container_${i} multihost
	dnet_cmd $(inst_id2port $i) container rm container_${i}
    done

    if [ -z "${2}" -o "${2}" != "skip_rm" ]; then
	dnet_cmd $(inst_id2port 2) network rm multihost
    fi
}

function check_etchosts() {
    local dnet sbid retval
    dnet=${1}
    shift
    sbid=${1}
    shift

    retval="true"

    for i in $*;
    do
	run runc ${dnet} ${sbid} "cat /etc/hosts"
	if [ "$status" -ne 0 ]; then
	    retval="${output}"
	    break
	fi

	line=$(echo ${output} | grep ${i})
	if [ "${line}" == "" ]; then
	    retval="false"
	fi
    done

    echo ${retval}
}

function test_overlay_singlehost() {
    dnet_suffix=$1
    shift

    echo $(docker ps)

    start=1
    end=3
    # Setup overlay network and connect containers ot it
    dnet_cmd $(inst_id2port 1) network create -d overlay multihost
    for i in `seq ${start} ${end}`;
    do
	dnet_cmd $(inst_id2port 1) container create container_${i}
	net_connect 1 container_${i} multihost
    done

    # Now test connectivity between all the containers using service names
    for i in `seq ${start} ${end}`;
    do
	for j in `seq ${start} ${end}`;
	do
	    if [ "$i" -eq "$j" ]; then
		continue
	    fi
	    runc $(dnet_container_name 1 $dnet_suffix) $(get_sbox_id 1 container_${i}) \
		 "ping -c 1 container_$j"
	done
    done

    # Teardown the container connections and the network
    for i in `seq ${start} ${end}`;
    do
	net_disconnect 1 container_${i} multihost
	dnet_cmd $(inst_id2port 1) container rm container_${i}
    done

    dnet_cmd $(inst_id2port 1) network rm multihost
}

function test_overlay_hostmode() {
    dnet_suffix=$1
    shift

    echo $(docker ps)

    start=1
    end=2
    # Setup overlay network and connect containers ot it
    dnet_cmd $(inst_id2port 1) network create -d overlay multihost1
    dnet_cmd $(inst_id2port 1) network create -d overlay multihost2
    dnet_cmd $(inst_id2port 1) network ls

    for i in `seq ${start} ${end}`;
    do
	dnet_cmd $(inst_id2port 1) container create mh1_${i}
	net_connect 1 mh1_${i} multihost1
    done

    for i in `seq ${start} ${end}`;
    do
	dnet_cmd $(inst_id2port 1) container create mh2_${i}
	net_connect 1 mh2_${i} multihost2
    done

    # Now test connectivity between all the containers using service names
    for i in `seq ${start} ${end}`;
    do
	for j in `seq ${start} ${end}`;
	do
	    if [ "$i" -eq "$j" ]; then
		continue
	    fi

	    # Find the IP addresses of the j containers on both networks
	    hrun runc $(dnet_container_name 1 $dnet_suffix) $(get_sbox_id 1 mh1_${i}) "nslookup mh1_$j"
	    mh1_j_ip=$(echo ${output} | awk '{print $11}')

	    hrun runc $(dnet_container_name 1 $dnet_suffix) $(get_sbox_id 1 mh2_${i}) "nslookup mh2_$j"
	    mh2_j_ip=$(echo ${output} | awk '{print $11}')

	    # Ping the j containers in the same network and ensure they are successful
	    runc $(dnet_container_name 1 $dnet_suffix) $(get_sbox_id 1 mh1_${i}) \
		 "ping -c 1 mh1_$j"
	    runc $(dnet_container_name 1 $dnet_suffix) $(get_sbox_id 1 mh2_${i}) \
		 "ping -c 1 mh2_$j"

	    # Try pinging j container IPs from the container in the other network and make sure that they are not successful
	    runc_nofail $(dnet_container_name 1 $dnet_suffix) $(get_sbox_id 1 mh1_${i}) "ping -c 1 ${mh2_j_ip}"
	    [ "${status}" -ne 0 ]

	    runc_nofail $(dnet_container_name 1 $dnet_suffix) $(get_sbox_id 1 mh2_${i}) "ping -c 1 ${mh1_j_ip}"
	    [ "${status}" -ne 0 ]

	    # Try pinging the j container IPS from the host(dnet container in this case) and make syre that they are not successful
	    hrun docker exec -it $(dnet_container_name 1 $dnet_suffix) "ping -c 1 ${mh1_j_ip}"
	    [ "${status}" -ne 0 ]

	    hrun docker exec -it $(dnet_container_name 1 $dnet_suffix) "ping -c 1 ${mh2_j_ip}"
	    [ "${status}" -ne 0 ]
	done
    done

    # Teardown the container connections and the network
    for i in `seq ${start} ${end}`;
    do
	net_disconnect 1 mh1_${i} multihost1
	dnet_cmd $(inst_id2port 1) container rm mh1_${i}
    done

    for i in `seq ${start} ${end}`;
    do
	net_disconnect 1 mh2_${i} multihost2
	dnet_cmd $(inst_id2port 1) container rm mh2_${i}
    done

    dnet_cmd $(inst_id2port 1) network rm multihost1
    dnet_cmd $(inst_id2port 1) network rm multihost2
}
