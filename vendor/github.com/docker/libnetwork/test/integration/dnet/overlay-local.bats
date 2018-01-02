# -*- mode: sh -*-
#!/usr/bin/env bats

load helpers

function test_overlay_local() {
    dnet_suffix=$1

    echo $(docker ps)

    start=1
    end=3
    for i in `seq ${start} ${end}`;
    do
	echo "iteration count ${i}"
	dnet_cmd $(inst_id2port $i) network create -d overlay --id=mhid --subnet=10.1.0.0/16 --ip-range=10.1.${i}.0/24 --opt=com.docker.network.driver.overlay.vxlanid_list=1024 multihost
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
	    #runc $(dnet_container_name $i $dnet_suffix) $(get_sbox_id ${i} container_${i}) "ping -c 1 10.1.${j}.1"
	    runc $(dnet_container_name $i $dnet_suffix) $(get_sbox_id ${i} container_${i}) "ping -c 1 container_${j}"
	done
    done

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

@test "Test overlay network in local scope" {
    skip_for_circleci
    test_overlay_local local
}

#"ping -c 1 10.1.${j}.1"
