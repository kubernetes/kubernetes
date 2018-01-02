# -*- mode: sh -*-
#!/usr/bin/env bats

load helpers

function is_network_exist() {
    line=$(dnet_cmd $(inst_id2port $1) network ls | grep ${2})
    name=$(echo ${line} | cut -d" " -f2)
    driver=$(echo ${line} | cut -d" " -f3)
    if [ "$name" == "$2"  -a  "$driver" == "$3" ]; then
	echo "true"
    else
	echo "false"
    fi
}

@test "Test multinode network create" {
    echo $(docker ps)
    for i in `seq 1 3`;
    do
	oname="mh$i"
	run dnet_cmd $(inst_id2port $i) network create -d test ${oname}
	echo ${output}
	[ "$status" -eq 0 ]

	for j in `seq 1 3`;
	do
	    result=$(is_network_exist $j ${oname} test)
	    [ "$result" = "true" ]
	done

	# Always try to remove the network from the second node
	dnet_cmd $(inst_id2port 2) network rm ${oname}
	echo "delete ${oname}"
	nresult=$(is_network_exist 1 ${oname} test)
	echo ${nresult}
	dnet_cmd $(inst_id2port 1) network ls
	[ "$nresult" = "false" ]
    done
}

@test "Test multinode service create" {
    echo $(docker ps)
    dnet_cmd $(inst_id2port 1) network create -d test multihost
    for i in `seq 1 3`;
    do
	oname="svc$i"
	run dnet_cmd $(inst_id2port $i) service publish ${oname}.multihost
	echo ${output}
	[ "$status" -eq 0 ]

	for j in `seq 1 3`;
	do
	    run dnet_cmd $(inst_id2port $j) service ls
	    [ "$status" -eq 0 ]
	    echo ${output}
	    echo ${lines[1]}
	    svc=$(echo ${lines[1]} | cut -d" " -f2)
	    network=$(echo ${lines[1]} | cut -d" " -f3)
	    echo ${svc} ${network}
	    [ "$network" = "multihost" ]
	    [ "$svc" = "${oname}" ]
	done
	dnet_cmd $(inst_id2port 2) service unpublish ${oname}.multihost
    done
    dnet_cmd $(inst_id2port 3) network rm multihost
}

@test "Test multinode service attach" {
    echo $(docker ps)
    dnet_cmd $(inst_id2port 2) network create -d test multihost
    dnet_cmd $(inst_id2port 3) service publish svc.multihost
    for i in `seq 1 3`;
    do
	dnet_cmd $(inst_id2port $i) container create container_${i}
	dnet_cmd $(inst_id2port $i) service attach container_${i} svc.multihost
	run dnet_cmd $(inst_id2port $i) service ls
	[ "$status" -eq 0 ]
	echo ${output}
	echo ${lines[1]}
	container=$(echo ${lines[1]} | cut -d" " -f4)
	[ "$container" = "container_$i" ]
	for j in `seq 1 3`;
	do
	    if [ "$j" = "$i" ]; then
		continue
	    fi
	    dnet_cmd $(inst_id2port $j) container create container_${j}
	    run dnet_cmd $(inst_id2port $j) service attach container_${j} svc.multihost
	    echo ${output}
	    [ "$status" -ne 0 ]
	    dnet_cmd $(inst_id2port $j) container rm container_${j}
	done
	dnet_cmd $(inst_id2port $i) service detach container_${i} svc.multihost
	dnet_cmd $(inst_id2port $i) container rm container_${i}
    done
    dnet_cmd $(inst_id2port 1) service unpublish svc.multihost
    dnet_cmd $(inst_id2port 3) network rm multihost
}

@test "Test multinode network and service delete" {
    echo $(docker ps)
    for i in `seq 1 3`;
    do
	oname="mh$i"
	osvc="svc$i"
	dnet_cmd $(inst_id2port $i) network create -d test ${oname}
	dnet_cmd $(inst_id2port $i) service publish ${osvc}.${oname}
	dnet_cmd $(inst_id2port $i) container create container_${i}
	dnet_cmd $(inst_id2port $i) network ls
	dnet_cmd $(inst_id2port $i) service attach container_${i} ${osvc}.${oname}

	for j in `seq 1 3`;
	do
	    run dnet_cmd $(inst_id2port $i) service unpublish ${osvc}.${oname}
	    echo ${output}
	    [ "$status" -ne 0 ]
	    run dnet_cmd $(inst_id2port $j) network rm ${oname}
	    echo ${output}
	    [ "$status" -ne 0 ]
	done

	dnet_cmd $(inst_id2port $i) service detach container_${i} ${osvc}.${oname}
	dnet_cmd $(inst_id2port $i) container rm container_${i}

	# Always try to remove the service from different nodes
	dnet_cmd $(inst_id2port 2) service unpublish ${osvc}.${oname}
	dnet_cmd $(inst_id2port 3) network rm ${oname}
    done
}
