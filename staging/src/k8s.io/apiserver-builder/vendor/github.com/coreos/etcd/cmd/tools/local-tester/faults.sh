#!/bin/bash

PROCFILE="tools/local-tester/Procfile"
HTTPFAIL=(127.0.0.1:11180 127.0.0.1:22280 127.0.0.1:33380)

function wait_time {
	expr $RANDOM % 10 + 1
}

function cycle {
	for a; do
		echo "cycling $a"
		goreman -f $PROCFILE run stop $a || echo "could not stop $a"
		sleep `wait_time`s
		goreman -f $PROCFILE run restart $a || echo "could not restart $a"
	done
}

function cycle_members {
	cycle etcd1 etcd2 etcd3
}
function cycle_pbridge {
	cycle pbridge1 pbridge2 pbridge3
}
function cycle_cbridge {
	cycle cbridge1 cbridge2 cbridge3
}
function cycle_stresser {
	cycle stress-put
}

function kill_maj {
	idx="etcd"`expr $RANDOM % 3 + 1`
	idx2="$idx"
	while [ "$idx" == "$idx2" ]; do
		idx2="etcd"`expr $RANDOM % 3 + 1`
	done
	echo "kill majority $idx $idx2"
	goreman -f $PROCFILE run stop $idx || echo "could not stop $idx"
	goreman -f $PROCFILE run stop $idx2 || echo "could not stop $idx2"
	sleep `wait_time`s
	goreman -f $PROCFILE run restart $idx || echo "could not restart $idx"
	goreman -f $PROCFILE run restart $idx2 || echo "could not restart $idx2"
}

function kill_all {
	for a in etcd1 etcd2 etcd3; do
		goreman -f $PROCFILE run stop $a || echo "could not stop $a"
	done
	sleep `wait_time`s
	for a in etcd1 etcd2 etcd3; do
		goreman -f $PROCFILE run restart $a || echo "could not restart $a"
	done
}

function rand_fp {
	echo "$FAILPOINTS" | sed `expr $RANDOM % $NUMFPS + 1`"q;d"
}

# fp_activate <http> <fppath> <value>
function fp_activate {
	curl "$1"/"$2" -XPUT -d "$3" >/dev/null 2>&1
}

function fp_rand_single {
	fp=`rand_fp`
	fp_activate ${HTTPFAIL[`expr $RANDOM % ${#HTTPFAIL[@]}`]} $fp 'panic("'$fp'")'
	sleep `wait_time`s
}

function fp_rand_all {
	fp=`rand_fp`
	for a in `seq ${#HTTPFAIL[@]}`; do fp_activate ${HTTPFAIL[$a]} "$fp" 'panic("'$fp'")'; done
	sleep `wait_time`s
}

function fp_all_rand_fire {
	for fp in $FAILPOINTS; do
		for url in "${HTTPFAIL[@]}"; do
			fp_activate "$url" "$fp" '0.5%panic("0.5%'$fp'")'
		done
	done
}

function choose {
	fault=${FAULTS[`expr $RANDOM % ${#FAULTS[@]}`]}
	echo $fault
	$fault || echo "failed: $fault"
}

sleep 2s

FAULTS=(cycle_members kill_maj kill_all cycle_pbridge cycle_cbridge cycle_stresser)

# add failpoint faults if available
FAILPOINTS=`curl http://"${HTTPFAIL[0]}" 2>/dev/null | cut -f1 -d'=' | grep -v "^$"`
NUMFPS=`echo $(echo "$FAILPOINTS" | wc -l)`
if [ "$NUMFPS" != "0" ]; then
	FAULTS+=(fp_rand_single)
	FAULTS+=(fp_rand_all)
fi

while [ 1 ]; do
	choose
	# start any nodes that have been killed by failpoints
	for a in etcd1 etcd2 etcd3; do goreman -f $PROCFILE run start $a; done
	fp_all_rand_fire
done
