#!/bin/sh

ACTION=$1
PODNAME=$2
EXTIP=$3
PODIP=$4

LOGFILE=/tmp/$PODNAME.log

rm -f $LOGFILE

create_mapping()
{
	local CHAINBASE=$1
	local EXTIP=$2
	local PODIP=$3

	local DNATCHAIN=$CHAINBASE-DNAT
	local SNATCHAIN=$CHAINBASE-SNAT

	# Create the private chains
	iptables -t nat -N $DNATCHAIN >> $LOGFILE
	iptables -t nat -N $SNATCHAIN >> $LOGFILE

	# Start by creating the  DNAT rule on the private chain
	echo "ADDING PRIVATE CHAIN $DNATCHAIN" >> $LOGFILE
	iptables -t nat -A $DNATCHAIN -i em1 --destination $EXTIP/32 -j DNAT --to $PODIP >> $LOGFILE

	# Then setup the MASQ rule to do the SNAT
	echo "ADDING PRIVATE CHAIN $SNATCHAIN" >> $LOGFILE
	iptables -t nat -A $SNATCHAIN -o flannel.1 -j MASQUERADE >> $LOGFILE


	# Then bind them in by jumping to them from the master chains
	echo "ATTACHING NEW PRIVATE CHAINS" >> $LOGFILE
	iptables -t nat -A PREROUTING -j $DNATCHAIN >> $LOGFILE
	iptables -t nat -A POSTROUTING -j $SNATCHAIN >> $LOGFILE
}

find_rulenum()
{
	local CHAINBASE=$1
	local MAINCHAIN=$2
	local rulenum
	local found

	echo "LOOKING FOR RULES MATCHING $CHAINBASE in $MAINCHAIN" >> $LOGFILE
	rulenum=1
        iptables -t nat -L $MAINCHAIN | while read a
        do
		echo "CHECKING RULE $rulenum" >> $LOGFILE
                echo $a | grep -q $CHAINBASE
                if [ $? -eq 0 ]
                then
			found="yes"
                else
                        rulenum=$(($rulenum+1))
                fi

		if [ -n "$found" ]
		then
			rulenum=$(($rulenum-2))
			echo "$rulenum"
			break
		fi
        done


}

delete_mapping()
{
	local CHAINBASE=$1
	local rulenum
	local DNATCHAIN=$CHAINBASE-DNAT
	local SNATCHAIN=$CHAINBASE-SNAT

	echo "DELETING $CHAINBASE FROM MAIN CHAINS" >> $LOGFILE
	rulenum=$(find_rulenum $CHAINBASE PREROUTING)

	echo "DELETING FOUND PREROUTING RULE $rulenum" >> $LOGFILE

	iptables -t nat -D PREROUTING $rulenum >> $LOGFILE

	rulenum=$(find_rulenum $CHAINBASE POSTROUTING)

	echo "DELETING FOUND POSTROUTING RULE $rulenum" >> $LOGFILE

	iptables -t nat -D POSTROUTING $rulenum >> $LOGFILE

	#now flush the private chains
	iptables -t nat --flush $DNATCHAIN >> $LOGFILE
	iptables -t nat --flush $SNATCHAIN >> $LOGFILE

	# And delete the chains
	iptables -t nat -X $DNATCHAIN >> $LOGFILE
	iptables -t nat -X $SNATCHAIN >> $LOGFILE
}


#START

echo "WORKING ON $ACTION $PODNAME FROM $EXTIP to $PODIP" >> $LOGFILE

case $ACTION in

ADD)
	create_mapping $PODNAME $EXTIP $PODIP
	;;
MODIFY)
	delete_mapping $PODNAME
	create_mapping $POSNAME $EXTIP $PODIP
	;;
DELETE)
	delete_mapping $PODNAME
	;;
esac

exit 0

