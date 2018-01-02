#!/bin/sh

if [ -f /backupdb/heketi.db.gz ] ; then
    gunzip -c /backupdb/heketi.db.gz > /var/lib/heketi/heketi.db
    if [ $? -ne 0 ] ; then
        echo "Unable to copy database"
        exit 1
    fi
    echo "Copied backup db to /var/lib/heketi/heketi.db"
elif [ -f /backupdb/heketi.db ] ; then
    cp /backupdb/heketi.db /var/lib/heketi/heketi.db
    if [ $? -ne 0 ] ; then
        echo "Unable to copy database"
        exit 1
    fi
    echo "Copied backup db to /var/lib/heketi/heketi.db"
fi

/usr/bin/heketi --config=/etc/heketi/heketi.json
