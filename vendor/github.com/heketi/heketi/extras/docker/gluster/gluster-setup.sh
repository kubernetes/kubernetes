#!/bin/bash

###
# Description: Script to move the glusterfs initial setup to bind mounted directories of Atomic Host.
# Copyright (c) 2016 Red Hat, Inc. <http://www.redhat.com>
#
# This file is part of GlusterFS.
#
# This file is licensed to you under your choice of the GNU Lesser
# General Public License, version 3 or any later version (LGPLv3 or
# later), or the GNU General Public License, version 2 (GPLv2), in all
# cases as published by the Free Software Foundation.
###

main () {
  if test "$(ls /var/lib/heketi/fstab)"
  then
        mount -a --fstab /var/lib/heketi/fstab
        if [ $? -eq 1 ]
        then
              echo "mount failed"
              exit 1
        fi
        echo "Mount Successful"
  else
        echo "heketi-fstab not found"
  fi
  DIR_1="/etc/glusterfs"
  DIR_2="/var/log/glusterfs"
  DIR_3="/var/lib/glusterd"
  var=0
  for i in $DIR_1 $DIR_2 $DIR_3
  do
    if test "$(ls $i)"
    then
      echo "$i is not empty"
      var=$((var+1))
    fi
  done

  if [ $var -eq 3 ]
  then
        exit 1
  fi
  
  cp -r /etc/glusterfs_bkp/* /etc/glusterfs
  if [ $? -eq 1 ]
  then
	echo "Failed to copy $DIR_1" 
        exit 1
  fi

  cp -r /var/log/glusterfs_bkp/* /var/log/glusterfs
  if [ $? -eq 1 ]
  then
	echo "Failed to copy $DIR_2"
        exit 1
  fi

  cp -r /var/lib/glusterd_bkp/* /var/lib/glusterd
  if [ $? -eq 1 ]
  then
	echo "Failed to copy $DIR_3"
	exit 1
  fi
 
  echo "Script Ran Successfully"
}
main

