#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

function start()
{
	unset gid
	# accept "-G gid" option
	while getopts "G:" opt; do
		case ${opt} in
			G) gid=${OPTARG};;
		esac
	done
	shift $(($OPTIND - 1))

	# prepare ganesha config
	# set mountd port to 20048
	cat <<-EOF >> /tmp/ganesha.conf
	NFS_Core_Param
	{
		MNT_Port = 20048;
		NFS_PROTOCOLS = 4;
	}
	EOF
	# assign unique export ids
	id=1
	# if testing group permissions, need to squash
	squash="no_root_squash"
	if [ -v gid ] ; then
		squash="root_id_squash"
	fi
	for i in "$@"; do
		mkdir -p $i
		if [ -v gid ] ; then
			chmod 071 $i
			chgrp $gid $i
		fi
		cat <<- EOF >> /tmp/ganesha.conf
		EXPORT
		{
			# Export Id (mandatory, each EXPORT must have a unique Export_Id)
			Export_Id = $id;

			Filesystem_Id = $id.$id;

			# Exported path (mandatory)
			Path = "$i";

			# Pseudo Path (required for NFS v4)
			Pseudo = "$i";

			# Required for access (default is None)
			# Could use CLIENT blocks instead
			Access_Type = RW;

			Squash = $squash;
			SecType = sys;
			Protocols = 4;
			# Exporting FSAL
			FSAL {
				Name = VFS;
			}
		}
		EOF
		# move index.html to here
		/bin/cp /tmp/index.html $i/
		chmod 644 $i/index.html
		echo "Serving $i"
		id=$((id + 1))
	done

	ganesha.nfsd -L /tmp/ganesha.log -f /tmp/ganesha.conf

	echo "NFS started"
}

function stop()
{
	echo "Stopping NFS"
	exit 0
}

trap stop TERM

start "$@"

# Ugly hack to do nothing and wait for SIGTERM
while true; do
	sleep 5
done
