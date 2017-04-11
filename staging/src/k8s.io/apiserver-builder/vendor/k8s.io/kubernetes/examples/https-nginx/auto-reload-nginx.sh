#!/bin/sh

# Copyright 2016 The Kubernetes Authors.
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

nginx "$@"
oldcksum=`cksum /etc/nginx/conf.d/default.conf`

inotifywait -e modify,move,create,delete -mr --timefmt '%d/%m/%y %H:%M' --format '%T' \
/etc/nginx/conf.d/ | while read date time; do

	newcksum=`cksum /etc/nginx/conf.d/default.conf`
	if [ "$newcksum" != "$oldcksum" ]; then
		echo "At ${time} on ${date}, config file update detected."
		oldcksum=$newcksum
		nginx -s reload
	fi

done
