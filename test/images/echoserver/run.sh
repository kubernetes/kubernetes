#!/bin/sh

# Copyright 2018 The Kubernetes Authors.
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

echo "Generating self-signed cert"
mkdir -p /certs
openssl req -x509 -sha256 -nodes -days 365 -newkey rsa:2048 \
-keyout /certs/privateKey.key \
-out /certs/certificate.crt \
-subj "/C=UK/ST=Warwickshire/L=Leamington/O=OrgName/OU=IT Department/CN=example.com"

# If we're running on Windows, skip loading the Linux .so modules.
if [ "$(uname)" = "Windows_NT" ]; then
	sed -i -E "s/^(load_module modules\/ndk_http_module.so;)$/#\1/" conf/nginx.conf
	sed -i -E "s/^(load_module modules\/ngx_http_lua_module.so;)$/#\1/" conf/nginx.conf
	sed -i -E "s/^(load_module modules\/ngx_http_lua_upstream_module.so;)$/#\1/" conf/nginx.conf

	# NOTE(claudiub): on Windows, nginx will take the paths in the nginx.conf file as relative paths.
	cmd /S /C "mklink /D C:\\openresty\\certs C:\\certs"
fi

echo "Starting nginx"
nginx -g "daemon off;"
