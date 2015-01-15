#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# This code below is designed to support two specific scenarios for
# using Elasticsearch and Kibana with Kubernetes. In both cases the
# environment variables PROXY_HOST and PROXY_PORT identify the instance
# of Elasticsearch to be used by Kibana. The default value for ES_HOST
# identifies the location that served the Javascript for Kibana and
# the default value of ES_PORT 5601 is the port to be used for connecting
# to Kibana. Both of these may be overriden if required. The two scenarios are:
# 1. Elasticsearch and Kibana containers running in a single pod. In this
#    case PROXY_HOST is set to the local host i.e. 127.0.0.1 and the
#    PROXY_PORT is set to 9200 because Elasticsearch is running on the
#    same name as Kibana. If KIBANA_IP is the external IP address of
#    the Kubernetes Kibna service then all requests to:
#       KIBANA_LOGGING_SERVICE:$ES_PORT/elasticsearch/XXX
#    are proxied to:
#       http://127.0.0.1:9200/XXX
# 2. Elasticsearch and Kibana are run in separate pods and Elasticsearch
#    has an IP and port exposed via a Kubernetes service. In this case
#    the Elasticsearch service *must* be called 'elasticsearch' and then
#    all requests sent to:
#       KIBANA_LOGGING_SERVICE:$ES_PORT/elasticsearch/XXX
#    are proxied to:
#       http://$ELASTICSEARCH_LOGGING_SERVICE_HOST:$ELASTICSEARCH_LOGGING_SERVICE_PORT:9200/XXX
# The proxy configuration occurs in a location block of the nginx configuration
# file /etc/nginx/sites-available/default.

set -o errexit
set -o nounset
set -o pipefail

# Report all environment variables containing 'elasticsearch'
set | grep -i elasticsearch
# Set the default value for the Elasticsearch host as seen by the client
# Javascript code for Kibana.
: ${ES_HOST:='"+window.location.hostname+"'}
echo ES_HOST=$ES_HOST
# Set the default port for Elasticsearch host as seen by the client
# Javascript for Kibana.
: ${ES_PORT:=5601}
echo ES_PORT=$ES_PORT
# Set the default host IP and port for Elasticsearch as seen by the proxy
# code in the configuration for nginx. If a Kubernetes Elasticsearch
# service called 'elasticsearch' is defined, use that. Otherwise, use
# a local instance of Elasticsearch on port 9200.
PROXY_HOST=${ELASTICSEARCH_LOGGING_SERVICE_HOST:-127.0.0.1}
echo PROXY_HOST=${PROXY_HOST}
PROXY_PORT=${ELASTICSEARCH_SERVICE_LOGGING_PORT:-9200}
echo PROXY_PORT=${PROXY_PORT}
 
# Create a config.hs that defines the Elasticsearch server to be
# at http://${ES_HOST}:${ES_PORT}/elasticsearch from the perspective of
# the client Javascript code.
cat << EOF > /usr/share/nginx/html/config.js
/** @scratch /configuration/config.js/1
 *
 * == Configuration
 * config.js is where you will find the core Kibana configuration. This file contains parameter that
 * must be set before kibana is run for the first time.
 */
define(['settings'],
function (Settings) {
  

  /** @scratch /configuration/config.js/2
   *
   * === Parameters
   */
  return new Settings({

    /** @scratch /configuration/config.js/5
     *
     * ==== elasticsearch
     *
     * The URL to your elasticsearch server. You almost certainly don't
     * want +http://localhost:9200+ here. Even if Kibana and Elasticsearch are on
     * the same host. By default this will attempt to reach ES at the same host you have
     * kibana installed on. You probably want to set it to the FQDN of your
     * elasticsearch host
     *
     * Note: this can also be an object if you want to pass options to the http client. For example:
     *
     *  +elasticsearch: {server: "http://localhost:9200", withCredentials: true}+
     *
     */
    elasticsearch: "http://${ES_HOST}:${ES_PORT}/elasticsearch",

    /** @scratch /configuration/config.js/5
     *
     * ==== default_route
     *
     * This is the default landing page when you don't specify a dashboard to load. You can specify
     * files, scripts or saved dashboards here. For example, if you had saved a dashboard called
     * WebLogs to elasticsearch you might use:
     *
     * default_route: '/dashboard/elasticsearch/WebLogs',
     */
    default_route     : '/dashboard/file/logstash.json',

    /** @scratch /configuration/config.js/5
     *
     * ==== kibana-int
     *
     * The default ES index to use for storing Kibana specific object
     * such as stored dashboards
     */
    kibana_index: "kibana-int",

    /** @scratch /configuration/config.js/5
     *
     * ==== panel_name
     *
     * An array of panel modules available. Panels will only be loaded when they are defined in the
     * dashboard, but this list is used in the "add panel" interface.
     */
    panel_names: [
      'histogram',
      'map',
      'goal',
      'table',
      'filtering',
      'timepicker',
      'text',
      'hits',
      'column',
      'trends',
      'bettermap',
      'query',
      'terms',
      'stats',
      'sparklines'
    ]
  });
});
EOF

# Proxy all calls to ...:80/elasticsearch to the location
# defined by http://${PROXY_HOST}:${PROXY_PORT}
cat <<EOF > /etc/nginx/sites-available/default
server {
        listen 80 default_server;
        listen [::]:80 default_server ipv6only=on;

        root /usr/share/nginx/html;
        index index.html index.htm;

        # Make site accessible from http://localhost/
        server_name localhost;

        location ~ /elasticsearch/?(.*)$ {
        proxy_http_version 1.1;
                proxy_set_header Upgrade \$http_upgrade;
                proxy_read_timeout 1d;
                proxy_set_header Connection "upgrade";
                proxy_pass http://${PROXY_HOST}:${PROXY_PORT}/\$1;
        }

        location / {
                # First attempt to serve request as file, then
                # as directory, then fall back to displaying a 404.
                try_files \$uri \$uri/ =404;
        }
}
EOF

exec nginx -c /etc/nginx/nginx.conf "$@"
