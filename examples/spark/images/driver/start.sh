#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

echo "$SPARK_MASTER_SERVICE_HOST spark-master" >> /etc/hosts
echo "SPARK_LOCAL_HOSTNAME=$(hostname -i)" >> /opt/spark/conf/spark-env.sh
echo "MASTER=spark://spark-master:$SPARK_MASTER_SERVICE_PORT" >> /opt/spark/conf/spark-env.sh
echo "Use kubectl exec spark-driver -it bash to invoke commands"
while true; do
  sleep 100
done
