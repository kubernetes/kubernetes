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

export MASTER="spark://spark-master:7077"
export SPARK_HOME=/opt/spark
export ZEPPELIN_JAVA_OPTS="-Dspark.jars=/opt/spark/lib/gcs-connector-latest-hadoop2.jar"
# TODO(zmerlynn): Setting global CLASSPATH *should* be unnecessary,
# but ZEPPELIN_JAVA_OPTS isn't enough here. :(
export CLASSPATH="/opt/spark/lib/gcs-connector-latest-hadoop2.jar"
export ZEPPELIN_NOTEBOOK_DIR="${ZEPPELIN_HOME}/notebook"
export ZEPPELIN_MEM=-Xmx1024m
export ZEPPELIN_PORT=8080
export PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.8.2.1-src.zip"
