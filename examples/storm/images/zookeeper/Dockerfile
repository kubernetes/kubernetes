# Copyright 2017 The Kubernetes Authors.
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

FROM fedora:21 

MAINTAINER Matthew Farrellee <matt@cs.wisc.edu>

RUN yum update -y; yum install -y java tar; yum install -y zookeeper; yum clean all
RUN cp /etc/zookeeper/zoo_sample.cfg /etc/zookeeper/zoo.cfg

# workaround packaging issue
RUN sed -i 's,/usr/share/java/log4j.jar,/usr/share/java/log4j-1.jar,' /usr/libexec/zkEnv.sh

EXPOSE 2181 2888 3888

ENTRYPOINT ["/bin/zkServer.sh"]
CMD ["start-foreground"]
