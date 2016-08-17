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

FROM ubuntu:trusty

# update the package repository and install python pip
RUN apt-get -y update && apt-get -y install python-dev python-pip

# install flower
RUN pip install flower

# Make sure we expose port 5555 so that we can connect to it
EXPOSE 5555

ADD run_flower.sh /usr/local/bin/run_flower.sh

# Running flower
CMD ["/bin/bash", "/usr/local/bin/run_flower.sh"]
