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

ARG BASEIMAGE
FROM $BASEIMAGE

CROSS_BUILD_COPY qemu-QEMUARCH-static /usr/bin/

RUN apt-get update && apt-get install -y wget time

RUN pip install tensorflow==2.20.0

RUN wget https://github.com/tensorflow/models/archive/v2.15.0.tar.gz \
&& tar xzf v2.15.0.tar.gz \
&& rm -f v2.15.0.tar.gz

RUN python /models-2.15.0/official/wide_deep/data_download.py

WORKDIR $HOME/models-2.15.0/official/wide_deep
ENV PYTHONPATH=$PYTHONPATH:$HOME/models-2.15.0

ENTRYPOINT python ./wide_deep.py
