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

RUN case $(uname -m) in \
    aarch64) \
      pip install tensorflow-aarch64==1.2; \
      ;; \
    *) \
      pip install tensorflow==1.9.0; \
      ;; \
    esac

RUN wget https://github.com/tensorflow/models/archive/v1.9.0.tar.gz \
&& tar xzf v1.9.0.tar.gz \
&& rm -f v1.9.0.tar.gz

RUN python /models-1.9.0/official/wide_deep/data_download.py

WORKDIR $HOME/models-1.9.0/official/wide_deep
ENV PYTHONPATH=$PYTHONPATH:$HOME/models-1.9.0

ENTRYPOINT python ./wide_deep.py
