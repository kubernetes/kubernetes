# Copyright The Kubernetes Authors.
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

# Install time, curl, and g++ (g++ required for torch.compile() inductor backend)
RUN apt-get update && apt-get install -y --no-install-recommends time curl g++ && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch and numpy from PyPI with pinned versions
RUN pip install --no-cache-dir torch==2.6.0 numpy==2.2.0

# Download Census Income dataset (same as TensorFlow wide_deep benchmark)
# Using HTTPS and verifying checksums for security
# Checksums from UCI ML Repository
RUN mkdir -p /data && \
    curl --retry 5 --retry-delay 5 -fsSL -o /data/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data && \
    curl --retry 5 --retry-delay 5 -fsSL -o /data/adult.test https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test && \
    echo "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d /data/adult.data" | sha256sum -c - && \
    echo "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05 /data/adult.test" | sha256sum -c - && \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY train_wide_deep.py /train_wide_deep.py

WORKDIR /

ENTRYPOINT ["time", "-p", "python", "/train_wide_deep.py"]
