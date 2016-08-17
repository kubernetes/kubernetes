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

FROM java:7-jre

RUN apt-get update && apt-get install -y \
	asciidoctor \
	unzip \
	--no-install-recommends \
	&& rm -rf /var/lib/apt/lists/*

# Install gradle
RUN wget -O /tmp/gradle.zip https://services.gradle.org/distributions/gradle-2.5-bin.zip \
	&& mkdir -p build/ \
	&& unzip /tmp/gradle.zip -d build/ \
	&& rm /tmp/gradle.zip \
	&& mkdir -p gradle-cache/

ENV GRADLE_USER_HOME=/gradle-cache

COPY build.gradle build/
COPY gen-swagger-docs.sh build/

# Run the script once to download the dependent java libraries into the image
RUN mkdir -p /output /swagger-source \
	&& wget https://raw.githubusercontent.com/kubernetes/kubernetes/master/api/swagger-spec/v1.json -O /swagger-source/v1.json \
	&& wget https://raw.githubusercontent.com/GoogleCloudPlatform/kubernetes/master/pkg/api/v1/register.go -O /register.go \
	&& build/gen-swagger-docs.sh v1 \
	&& rm -rf /output/* /swagger-source/* /register.go

RUN chmod -R 777 build/ \
	&& chmod -R 777 gradle-cache/

ENTRYPOINT ["build/gen-swagger-docs.sh"]
