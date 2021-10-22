# Copyright 2013 Prometheus Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

KEY_ID ?= _DEFINE_ME_

all: go

SUFFIXES:

go: go/metrics.pb.go

go/metrics.pb.go: metrics.proto
	protoc $< --go_out=import_path=github.com/prometheus/client_model/,paths=source_relative:go/

clean:
	-rm -rf go/*

.PHONY: all clean go
