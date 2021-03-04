# Copyright 2020 The Kubernetes Authors.
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

ARG OS_VERSION
FROM --platform=windows/amd64 mcr.microsoft.com/windows/servercore:$OS_VERSION as prep
FROM scratch

COPY --from=prep /Windows/System32/en-US/nltest.exe.mui /Windows/System32/en-US/nltest.exe.mui
COPY --from=prep /Windows/System32/nltest.exe /Windows/System32/nltest.exe
COPY --from=prep /Windows/System32/netapi32.dll /Windows/System32/netapi32.dll
COPY --from=prep /Windows/System32/ntdsapi.dll /Windows/System32/ntdsapi.dll
