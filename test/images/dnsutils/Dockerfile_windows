# Copyright 2019 The Kubernetes Authors.
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

FROM BASEIMAGE

USER ContainerAdministrator
ENV chocolateyUseWindowsCompression false
RUN powershell -Command \
    iex ((new-object net.webclient).DownloadString('https://chocolatey.org/install.ps1')); \
    choco feature disable --name showDownloadProgress; \
    powershell -Command choco install bind-toolsonly --version 9.10.3 -y
RUN powershell -Command "\
    wget -uri 'https://github.com/coredns/coredns/releases/download/v1.5.0/coredns_1.5.0_windows_amd64.tgz' -OutFile 'C:\coredns.tgz';\
    tar -xzvf 'C:\coredns.tgz';\
    Remove-Item 'C:\coredns.tgz'"
ADD hostname /bin/hostname.exe
