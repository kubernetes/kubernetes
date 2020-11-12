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

FROM mcr.microsoft.com/windows/servercore:ltsc2019 as prep

ENV CURL_VERSION=7.57.0

ADD https://github.com/kubernetes-sigs/windows-testing/raw/master/images/busybox/busybox.exe /bin/busybox.exe
ADD https://skanthak.homepage.t-online.de/download/curl-$CURL_VERSION.cab /curl/curl.cab
ADD https://eternallybored.org/misc/netcat/netcat-win32-1.12.zip /netcat/netcat.zip
ADD https://downloads.isc.org/isc/bind9/9.14.10/BIND9.14.10.x64.zip /bind.zip

USER ContainerAdministrator

# NOTE(claudiub): We have to create relative path symlinks because docker buildx has an issue when copying
# over symlinks, it prepends "Files\" to the symlink target. "Files\C:\bin\busybox.exe" would be an
# invalid path.
RUN cd C:\bin && FOR /f "tokens=*" %i IN ('.\busybox --list') DO mklink .\%i.exe busybox.exe

RUN cd C:\curl &\
    expand /R curl.cab /F:* . &\
    del C:\curl\curl.cab &\
    cd C:\netcat &\
    tar.exe -xf netcat.zip &\
    del netcat.zip &\
    setx /M PATH "C:\bin;C:\curl\;%PATH%"

# Download bind and prepare a folder just for dig.
RUN powershell -Command "\
      Expand-Archive -Path C:\bind.zip -DestinationPath C:\bind; \
      $s = [System.Diagnostics.Process]::Start('C:\bind\vcredist_x64.exe', '/quiet'); \
      $s.WaitForExit(); \
      mkdir C:\dig; \
      cp C:\bind\dig.exe C:\dig\; \
      cp C:\bind\nslookup.exe C:\dig\; \
      cp C:\bind\*.dll C:\dig\; \
      cp C:\Windows\System32\vcruntime140.dll C:\dig\; \
      rm C:\bind.zip;"

FROM mcr.microsoft.com/windows/nanoserver:1809

COPY --from=prep /bin /bin
COPY --from=prep /curl/AMD64 /curl/CURL.LIC /curl/
COPY --from=prep /dig /dig
COPY --from=prep /netcat /netcat
