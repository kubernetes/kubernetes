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

ARG SOURCE

# NOTE(claudiub): The src image is already based on nanoserver, but it seems that docker buildx
# has an issue pulling cross-registry images, and we avoid that by pulling it ourselves.
FROM --platform="windows/amd64" mcr.microsoft.com/windows/nanoserver:1809 as nanoserver
FROM --platform="windows/amd64" $SOURCE as src
FROM scratch

# NOTE(claudiub): We're basically forcing docker buildx to finish pulling the nanoserver image
# before continuing with the rest of this Dockerfile.
COPY --from=nanoserver /License.txt /License.txt

COPY --from=src /PowerShell /PowerShell
COPY --from=src /Users/ContainerAdministrator/AppData/Local/Microsoft/Windows/PowerShell/docker/ModuleAnalysisCache /Users/ContainerAdministrator/AppData/Local/Microsoft/Windows/PowerShell/docker/ModuleAnalysisCache
