// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package collector

import "github.com/google/cadvisor/container"

func (ec *EndpointConfig) configure(containerHandler container.ContainerHandler) {
	// If the exact URL was not specified, generate it based on the ip address of the container.
	if ec.URL == "" {
		ipAddress := containerHandler.GetContainerIPAddress()
		ec.URL = ec.URLConfig.Protocol + "://" + ipAddress + ":" + ec.URLConfig.Port.String() + ec.URLConfig.Path
	}
}
