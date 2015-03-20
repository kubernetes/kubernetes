/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package envvars

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// FromServices builds environment variables that a container is started with,
// which tell the container where to find the services it may need, which are
// provided as an argument.
func FromServices(services *api.ServiceList) []api.EnvVar {
	var result []api.EnvVar
	for _, service := range services.Items {
		// ignore services where PortalIP is "None" or empty
		// the services passed to this method should be pre-filtered
		// only services that have the portal IP set should be included here
		if !api.IsServiceIPSet(&service) {
			continue
		}
		// Host
		name := makeEnvVariableName(service.Name) + "_SERVICE_HOST"
		result = append(result, api.EnvVar{Name: name, Value: service.Spec.PortalIP})
		// Port
		name = makeEnvVariableName(service.Name) + "_SERVICE_PORT"
		result = append(result, api.EnvVar{Name: name, Value: strconv.Itoa(service.Spec.Port)})
		// Docker-compatible vars.
		result = append(result, makeLinkVariables(service)...)
	}
	return result
}

func makeEnvVariableName(str string) string {
	// TODO: If we simplify to "all names are DNS1123Subdomains" this
	// will need two tweaks:
	//   1) Handle leading digits
	//   2) Handle dots
	return strings.ToUpper(strings.Replace(str, "-", "_", -1))
}

func makeLinkVariables(service api.Service) []api.EnvVar {
	prefix := makeEnvVariableName(service.Name)
	protocol := string(api.ProtocolTCP)
	if service.Spec.Protocol != "" {
		protocol = string(service.Spec.Protocol)
	}
	portPrefix := fmt.Sprintf("%s_PORT_%d_%s", prefix, service.Spec.Port, strings.ToUpper(protocol))
	return []api.EnvVar{
		{
			Name:  prefix + "_PORT",
			Value: fmt.Sprintf("%s://%s:%d", strings.ToLower(protocol), service.Spec.PortalIP, service.Spec.Port),
		},
		{
			Name:  portPrefix,
			Value: fmt.Sprintf("%s://%s:%d", strings.ToLower(protocol), service.Spec.PortalIP, service.Spec.Port),
		},
		{
			Name:  portPrefix + "_PROTO",
			Value: strings.ToLower(protocol),
		},
		{
			Name:  portPrefix + "_PORT",
			Value: strconv.Itoa(service.Spec.Port),
		},
		{
			Name:  portPrefix + "_ADDR",
			Value: service.Spec.PortalIP,
		},
	}
}
