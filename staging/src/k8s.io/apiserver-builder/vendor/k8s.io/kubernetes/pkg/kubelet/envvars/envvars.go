/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api/v1"
)

// FromServices builds environment variables that a container is started with,
// which tell the container where to find the services it may need, which are
// provided as an argument.
func FromServices(services []*v1.Service) []v1.EnvVar {
	var result []v1.EnvVar
	for i := range services {
		service := services[i]

		// ignore services where ClusterIP is "None" or empty
		// the services passed to this method should be pre-filtered
		// only services that have the cluster IP set should be included here
		if !v1.IsServiceIPSet(service) {
			continue
		}

		// Host
		name := makeEnvVariableName(service.Name) + "_SERVICE_HOST"
		result = append(result, v1.EnvVar{Name: name, Value: service.Spec.ClusterIP})
		// First port - give it the backwards-compatible name
		name = makeEnvVariableName(service.Name) + "_SERVICE_PORT"
		result = append(result, v1.EnvVar{Name: name, Value: strconv.Itoa(int(service.Spec.Ports[0].Port))})
		// All named ports (only the first may be unnamed, checked in validation)
		for i := range service.Spec.Ports {
			sp := &service.Spec.Ports[i]
			if sp.Name != "" {
				pn := name + "_" + makeEnvVariableName(sp.Name)
				result = append(result, v1.EnvVar{Name: pn, Value: strconv.Itoa(int(sp.Port))})
			}
		}
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

func makeLinkVariables(service *v1.Service) []v1.EnvVar {
	prefix := makeEnvVariableName(service.Name)
	all := []v1.EnvVar{}
	for i := range service.Spec.Ports {
		sp := &service.Spec.Ports[i]

		protocol := string(v1.ProtocolTCP)
		if sp.Protocol != "" {
			protocol = string(sp.Protocol)
		}
		if i == 0 {
			// Docker special-cases the first port.
			all = append(all, v1.EnvVar{
				Name:  prefix + "_PORT",
				Value: fmt.Sprintf("%s://%s:%d", strings.ToLower(protocol), service.Spec.ClusterIP, sp.Port),
			})
		}
		portPrefix := fmt.Sprintf("%s_PORT_%d_%s", prefix, sp.Port, strings.ToUpper(protocol))
		all = append(all, []v1.EnvVar{
			{
				Name:  portPrefix,
				Value: fmt.Sprintf("%s://%s:%d", strings.ToLower(protocol), service.Spec.ClusterIP, sp.Port),
			},
			{
				Name:  portPrefix + "_PROTO",
				Value: strings.ToLower(protocol),
			},
			{
				Name:  portPrefix + "_PORT",
				Value: strconv.Itoa(int(sp.Port)),
			},
			{
				Name:  portPrefix + "_ADDR",
				Value: service.Spec.ClusterIP,
			},
		}...)
	}
	return all
}
