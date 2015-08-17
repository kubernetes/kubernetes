/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"fmt"
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

// The only difference between ServiceGeneratorV1 and V2 is that the service port is named "default" in V1, while it is left unnamed in V2.
type ServiceGeneratorV1 struct{}

func (ServiceGeneratorV1) ParamNames() []GeneratorParam {
	return paramNames()
}

func (ServiceGeneratorV1) Generate(params map[string]string) (runtime.Object, error) {
	params["port-name"] = "default"
	return generate(params)
}

type ServiceGeneratorV2 struct{}

func (ServiceGeneratorV2) ParamNames() []GeneratorParam {
	return paramNames()
}

func (ServiceGeneratorV2) Generate(params map[string]string) (runtime.Object, error) {
	return generate(params)
}

func paramNames() []GeneratorParam {
	return []GeneratorParam{
		{"default-name", true},
		{"name", false},
		{"selector", true},
		{"port", true},
		{"labels", false},
		{"public-ip", false},
		{"create-external-load-balancer", false},
		{"type", false},
		{"protocol", false},
		{"container-port", false}, // alias of target-port
		{"target-port", false},
		{"port-name", false},
	}
}

func generate(params map[string]string) (runtime.Object, error) {
	selectorString, found := params["selector"]
	if !found || len(selectorString) == 0 {
		return nil, fmt.Errorf("'selector' is a required parameter.")
	}
	selector, err := ParseLabels(selectorString)
	if err != nil {
		return nil, err
	}

	labelsString, found := params["labels"]
	var labels map[string]string
	if found && len(labelsString) > 0 {
		labels, err = ParseLabels(labelsString)
		if err != nil {
			return nil, err
		}
	}

	name, found := params["name"]
	if !found || len(name) == 0 {
		name, found = params["default-name"]
		if !found || len(name) == 0 {
			return nil, fmt.Errorf("'name' is a required parameter.")
		}
	}
	portString, found := params["port"]
	if !found {
		return nil, fmt.Errorf("'port' is a required parameter.")
	}
	port, err := strconv.Atoi(portString)
	if err != nil {
		return nil, err
	}
	servicePortName, found := params["port-name"]
	if !found {
		// Leave the port unnamed.
		servicePortName = ""
	}
	service := api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: api.ServiceSpec{
			Selector: selector,
			Ports: []api.ServicePort{
				{
					Name:     servicePortName,
					Port:     port,
					Protocol: api.Protocol(params["protocol"]),
				},
			},
		},
	}
	targetPort, found := params["target-port"]
	if !found {
		targetPort, found = params["container-port"]
	}
	if found && len(targetPort) > 0 {
		if portNum, err := strconv.Atoi(targetPort); err != nil {
			service.Spec.Ports[0].TargetPort = util.NewIntOrStringFromString(targetPort)
		} else {
			service.Spec.Ports[0].TargetPort = util.NewIntOrStringFromInt(portNum)
		}
	} else {
		service.Spec.Ports[0].TargetPort = util.NewIntOrStringFromInt(port)
	}
	if params["create-external-load-balancer"] == "true" {
		service.Spec.Type = api.ServiceTypeLoadBalancer
	}
	if len(params["public-ip"]) != 0 {
		service.Spec.DeprecatedPublicIPs = []string{params["public-ip"]}
	}
	if len(params["type"]) != 0 {
		service.Spec.Type = api.ServiceType(params["type"])
	}
	return &service, nil
}
