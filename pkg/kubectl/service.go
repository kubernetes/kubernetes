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

package kubectl

import (
	"fmt"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type ServiceGenerator struct{}

func (ServiceGenerator) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"selector", true},
		{"port", true},
		{"public-ip", false},
		{"create-external-load-balancer", false},
		{"protocol", false},
		{"container-port", false}, // alias of target-port
		{"target-port", false},
	}
}

func (ServiceGenerator) Generate(params map[string]string) (runtime.Object, error) {
	selectorString, found := params["selector"]
	if !found || len(selectorString) == 0 {
		return nil, fmt.Errorf("'selector' is a required parameter.")
	}
	selector := ParseLabels(selectorString)

	labelsString, found := params["labels"]
	var labels map[string]string
	if found && len(labelsString) > 0 {
		labels = ParseLabels(labelsString)
	}

	name, found := params["name"]
	if !found {
		return nil, fmt.Errorf("'name' is a required parameter.")
	}
	portString, found := params["port"]
	if !found {
		return nil, fmt.Errorf("'port' is a required parameter.")
	}
	port, err := strconv.Atoi(portString)
	if err != nil {
		return nil, err
	}
	service := api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: api.ServiceSpec{
			Port:     port,
			Protocol: api.Protocol(params["protocol"]),
			Selector: selector,
		},
	}
	targetPort, found := params["target-port"]
	if !found {
		targetPort, found = params["container-port"]
	}
	if found && len(targetPort) > 0 {
		if portNum, err := strconv.Atoi(targetPort); err != nil {
			service.Spec.TargetPort = util.NewIntOrStringFromString(targetPort)
		} else {
			service.Spec.TargetPort = util.NewIntOrStringFromInt(portNum)
		}
	} else {
		service.Spec.TargetPort = util.NewIntOrStringFromInt(port)
	}
	if params["create-external-load-balancer"] == "true" {
		service.Spec.CreateExternalLoadBalancer = true
	}
	if len(params["public-ip"]) != 0 {
		service.Spec.PublicIPs = []string{params["public-ip"]}
	}
	return &service, nil
}
