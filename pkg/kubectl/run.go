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
)

type BasicReplicationController struct{}

func (BasicReplicationController) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"labels", false},
		{"default-name", false},
		{"name", true},
		{"replicas", true},
		{"image", true},
		{"port", false},
		{"hostport", false},
	}
}

func (BasicReplicationController) Generate(params map[string]string) (runtime.Object, error) {
	name, found := params["name"]
	if !found || len(name) == 0 {
		name, found = params["default-name"]
		if !found || len(name) == 0 {
			return nil, fmt.Errorf("'name' is a required parameter.")
		}
	}
	// TODO: extract this flag to a central location.
	labelString, found := params["labels"]
	var labels map[string]string
	var err error
	if found && len(labelString) > 0 {
		labels, err = ParseLabels(labelString)
		if err != nil {
			return nil, err
		}
	} else {
		labels = map[string]string{
			"run": name,
		}
	}
	count, err := strconv.Atoi(params["replicas"])
	if err != nil {
		return nil, err
	}
	controller := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: count,
			Selector: labels,
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: labels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  name,
							Image: params["image"],
						},
					},
				},
			},
		},
	}

	port := -1
	hostPort := -1
	if len(params["port"]) > 0 {
		port, err = strconv.Atoi(params["port"])
		if err != nil {
			return nil, err
		}
	}

	if len(params["hostport"]) > 0 {
		hostPort, err = strconv.Atoi(params["hostport"])
		if err != nil {
			return nil, err
		}
		if hostPort > 0 && port < 0 {
			return nil, fmt.Errorf("--hostport requires --port to be specified")
		}
	}

	// Don't include the port if it was not specified.
	if port > 0 {
		controller.Spec.Template.Spec.Containers[0].Ports = []api.ContainerPort{
			{
				ContainerPort: port,
			},
		}
		if hostPort > 0 {
			controller.Spec.Template.Spec.Containers[0].Ports[0].HostPort = hostPort
		}
	}
	return &controller, nil
}
