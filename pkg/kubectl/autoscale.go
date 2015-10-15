/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/runtime"
)

const (
	scaleSubResource = "scale"
)

type HorizontalPodAutoscalerV1Beta1 struct{}

func (HorizontalPodAutoscalerV1Beta1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"default-name", true},
		{"name", false},
		{"scaleRef-kind", false},
		{"scaleRef-namespace", false},
		{"scaleRef-name", false},
		{"scaleRef-apiVersion", false},
		{"scaleRef-subresource", false},
		{"min", false},
		{"max", true},
		{"cpu-percent", false},
	}
}

func (HorizontalPodAutoscalerV1Beta1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}

	name, found := params["name"]
	if !found || len(name) == 0 {
		name, found = params["default-name"]
		if !found || len(name) == 0 {
			return nil, fmt.Errorf("'name' is a required parameter.")
		}
	}
	minString, found := params["min"]
	min := -1
	var err error
	if found {
		if min, err = strconv.Atoi(minString); err != nil {
			return nil, err
		}
	}
	maxString, found := params["max"]
	if !found {
		return nil, fmt.Errorf("'max' is a required parameter.")
	}
	max, err := strconv.Atoi(maxString)
	if err != nil {
		return nil, err
	}
	cpuString, found := params["cpu-percent"]
	cpu := -1
	if found {
		if cpu, err = strconv.Atoi(cpuString); err != nil {
			return nil, err
		}
	}

	scaler := extensions.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: extensions.HorizontalPodAutoscalerSpec{
			ScaleRef: extensions.SubresourceReference{
				Kind:        params["scaleRef-kind"],
				Namespace:   params["scaleRef-namespace"],
				Name:        params["scaleRef-name"],
				APIVersion:  params["scaleRef-apiVersion"],
				Subresource: scaleSubResource,
			},
			MaxReplicas: max,
		},
	}
	if min > 0 {
		scaler.Spec.MinReplicas = &min
	}
	if cpu >= 0 {
		scaler.Spec.CPUUtilization = &extensions.CPUTargetUtilization{cpu}
	}
	return &scaler, nil
}
