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
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/golang/glog"
)

type BasicReplicationController struct{}

func (BasicReplicationController) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"labels", false},
		{"name", true},
		{"replicas", true},
		{"image", true},
	}
}

func (BasicReplicationController) Generate(params map[string]string) (runtime.Object, error) {
	// TODO: extract this flag to a central location.
	labelString, found := params["labels"]
	var labels map[string]string
	if found && len(labelString) > 0 {
		labels = ParseLabels(labelString)
	} else {
		labels = map[string]string{
			"run-container": params["name"],
		}
	}
	count, err := strconv.Atoi(params["replicas"])
	if err != nil {
		return nil, err
	}
	controller := api.ReplicationController{
		NSObjectMeta: api.NSObjectMeta{
			ObjectMeta: api.ObjectMeta{
				Name:   params["name"],
				Labels: labels,
			},
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: count,
			Selector: labels,
			Template: &api.PodTemplateSpec{
				NSObjectMeta: api.NSObjectMeta{
					ObjectMeta: api.ObjectMeta{
						Labels: labels,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  params["name"],
							Image: params["image"],
						},
					},
				},
			},
		},
	}
	return &controller, nil
}

// TODO: extract this to a common location.
func ParseLabels(labelString string) map[string]string {
	if len(labelString) == 0 {
		return nil
	}
	labels := map[string]string{}
	labelSpecs := strings.Split(labelString, ",")
	for ix := range labelSpecs {
		labelSpec := strings.Split(labelSpecs[ix], "=")
		if len(labelSpec) != 2 {
			glog.Errorf("unexpected label spec: %s", labelSpecs[ix])
			continue
		}
		labels[labelSpec[0]] = labelSpec[1]
	}
	return labels
}
