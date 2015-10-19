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

package examples

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
)

type staticExampleBuilder struct {
	creater   runtime.ObjectCreater
	convertor runtime.ObjectConvertor
	examples  map[string]runtime.Object
}

var _ ExampleBuilder = &staticExampleBuilder{}

func (f *staticExampleBuilder) NewExample(version, kind string) (runtime.Object, bool, error) {
	if example, ok := f.examples[kind]; ok {
		versionedExample, err := f.convertor.ConvertToVersion(example, version)
		if err != nil {
			return nil, true, err
		}

		return versionedExample, true, nil
	}

	example, err := f.creater.New(version, kind)
	return example, false, err
}

func NewStaticExampleBuilder(creater runtime.ObjectCreater, convertor runtime.ObjectConvertor) ExampleBuilder {
	return &staticExampleBuilder{
		creater:   creater,
		convertor: convertor,
		examples: map[string]runtime.Object{
			"Pod": &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "webserver",
							Image: "nginx",
							Ports: []api.ContainerPort{
								{Name: "http", ContainerPort: 80, Protocol: "TCP"},
							},
							VolumeMounts: []api.VolumeMount{
								{Name: "html", ReadOnly: true, MountPath: "/usr/share/nginx/html"},
							},
						},
					},
					Volumes: []api.Volume{
						{Name: "html", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
					},
				},
			},
		},
	}
}
