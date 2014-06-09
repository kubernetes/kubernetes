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
package registry

import (
	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type ManifestFactory interface {
	// Make a container object for a given pod, given the machine that the pod is running on.
	MakeManifest(machine string, pod Pod) (ContainerManifest, error)
}

type BasicManifestFactory struct {
	serviceRegistry ServiceRegistry
}

func (b *BasicManifestFactory) MakeManifest(machine string, pod Pod) (ContainerManifest, error) {
	envVars, err := GetServiceEnvironmentVariables(b.serviceRegistry, machine)
	if err != nil {
		return ContainerManifest{}, err
	}
	for ix, container := range pod.DesiredState.Manifest.Containers {
		pod.DesiredState.Manifest.Id = pod.ID
		pod.DesiredState.Manifest.Containers[ix].Env = append(container.Env, envVars...)
	}
	return pod.DesiredState.Manifest, nil
}
