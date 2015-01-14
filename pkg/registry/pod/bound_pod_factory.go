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

package pod

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/envvars"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type BoundPodFactory interface {
	// Make a container object for a given pod, given the machine that the pod is running on.
	MakeBoundPod(machine string, pod *api.Pod) (*api.BoundPod, error)
}

type BasicBoundPodFactory struct {
	// TODO: this should really point at the API rather than a registry
	ServiceRegistry        service.Registry
	MasterServiceNamespace string
}

var masterServiceNames = util.NewStringSet("kubernetes", "kubernetes-ro")

// getServiceEnvironmentVariables populates a list of environment variables that are used
// in the container environment to get access to services.
func (b *BasicBoundPodFactory) getServiceEnvironmentVariables(ctx api.Context, registry service.Registry, machine string) ([]api.EnvVar, error) {
	var result []api.EnvVar
	servicesInNs, err := registry.ListServices(ctx)
	if err != nil {
		return result, err
	}

	masterServices, err := registry.ListServices(api.WithNamespace(api.NewContext(), b.MasterServiceNamespace))
	if err != nil {
		return result, err
	}

	projection := map[string]api.Service{}
	services := []api.Service{}
	for _, service := range masterServices.Items {
		if masterServiceNames.Has(service.Name) {
			projection[service.Name] = service
		}
	}
	for _, service := range servicesInNs.Items {
		projection[service.Name] = service
	}
	for _, service := range projection {
		services = append(services, service)
	}

	return envvars.FromServices(&api.ServiceList{Items: services}), nil
}

func (b *BasicBoundPodFactory) MakeBoundPod(machine string, pod *api.Pod) (*api.BoundPod, error) {
	envVars, err := b.getServiceEnvironmentVariables(api.WithNamespace(api.NewContext(), pod.Namespace), b.ServiceRegistry, machine)
	if err != nil {
		return nil, err
	}

	boundPod := &api.BoundPod{}
	if err := api.Scheme.Convert(pod, boundPod); err != nil {
		return nil, err
	}
	for ix, container := range boundPod.Spec.Containers {
		boundPod.Spec.Containers[ix].Env = append(container.Env, envVars...)
	}
	// Make a dummy self link so that references to this bound pod will work.
	boundPod.SelfLink = "/api/v1beta1/boundPods/" + boundPod.Name
	return boundPod, nil
}
