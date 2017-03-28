/*
Copyright 2015 The Kubernetes Authors.

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

import "k8s.io/kubernetes/pkg/api"

const (
	// TODO: to be de!eted after v1.3 is released. PodSpec has a dedicated Hostname field.
	// The annotation value is a string specifying the hostname to be used for the pod e.g 'my-webserver-1'
	PodHostnameAnnotation = "pod.beta.kubernetes.io/hostname"

	// TODO: to be de!eted after v1.3 is released. PodSpec has a dedicated Subdomain field.
	// The annotation value is a string specifying the subdomain e.g. "my-web-service"
	// If specified, on the pod itself, "<hostname>.my-web-service.<namespace>.svc.<cluster domain>" would resolve to
	// the pod's IP.
	// If there is a headless service named "my-web-service" in the same namespace as the pod, then,
	// <hostname>.my-web-service.<namespace>.svc.<cluster domain>" would be resolved by the cluster DNS Server.
	PodSubdomainAnnotation = "pod.beta.kubernetes.io/subdomain"
)

// VisitPodSecretNames invokes the visitor function with the name of every secret
// referenced by the pod spec. If visitor returns false, visiting is short-circuited.
// Transitive references (e.g. pod -> pvc -> pv -> secret) are not visited.
// Returns true if visiting completed, false if visiting was short-circuited.
func VisitPodSecretNames(pod *api.Pod, visitor func(string) bool) bool {
	for _, reference := range pod.Spec.ImagePullSecrets {
		if !visitor(reference.Name) {
			return false
		}
	}
	for i := range pod.Spec.InitContainers {
		if !visitContainerSecretNames(&pod.Spec.InitContainers[i], visitor) {
			return false
		}
	}
	for i := range pod.Spec.Containers {
		if !visitContainerSecretNames(&pod.Spec.Containers[i], visitor) {
			return false
		}
	}
	var source *api.VolumeSource
	for i := range pod.Spec.Volumes {
		source = &pod.Spec.Volumes[i].VolumeSource
		switch {
		case source.AzureFile != nil:
			if len(source.AzureFile.SecretName) > 0 && !visitor(source.AzureFile.SecretName) {
				return false
			}
		case source.CephFS != nil:
			if source.CephFS.SecretRef != nil && !visitor(source.CephFS.SecretRef.Name) {
				return false
			}
		case source.FlexVolume != nil:
			if source.FlexVolume.SecretRef != nil && !visitor(source.FlexVolume.SecretRef.Name) {
				return false
			}
		case source.Projected != nil:
			for j := range source.Projected.Sources {
				if source.Projected.Sources[j].Secret != nil {
					if !visitor(source.Projected.Sources[j].Secret.Name) {
						return false
					}
				}
			}
		case source.RBD != nil:
			if source.RBD.SecretRef != nil && !visitor(source.RBD.SecretRef.Name) {
				return false
			}
		case source.Secret != nil:
			if !visitor(source.Secret.SecretName) {
				return false
			}
		case source.ScaleIO != nil:
			if source.ScaleIO.SecretRef != nil && !visitor(source.ScaleIO.SecretRef.Name) {
				return false
			}
		}
	}
	return true
}

func visitContainerSecretNames(container *api.Container, visitor func(string) bool) bool {
	for _, env := range container.EnvFrom {
		if env.SecretRef != nil {
			if !visitor(env.SecretRef.Name) {
				return false
			}
		}
	}
	for _, envVar := range container.Env {
		if envVar.ValueFrom != nil && envVar.ValueFrom.SecretKeyRef != nil {
			if !visitor(envVar.ValueFrom.SecretKeyRef.Name) {
				return false
			}
		}
	}
	return true
}
