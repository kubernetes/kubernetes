/*
Copyright 2024 The Kubernetes Authors.

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

package testing

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

// Tweak is a function that modifies a Pod.
type Tweak func(*api.Pod)
type TweakContainer func(*api.Container)
type TweakPodStatus func(*api.PodStatus)

// MakePod helps construct Pod objects (which pass API validation) more
// legibly and tersely than a Go struct definition.  By default this produces
// a Pod with a single container, ctr.  The caller can pass any number of tweak
// functions to further modify the result.
func MakePod(name string, tweaks ...Tweak) *api.Pod {
	// NOTE: Any field that would be populated by defaulting needs to be
	// present and valid here.
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: api.PodSpec{
			Containers:                    []api.Container{MakeContainer("ctr")},
			DNSPolicy:                     api.DNSClusterFirst,
			RestartPolicy:                 api.RestartPolicyAlways,
			TerminationGracePeriodSeconds: ptr.To[int64](v1.DefaultTerminationGracePeriodSeconds),
		},
	}

	for _, tweak := range tweaks {
		tweak(pod)
	}

	return pod
}

func MakePodSpec(tweaks ...Tweak) api.PodSpec {
	return MakePod("", tweaks...).Spec
}

func SetNamespace(ns string) Tweak {
	return func(pod *api.Pod) {
		pod.Namespace = ns
	}
}

func SetResourceVersion(rv string) Tweak {
	return func(pod *api.Pod) {
		pod.ResourceVersion = rv
	}
}

func SetPodResources(resources *api.ResourceRequirements) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.Resources = resources
	}
}

func SetContainers(containers ...api.Container) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.Containers = containers
	}
}

func SetInitContainers(containers ...api.Container) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.InitContainers = containers
	}
}

func SetEphemeralContainers(containers ...api.EphemeralContainer) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.EphemeralContainers = containers
	}
}

func SetVolumes(volumes ...api.Volume) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.Volumes = volumes
	}
}

func MakeEmptyVolume(name string) api.Volume {
	return api.Volume{Name: name, VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}
}

func SetNodeSelector(nodeSelector map[string]string) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.NodeSelector = nodeSelector
	}
}

func SetNodeName(name string) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.NodeName = name
	}
}

func SetActiveDeadlineSeconds(deadline int64) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.ActiveDeadlineSeconds = &deadline
	}
}

func SetServiceAccountName(name string) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.ServiceAccountName = name
	}
}

func SetSecurityContext(ctx *api.PodSecurityContext) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.SecurityContext = ctx
	}
}

func SetAffinity(affinity *api.Affinity) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.Affinity = affinity
	}
}

func SetHostAliases(hostAliases ...api.HostAlias) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.HostAliases = hostAliases
	}
}

func SetPriorityClassName(name string) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.PriorityClassName = name
	}
}

func SetRuntimeClassName(name string) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.RuntimeClassName = &name
	}
}

func SetOverhead(overhead api.ResourceList) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.Overhead = overhead
	}
}

func SetDNSPolicy(policy api.DNSPolicy) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.DNSPolicy = policy
	}
}

func SetDNSConfig(config *api.PodDNSConfig) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.DNSConfig = config
	}
}

func SetRestartPolicy(policy api.RestartPolicy) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.RestartPolicy = policy
	}
}

func SetTolerations(tolerations ...api.Toleration) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.Tolerations = tolerations
	}
}

func SetAnnotations(annos map[string]string) Tweak {
	return func(pod *api.Pod) {
		pod.Annotations = annos
	}
}

func SetLabels(annos map[string]string) Tweak {
	return func(pod *api.Pod) {
		pod.Labels = annos
	}
}

func SetGeneration(gen int64) Tweak {
	return func(pod *api.Pod) {
		pod.Generation = gen
	}
}

func SetSchedulingGates(gates ...api.PodSchedulingGate) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.SchedulingGates = gates
	}
}

func SetTerminationGracePeriodSeconds(grace int64) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.TerminationGracePeriodSeconds = &grace
	}
}

func SetOS(name api.OSName) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.OS = &api.PodOS{Name: name}
	}
}

func SetStatus(status api.PodStatus) Tweak {
	return func(pod *api.Pod) {
		pod.Status = status
	}
}

func SetResourceClaims(claims ...api.PodResourceClaim) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.ResourceClaims = claims
	}
}

func SetTopologySpreadConstraints(tsc ...api.TopologySpreadConstraint) Tweak {
	return func(pod *api.Pod) {
		pod.Spec.TopologySpreadConstraints = tsc
	}
}

func SetObjectMeta(objectMeta metav1.ObjectMeta) Tweak {
	return func(pod *api.Pod) {
		pod.ObjectMeta = objectMeta
	}
}

func MakeContainer(name string, tweaks ...TweakContainer) api.Container {
	cnr := api.Container{
		Name: name, Image: "image", ImagePullPolicy: "IfNotPresent",
		TerminationMessagePolicy: "File",
		TerminationMessagePath:   v1.TerminationMessagePathDefault,
	}

	for _, tweak := range tweaks {
		tweak(&cnr)
	}

	return cnr
}

func SetContainerImage(image string) TweakContainer {
	return func(cnr *api.Container) {
		cnr.Image = image
	}
}

func SetContainerLifecycle(lifecycle api.Lifecycle) TweakContainer {
	return func(cnr *api.Container) {
		cnr.Lifecycle = &lifecycle
	}
}

func MakeResourceRequirements(requests, limits map[string]string) api.ResourceRequirements {
	rr := api.ResourceRequirements{Requests: api.ResourceList{}, Limits: api.ResourceList{}}
	for k, v := range requests {
		rr.Requests[api.ResourceName(k)] = resource.MustParse(v)
	}
	for k, v := range limits {
		rr.Limits[api.ResourceName(k)] = resource.MustParse(v)
	}
	return rr
}

func SetContainerResources(rr api.ResourceRequirements) TweakContainer {
	return func(cnr *api.Container) {
		cnr.Resources = rr
	}
}

func SetContainerPorts(ports ...api.ContainerPort) TweakContainer {
	return func(cnr *api.Container) {
		cnr.Ports = ports
	}
}

func SetContainerResizePolicy(policies ...api.ContainerResizePolicy) TweakContainer {
	return func(cnr *api.Container) {
		cnr.ResizePolicy = policies
	}
}

func SetContainerSecurityContext(ctx api.SecurityContext) TweakContainer {
	return func(cnr *api.Container) {
		cnr.SecurityContext = &ctx
	}
}

func SetContainerRestartPolicy(policy api.ContainerRestartPolicy) TweakContainer {
	return func(cnr *api.Container) {
		cnr.RestartPolicy = &policy
	}
}

func MakePodStatus(tweaks ...TweakPodStatus) api.PodStatus {
	ps := api.PodStatus{}

	for _, tweak := range tweaks {
		tweak(&ps)
	}

	return ps
}

func SetContainerStatuses(containerStatuses ...api.ContainerStatus) TweakPodStatus {
	return func(podstatus *api.PodStatus) {
		podstatus.ContainerStatuses = containerStatuses
	}
}

func SetInitContainerStatuses(containerStatuses ...api.ContainerStatus) TweakPodStatus {
	return func(podstatus *api.PodStatus) {
		podstatus.InitContainerStatuses = containerStatuses
	}
}

func SetEphemeralContainerStatuses(containerStatuses ...api.ContainerStatus) TweakPodStatus {
	return func(podstatus *api.PodStatus) {
		podstatus.EphemeralContainerStatuses = containerStatuses
	}
}

func MakeContainerStatus(name string, allocatedResources api.ResourceList) api.ContainerStatus {
	cs := api.ContainerStatus{
		Name:               name,
		AllocatedResources: allocatedResources,
	}

	return cs
}

// TweakContainers applies the container tweaks to all containers (regular & init) in the pod.
// Note: this should typically be added to pod tweaks after all containers have been added.
func TweakContainers(tweaks ...TweakContainer) Tweak {
	return func(pod *api.Pod) {
		for i := range pod.Spec.InitContainers {
			for _, tweak := range tweaks {
				tweak(&pod.Spec.InitContainers[i])
			}
		}
		for i := range pod.Spec.Containers {
			for _, tweak := range tweaks {
				tweak(&pod.Spec.Containers[i])
			}
		}
	}
}
