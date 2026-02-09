/*
Copyright 2020 The Kubernetes Authors.

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

package netpol

import (
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// Model defines the namespaces, deployments, services, pods, containers and associated
// data for network policy test cases and provides the source of truth
type Model struct {
	Namespaces []*Namespace
	PodNames   []string
	Ports      []int32
	Protocols  []v1.Protocol
}

// NewModel instantiates a model based on:
// - namespaceNames
// - pods
// - ports to listen on
// - protocols to listen on
// The total number of pods is the number of namespaces x the number of pods per namespace.
// The number of containers per pod is the number of ports x the number of protocols.
// The *total* number of containers is namespaces x pods x ports x protocols.
func NewModel(namespaceNames []string, podNames []string, ports []int32, protocols []v1.Protocol) *Model {
	model := &Model{
		PodNames:  podNames,
		Ports:     ports,
		Protocols: protocols,
	}

	// build the entire "model" for the overall test, which means, building
	// namespaces, pods, containers for each protocol.
	for _, ns := range namespaceNames {
		var pods []*Pod
		for _, podName := range podNames {
			pods = append(pods, &Pod{
				Name:      podName,
				Ports:     ports,
				Protocols: protocols,
			})
		}
		model.Namespaces = append(model.Namespaces, &Namespace{
			Name: ns,
			Pods: pods,
		})
	}
	return model
}

// Namespace is the abstract representation of what matters to network policy
// tests for a namespace; i.e. it ignores kube implementation details
type Namespace struct {
	Name string
	Pods []*Pod
}

// Pod is the abstract representation of what matters to network policy tests for
// a pod; i.e. it ignores kube implementation details
type Pod struct {
	Name      string
	Ports     []int32
	Protocols []v1.Protocol
}

func podNameLabelKey() string {
	return "pod"
}

// Labels returns the default labels that should be placed on a pod/deployment
// in order for it to be uniquely selectable by label selectors
func (p *Pod) Labels() map[string]string {
	return map[string]string{
		podNameLabelKey(): p.Name,
	}
}

// KubePod returns the kube pod (will add label selectors for windows if needed).
func (p *Pod) KubePod(namespace string) *v1.Pod {
	zero := int64(0)

	thePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      p.Name,
			Labels:    p.Labels(),
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			TerminationGracePeriodSeconds: &zero,
			Containers:                    p.ContainerSpecs(),
		},
	}

	if framework.NodeOSDistroIs("windows") {
		thePod.Spec.NodeSelector = map[string]string{
			"kubernetes.io/os": "windows",
		}
	}
	return thePod
}

// QualifiedServiceAddress returns the address that can be used to access the service
func (p *Pod) QualifiedServiceAddress(namespace string, dnsDomain string) string {
	return fmt.Sprintf("%s.%s.svc.%s", p.ServiceName(namespace), namespace, dnsDomain)
}

// ServiceName returns the unqualified service name
func (p *Pod) ServiceName(namespace string) string {
	return fmt.Sprintf("s-%s-%s", namespace, p.Name)
}

// Service returns a kube service spec
func (p *Pod) Service(namespace string) *v1.Service {
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      p.ServiceName(namespace),
			Namespace: namespace,
		},
		Spec: v1.ServiceSpec{
			Selector: p.Labels(),
		},
	}
	for _, protocol := range p.Protocols {
		for _, port := range p.Ports {
			service.Spec.Ports = append(service.Spec.Ports, v1.ServicePort{
				Name:     fmt.Sprintf("service-port-%s-%d", strings.ToLower(string(protocol)), port),
				Protocol: protocol,
				Port:     port,
			})
		}
	}
	return service
}

// ContainerSpecs builds kubernetes container specs for the pod
func (p *Pod) ContainerSpecs() []v1.Container {
	env := make([]v1.EnvVar, 0, len(p.Ports)*len(p.Protocols))
	ports := make([]v1.ContainerPort, 0, len(p.Ports)*len(p.Protocols))

	for _, protocol := range p.Protocols {
		for _, port := range p.Ports {
			env = append(env, v1.EnvVar{
				Name:  fmt.Sprintf("SERVE_%s_PORT_%d", protocol, port),
				Value: "foo",
			})
			ports = append(ports, v1.ContainerPort{
				Name:          fmt.Sprintf("serve-%d-%s", port, strings.ToLower(string(protocol))),
				Protocol:      protocol,
				ContainerPort: port,
			})
		}
	}

	return []v1.Container{{
		Name:            "agnhost",
		ImagePullPolicy: v1.PullIfNotPresent,
		Image:           imageutils.GetE2EImage(imageutils.Agnhost),
		Command:         []string{"/agnhost", "porter"},
		SecurityContext: &v1.SecurityContext{},
		Env:             env,
		Ports:           ports,
	}}
}
