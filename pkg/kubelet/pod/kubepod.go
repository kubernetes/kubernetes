/*
Copyright 2017 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

type Pod struct {
	apiPod *v1.Pod
}

func NewPod(newPod *v1.Pod) *Pod {
	return &Pod{apiPod: newPod}
}

// GetStatus returns the most recent status passed to NewPod or SetStatus
func (p *Pod) GetStatus() *v1.PodStatus {
	return &p.apiPod.Status
}

// SetStatus sets the status of the pod.
func (p *Pod) SetStatus(status *v1.PodStatus) {
	p.apiPod.Status = *status
}

// GetSpec returns the most recent spec passed to NewPod or SetSpec
func (p *Pod) GetSpec() *v1.PodSpec {
	return &p.apiPod.Spec
}

// SetSpec sets the spec of the pod
func (p *Pod) SetSpec(spec *v1.PodSpec) {
	p.apiPod.Spec = *spec
}

// GetAPIPod returns a v1.Pod representation of the pod
func (p *Pod) GetAPIPod() *v1.Pod {
	return p.apiPod
}

// GetFullName returns the full pod name, which uniquely identifies a pod
func (p *Pod) GetFullName() string {
	return kubecontainer.GetPodFullName(p.GetAPIPod())
}

// String returns a string representation of the pod.
func (p *Pod) String() string {
	return format.Pod(p.GetAPIPod())
}

// objectMeta returns the ObjectMeta of the pod.
func (p *Pod) objectMeta() metav1.ObjectMeta {
	return p.apiPod.ObjectMeta
}

// DeletionTimestampIsSet returns true if the deletion timestamp of the pod has been set by the source.
func (p *Pod) DeletionTimestampIsSet() bool {
	return p.objectMeta().DeletionTimestamp != nil
}

// Annotations returns the annotations of the pod
func (p *Pod) Annotations() map[string]string {
	return p.objectMeta().Annotations
}

// UID returns the pod's UID
func (p *Pod) UID() types.UID {
	return p.objectMeta().UID
}

// Namespace returns the pod's namespace
func (p *Pod) Namespace() string {
	return p.objectMeta().Namespace
}

// Name returns the pod's name
func (p *Pod) Name() string {
	return p.objectMeta().Name
}

// IsMirror returns true if the pod is a mirror pod
func (p *Pod) IsMirror() bool {
	return IsMirrorPod(p.GetAPIPod())
}

// IsStatic returns true if the pod is a static pod
func (p *Pod) IsStatic() bool {
	source, err := p.GetSource()
	return err == nil && source != kubelettypes.ApiserverSource
}

// IsCritical returns true if the pod is a critical pod and if the critical pod feature is enabled
func (p *Pod) IsCritical() bool {
	return kubelettypes.IsCriticalPod(p.apiPod) && utilfeature.DefaultFeatureGate.Enabled(features.ExperimentalCriticalPodAnnotation)
}

// GetPodSource returns the source of the pod.  E.g. "api"
func (p *Pod) GetSource() (string, error) {
	return kubelettypes.GetPodSource(p.GetAPIPod())
}

// GetQOS returns the PodQOSClass of this pod
func (p *Pod) GetQOS() v1.PodQOSClass {
	return qos.GetPodQOS(p.GetSpec())
}

// ToAPIPods returns a list of v1.Pod representations of each pod in the input list.
func ToAPIPods(pods []*Pod) []*v1.Pod {
	apiPods := []*v1.Pod{}
	for _, pod := range pods {
		apiPods = append(apiPods, pod.GetAPIPod())
	}
	return apiPods
}

// FromAPIPods creates a new pod from each v1.Pod in the input list, and returns these new pods as a list.
func FromAPIPods(apiPods []*v1.Pod) []*Pod {
	pods := []*Pod{}
	for _, apiPod := range apiPods {
		pods = append(pods, NewPod(apiPod))
	}
	return pods
}

// Format returns a string representation of a list of pods
func Format(pods []*Pod) string {
	return format.Pods(ToAPIPods(pods))
}
