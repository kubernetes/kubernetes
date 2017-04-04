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
	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

type Pod struct {
	apiPod *v1.Pod
}

func NewPod(newPod *v1.Pod) *Pod {
	return &Pod{apiPod: newPod}
}

func (p *Pod) GetStatus() *v1.PodStatus {
	return &p.apiPod.Status
}

func (p *Pod) SetStatus(newStatus *v1.PodStatus) {
	p.apiPod.Status = *newStatus
}

func (p *Pod) GetSpec() *v1.PodSpec {
	return &p.apiPod.Spec
}

func (p *Pod) SetSpec(newSpec *v1.PodSpec) {
	p.apiPod.Spec = *newSpec
}

func (p *Pod) GetAPIPod() *v1.Pod {
	return p.apiPod
}

func (p *Pod) GetPodFullName() string {
	return kubecontainer.GetPodFullName(p.GetAPIPod())
}

func (p *Pod) String() string {
	return format.Pod(p.GetAPIPod())
}

func (p *Pod) ObjectMeta() metav1.ObjectMeta {
	return p.apiPod.ObjectMeta
}

func (p *Pod) DeletionTimestampIsSet() bool {
	return p.ObjectMeta().DeletionTimestamp != nil
}

func (p *Pod) UID() types.UID {
	return p.ObjectMeta().UID
}

func (p *Pod) Namespace() string {
	return p.ObjectMeta().Namespace
}

func (p *Pod) Name() string {
	return p.ObjectMeta().Name
}

func ToAPIPods(pods []*Pod) []*v1.Pod {
	apiPods := []*v1.Pod{}
	for _, pod := range pods {
		apiPods = append(apiPods, pod.GetAPIPod())
	}
	return apiPods
}

func FromAPIPods(apiPods []*v1.Pod) []*Pod {
	pods := []*Pod{}
	for _, apiPod := range apiPods {
		pods = append(pods, NewPod(apiPod))
	}
	return pods
}

func Format(pods []*Pod) string {
	return format.Pods(ToAPIPods(pods))
}
