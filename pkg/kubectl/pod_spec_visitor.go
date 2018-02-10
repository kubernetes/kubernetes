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

package kubectl

import (
	"encoding/json"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kapps "k8s.io/kubernetes/pkg/kubectl/apps"
)

type PodSpecVisitor struct {
	Object runtime.Unstructured
	// MungeFn eventually should not depend on corev1.
	MungeFn func(spec *corev1.PodSpec) error
	err     error
}

var _ kapps.KindVisitor = &PodSpecVisitor{}

func (v *PodSpecVisitor) VisitDeployment(elem kapps.GroupKindElement) {
	v.err = v.mungePodSpec([]string{"spec", "template", "spec"})
}

func (v *PodSpecVisitor) VisitStatefulSet(kind kapps.GroupKindElement) {
	v.err = v.mungePodSpec([]string{"spec", "template", "spec"})
}

func (v *PodSpecVisitor) VisitDaemonSet(kind kapps.GroupKindElement) {
	v.err = v.mungePodSpec([]string{"spec", "template", "spec"})
}

func (v *PodSpecVisitor) VisitJob(kind kapps.GroupKindElement) {
	v.err = v.mungePodSpec([]string{"spec", "template", "spec"})
}

func (v *PodSpecVisitor) VisitReplicaSet(kind kapps.GroupKindElement) {
	v.err = v.mungePodSpec([]string{"spec", "template", "spec"})
}

func (v *PodSpecVisitor) VisitPod(kind kapps.GroupKindElement) {
	v.err = v.mungePodSpec([]string{"spec"})
}

func (v *PodSpecVisitor) VisitReplicationController(kind kapps.GroupKindElement) {
	v.err = v.mungePodSpec([]string{"spec", "template", "spec"})
}

func (v *PodSpecVisitor) VisitCronJob(kind kapps.GroupKindElement) {
	v.err = v.mungePodSpec([]string{"spec", "jobTemplate", "spec", "template", "spec"})
}

func walkMapPath(start map[string]interface{}, path []string) (map[string]interface{}, error) {
	finish := start
	for i := 0; i < len(path); i++ {
		var ok bool
		finish, ok = finish[path[i]].(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("key:%s of path:%v not found in map:%v", path[i], path, start)
		}
	}

	return finish, nil
}

func (v *PodSpecVisitor) mungePodSpec(pathToPodSpec []string) error {
	obj := v.Object.UnstructuredContent()
	podSpec, err := walkMapPath(obj, pathToPodSpec)
	if err != nil {
		return err
	}
	jsonPodSpec, err := json.Marshal(podSpec)
	if err != nil {
		return err
	}
	var ps corev1.PodSpec
	json.Unmarshal(jsonPodSpec, &ps)
	return v.MungeFn(&ps)
}
