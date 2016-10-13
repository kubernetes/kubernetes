/*
Copyright 2016 The Kubernetes Authors.

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

package api

// PodSpecer is implemented by API objects that embed a PodSpec.  It allows
// generic inspection, admission, and mutation of things that contain a PodSpec.
// For instance, PSP admission for Pods, RCs, RSs, Deployments, etc.
type PodSpecer interface {
	// PodSpec returns a reference to the PodSpec contained in the object and the
	// fieldPath to the PodSpec for any future messages.
	PodSpec() (spec *PodSpec, fieldPath string)
}

func (obj *Pod) PodSpec() (*PodSpec, string) {
	return &obj.Spec, "spec"
}

func (obj *ReplicationController) PodSpec() (*PodSpec, string) {
	return &obj.Spec.Template.Spec, "spec.template.spec"
}
