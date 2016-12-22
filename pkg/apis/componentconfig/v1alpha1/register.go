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

package v1alpha1

import (
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

// GroupName is the group name use in this package
const GroupName = "componentconfig"

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1alpha1"}

var (
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes, addDefaultingFuncs)
	AddToScheme   = SchemeBuilder.AddToScheme
)

func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&KubeProxyConfiguration{},
		&KubeSchedulerConfiguration{},
		&KubeletConfiguration{},
	)
	return nil
}

func (obj *KubeProxyConfiguration) GetObjectKind() schema.ObjectKind     { return &obj.TypeMeta }
func (obj *KubeSchedulerConfiguration) GetObjectKind() schema.ObjectKind { return &obj.TypeMeta }
func (obj *KubeletConfiguration) GetObjectKind() schema.ObjectKind       { return &obj.TypeMeta }
