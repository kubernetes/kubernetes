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

package v2alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/v1"
)

// GroupName is the group name use in this package
const GroupName = "batch"

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v2alpha1"}

// Resource takes an unqualified resource and returns a Group qualified GroupResource
func Resource(resource string) schema.GroupResource {
	return SchemeGroupVersion.WithResource(resource).GroupResource()
}

var (
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes, addDefaultingFuncs, addConversionFuncs)
	AddToScheme   = SchemeBuilder.AddToScheme
)

// Adds the list of known types to api.Scheme.
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Job{},
		&JobList{},
		&JobTemplate{},
		&CronJob{},
		&CronJobList{},
		&v1.ListOptions{},
		&v1.DeleteOptions{},
		&metav1.ExportOptions{},
		&metav1.GetOptions{},
	)
	scheme.AddKnownTypeWithName(SchemeGroupVersion.WithKind("ScheduledJob"), &CronJob{})
	scheme.AddKnownTypeWithName(SchemeGroupVersion.WithKind("ScheduledJobList"), &CronJobList{})
	metav1.AddToGroupVersion(scheme, SchemeGroupVersion)
	return nil
}
