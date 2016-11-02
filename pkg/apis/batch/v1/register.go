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

package v1

import (
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/announced"
	"k8s.io/kubernetes/pkg/runtime"
	versionedwatch "k8s.io/kubernetes/pkg/watch/versioned"
)

var (
	// VersionArgs is the canonical place for facts about this group version.
	VersionArgs = &announced.GroupVersionFactoryArgs{
		GroupName:   "batch",
		VersionName: "v1",
		VersionedObjects: []runtime.Object{
			&Job{},
			&JobList{},
			&v1.ListOptions{},
			&v1.DeleteOptions{},
		},
		SchemeAdjustment: runtime.NewSchemeBuilder(
			addDefaultingFuncs, addConversionFuncs,
		).VersionedRegister(versionedwatch.AddToGroupVersion),
	}
	SchemeBuilder = VersionArgs.SchemeAdjustment
)

var (
	AddToScheme        = SchemeBuilder.AddToScheme
	GroupName          = VersionArgs.GroupName
	SchemeGroupVersion = VersionArgs.SchemeGroupVersion()
)

func init() {
	if err := announced.AnnounceGroupVersion(VersionArgs); err != nil {
		panic(err)
	}
}
