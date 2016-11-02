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

package batch

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apimachinery/announced"
	"k8s.io/kubernetes/pkg/runtime"
)

var (
	// GroupArgs is the canonical place for facts about this group.
	GroupArgs = &announced.GroupMetaFactoryArgs{
		GroupName:                "batch",
		VersionPreferenceOrder:   []string{"v1", "v2alpha1"},
		ImportPrefix:             "k8s.io/kubernetes/pkg/apis/batch",
		InternalSchemeAdjustment: runtime.NewSchemeBuilder(),
		InternalObjects: []runtime.Object{
			&Job{},
			&JobList{},
			&JobTemplate{},
			&ScheduledJob{},
			&ScheduledJobList{},
			&api.ListOptions{},
		},
	}

	// SchemeBuilder allows other files to register things that need to be
	// done to a scheme for this group.
	SchemeBuilder = GroupArgs.InternalSchemeAdjustment
)

// Copy-paste block to enable refactoring. Really people calling these
// functions should get their info from the api group registration manager.
// TODO: delete these.
var (
	AddToScheme        = SchemeBuilder.AddToScheme
	GroupName          = GroupArgs.GroupName
	SchemeGroupVersion = GroupArgs.SchemeGroupVersion()
	Kind               = GroupArgs.Kind
	Resource           = GroupArgs.Resource
)

func init() {
	if err := announced.AnnounceGroup(GroupArgs); err != nil {
		panic(err)
	}
}
