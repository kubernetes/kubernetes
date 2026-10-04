/*
Copyright The Kubernetes Authors.

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

package scheduling

const (
	// GroupTemplateNameAnnotation maps a child resource to its corresponding
	// PodGroupTemplate or CompositePodGroupTemplate defined in the parent Workload.
	// For example, in a TrainJob -> JobSet -> Job hierarchy, this annotation tells
	// the child Job which template from the root Workload to use to create its PodGroup.
	GroupTemplateNameAnnotation = GroupName + "/group-template-name"

	// ParentCompositePodGroupAnnotation specifies the exact name of the runtime
	// CompositePodGroup instance the child's PodGroup must attach to.
	//
	// This is required only in a delegated lifecycle model, where the parent
	// (e.g., JobSet) creates the Workload and CompositePodGroup, but the child
	// (e.g., Job) creates its own PodGroup. If the parent centrally manages
	// both the CompositePodGroup and PodGroups, this annotation is unnecessary.
	ParentCompositePodGroupAnnotation = GroupName + "/parent-compositepodgroup"
)
