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

package workloadbuilder

import (
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
)

const (
	// GroupTemplateNameAnnotation maps a delegated child to the PodGroupTemplate
	// (or CompositePodGroupTemplate) in the parent Workload its pods belong to.
	GroupTemplateNameAnnotation = schedulingv1alpha3.GroupName + "/group-template-name"

	// ParentCompositePodGroupAnnotation names the parent CompositePodGroup
	// instance a delegated child's group must attach to. It disambiguates 
	// when one template is instantiated multiple times (e.g. LWS).
	ParentCompositePodGroupAnnotation = schedulingv1alpha3.GroupName + "/parent-composite-podgroup"
)
