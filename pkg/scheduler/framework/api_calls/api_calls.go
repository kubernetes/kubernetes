/*
Copyright 2025 The Kubernetes Authors.

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

package apicalls

import (
	fwk "k8s.io/kube-scheduler/framework"
)

const (
	PodStatusPatch fwk.APICallType = "pod_status_patch"
	PodBinding     fwk.APICallType = "pod_binding"
	PodPreemption  fwk.APICallType = "pod_preemption"
)

// APICallRelevances is a built-in mapping types to relevances.
// Types of the same relevance should only be defined for different object types.
var APICallRelevances = fwk.APICallRelevances{
	PodStatusPatch: 1,
	PodBinding:     2,
	PodPreemption:  3,
}
