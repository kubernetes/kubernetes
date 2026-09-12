/*
Copyright 2026 The Kubernetes Authors.

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

package v1beta1

import (
	"k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

// Convert_scheduling_PriorityClass_To_v1beta1_PriorityClass converts the internal PriorityClass to
// the removed scheduling.k8s.io/v1beta1 version. AllowDisruptionByPriorityGreaterThanOrEqual has no
// equivalent field in v1beta1 (which is no longer served, see
// k8s.io/api/scheduling/v1beta1.PriorityClass) and is intentionally dropped.
func Convert_scheduling_PriorityClass_To_v1beta1_PriorityClass(in *scheduling.PriorityClass, out *v1beta1.PriorityClass, s conversion.Scope) error {
	return autoConvert_scheduling_PriorityClass_To_v1beta1_PriorityClass(in, out, s)
}
