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

package job

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/features"
)

// DropDisabledFields removes feature-gated fields from the JobSpec when their
// feature gate is disabled and they are not already in use in the old spec.
// oldJobSpec is nil on create. It is shared by the Job and CronJob registry
// strategies, which both embed a JobSpec.
func DropDisabledFields(jobSpec, oldJobSpec *batch.JobSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadWithJob) || schedulingInUse(oldJobSpec) {
		return
	}
	jobSpec.Scheduling = nil
}

func schedulingInUse(jobSpec *batch.JobSpec) bool {
	return jobSpec != nil && jobSpec.Scheduling != nil
}
