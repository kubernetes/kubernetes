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

package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// NodeLifecycleControllerConfiguration contains elements describing CloudNodeLifecycleController.
type NodeLifecycleControllerConfiguration struct {
	// NodeMonitorPeriod is the period for syncing NodeStatus in CloudNodeLifecycleController.
	NodeMonitorPeriod metav1.Duration
	// ConcurrentNodeLifecycleSyncs is the number of workers for syncing NodeStatus in CloudNodeLifecycleController.
	ConcurrentNodeLifecycleSyncs int32
}
