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

package framework

import (
	"context"

	v1 "k8s.io/api/core/v1"
)

// APICacher defines methods that send API calls through the scheduler's cache
// before they are executed asynchronously by the APIDispatcher.
// This ensures the scheduler's internal state is updated optimistically,
// reflecting the intended outcome of the call.
// This methods should be used only if the SchedulerAsyncAPICalls feature gate is enabled.
type APICacher interface {
	// PatchPodStatus sends a patch request for a Pod's status.
	// The patch could be first applied to the cached Pod object and then the API call is executed asynchronously.
	// It returns a channel that can be used to wait for the call's completion.
	PatchPodStatus(pod *v1.Pod, condition *v1.PodCondition, nominatingInfo *NominatingInfo) (<-chan error, error)

	// BindPod sends a binding request. The binding could be first applied to the cached Pod object
	// and then the API call is executed asynchronously.
	// It returns a channel that can be used to wait for the call's completion.
	BindPod(binding *v1.Binding) (<-chan error, error)

	// WaitOnFinish blocks until the result of an API call is sent to the given onFinish channel
	// (returned by methods BindPod or PreemptPod).
	//
	// It returns the error received from the channel.
	// It also returns nil if the call was skipped or overwritten,
	// as these are considered successful lifecycle outcomes.
	// Direct onFinish channel read can be used to access these results.
	WaitOnFinish(ctx context.Context, onFinish <-chan error) error
}

// APICallImplementations define constructors for each APICall that is used by the scheduler internally.
type APICallImplementations[T, K APICall] struct {
	// PodStatusPatch is a constructor used to create APICall object for pod status patch.
	PodStatusPatch func(pod *v1.Pod, condition *v1.PodCondition, nominatingInfo *NominatingInfo) T
	// PodBinding is a constructor used to create APICall object for pod binding.
	PodBinding func(binding *v1.Binding) K
}
