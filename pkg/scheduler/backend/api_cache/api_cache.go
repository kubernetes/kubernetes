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

package apicache

import (
	"context"

	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
)

// APICache is responsible for sending API calls' requests through scheduling queue or cache.
type APICache struct {
	schedulingQueue internalqueue.SchedulingQueue
	cache           internalcache.Cache
}

func New(schedulingQueue internalqueue.SchedulingQueue, cache internalcache.Cache) *APICache {
	return &APICache{
		schedulingQueue: schedulingQueue,
		cache:           cache,
	}
}

// PatchPodStatus sends a patch request for a Pod's status through a scheduling queue.
// The patch could be first applied to the cached Pod object and then the API call is executed asynchronously.
// It returns a channel that can be used to wait for the call's completion.
func (c *APICache) PatchPodStatus(pod *v1.Pod, condition *v1.PodCondition, nominatingInfo *fwk.NominatingInfo) (<-chan error, error) {
	return c.schedulingQueue.PatchPodStatus(pod, condition, nominatingInfo)
}

// BindPod sends a binding request through a cache. The binding could be first applied to the cached Pod object
// and then the API call is executed asynchronously.
// It returns a channel that can be used to wait for the call's completion.
func (c *APICache) BindPod(binding *v1.Binding) (<-chan error, error) {
	return c.cache.BindPod(binding)
}

// WaitOnFinish blocks until the result of an API call is sent to the given onFinish channel
// (returned by methods BindPod or PreemptPod).
//
// It returns the error received from the channel.
// It also returns nil if the call was skipped or overwritten,
// as these are considered successful lifecycle outcomes.
func (c *APICache) WaitOnFinish(ctx context.Context, onFinish <-chan error) error {
	select {
	case err := <-onFinish:
		if fwk.IsUnexpectedError(err) {
			return err
		}
	case <-ctx.Done():
		return ctx.Err()
	}
	return nil
}
