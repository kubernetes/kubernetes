/*
Copyright 2021 The Kubernetes Authors.

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

package queue

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// NewTestQueueWithObjects creates a priority queue with an informer factory
// populated with the provided objects.
func NewTestQueueWithObjects(
	ctx context.Context,
	lessFn framework.LessFunc,
	objs []runtime.Object,
	opts ...Option,
) *PriorityQueue {
	informerFactory := informers.NewSharedInformerFactory(fake.NewSimpleClientset(objs...), 0)
	pq := NewPriorityQueue(lessFn, informerFactory, opts...)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	return pq
}

// NewTestQueue creates a priority queue with an empty informer factory.
func NewTestQueue(ctx context.Context, lessFn framework.LessFunc, opts ...Option) *PriorityQueue {
	return NewTestQueueWithObjects(ctx, lessFn, nil, opts...)
}
