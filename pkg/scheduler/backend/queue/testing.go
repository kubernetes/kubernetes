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
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// NewTestQueue creates a priority queue with an empty informer factory.
func NewTestQueue(ctx context.Context, lessFn framework.LessFunc, opts ...Option) *PriorityQueue {
	return NewTestQueueWithObjects(ctx, lessFn, nil, opts...)
}

// NewTestQueueWithObjects creates a priority queue with an informer factory
// populated with the provided objects.
func NewTestQueueWithObjects(
	ctx context.Context,
	lessFn framework.LessFunc,
	objs []runtime.Object,
	opts ...Option,
) *PriorityQueue {
	informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(objs...), 0)

	// Because some major functions (e.g., Pop) requires the metric recorder to be set,
	// we always set a metric recorder here.
	recorder := metrics.NewMetricsAsyncRecorder(10, 20*time.Microsecond, ctx.Done())
	// We set it before the options that users provide, so that users can override it.
	opts = append([]Option{WithMetricsRecorder(*recorder)}, opts...)
	return NewTestQueueWithInformerFactory(ctx, lessFn, informerFactory, opts...)
}

func NewTestQueueWithInformerFactory(
	ctx context.Context,
	lessFn framework.LessFunc,
	informerFactory informers.SharedInformerFactory,
	opts ...Option,
) *PriorityQueue {
	pq := NewPriorityQueue(lessFn, informerFactory, opts...)
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	return pq
}
