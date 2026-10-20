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

package scheduler

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

// The DRA plugin reports not-yet-published sharing affinity metadata as an
// Error status, because the event that triggered publication may already have
// been consumed. The pod must go to backoff and be retried without any
// cluster event.
func TestAffinityNotCoherentErrorGoesToBackoff(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	pod := st.MakePod().Name("foo").Namespace("ns").UID("pod-uid").SchedulerName(testSchedulerName).Obj()
	client := clientsetfake.NewClientset(pod)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	schedFramework, err := tf.NewFramework(ctx,
		[]tf.RegisterPluginFunc{
			tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
			tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		},
		testSchedulerName,
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithEventRecorder(eventBroadcaster.NewRecorder(scheme.Scheme, testSchedulerName)),
		frameworkruntime.WithInformerFactory(informerFactory),
	)
	if err != nil {
		t.Fatal(err)
	}
	ar := metrics.NewMetricsAsyncRecorder(10, time.Second, ctx.Done())
	queue := internalqueue.NewSchedulingQueue(nil, informerFactory,
		internalqueue.WithMetricsRecorder(ar),
		internalqueue.WithPodInitialBackoffDuration(100*time.Millisecond),
		internalqueue.WithPodMaxBackoffDuration(100*time.Millisecond))
	sched := &Scheduler{client: client, SchedulingQueue: queue}
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	queue.Run(logger)
	defer queue.Close()

	queue.Add(ctx, pod)
	popped, err := queue.Pop(logger)
	if err != nil {
		t.Fatalf("Pop: %v", err)
	}
	notCoherent := errors.New("sharing affinity metadata not yet coherent")
	status := fwk.AsStatus(fmt.Errorf("sharing affinity metadata for pools [driver/pool] not yet published: %w", notCoherent)).WithPlugin("DynamicResources")
	sched.handleSchedulingFailure(ctx, schedFramework, popped.(*framework.QueuedPodInfo), status, nil, time.Now())

	if got := queue.UnschedulablePods(); len(got) != 0 {
		t.Fatalf("Error status parked pod as unschedulable: %v", got)
	}
	retried := make(chan error, 1)
	go func() {
		p, err := queue.Pop(logger)
		if err == nil && p.(*framework.QueuedPodInfo).Pod.UID != pod.UID {
			err = fmt.Errorf("popped unexpected pod %v", p)
		}
		retried <- err
	}()
	select {
	case err := <-retried:
		if err != nil {
			t.Fatalf("retry Pop: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("pod was not retried from backoff without a cluster event")
	}
}
