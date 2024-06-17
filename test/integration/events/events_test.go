/*
Copyright 2019 The Kubernetes Authors.

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

package events

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/scheme"
	typedv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/client-go/tools/record"
	ref "k8s.io/client-go/tools/reference"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestEventCompatibility(t *testing.T) {
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer result.TearDownFn()

	client := clientset.NewForConfigOrDie(result.ClientConfig)

	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
			UID:       "bar",
		},
	}

	regarding, err := ref.GetReference(scheme.Scheme, testPod)
	if err != nil {
		t.Fatal(err)
	}

	related, err := ref.GetPartialReference(scheme.Scheme, testPod, ".spec.containers[0]")
	if err != nil {
		t.Fatal(err)
	}

	stopCh := make(chan struct{})
	defer close(stopCh)
	oldBroadcaster := record.NewBroadcaster()
	defer oldBroadcaster.Shutdown()
	oldRecorder := oldBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "integration"})
	oldBroadcaster.StartRecordingToSink(&typedv1.EventSinkImpl{Interface: client.CoreV1().Events("")})
	oldRecorder.Eventf(regarding, v1.EventTypeNormal, "started", "note")

	newBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	defer newBroadcaster.Shutdown()
	newRecorder := newBroadcaster.NewRecorder(scheme.Scheme, "k8s.io/kube-scheduler")
	newBroadcaster.StartRecordingToSink(stopCh)
	newRecorder.Eventf(regarding, related, v1.EventTypeNormal, "memoryPressure", "killed", "memory pressure")
	err = wait.PollImmediate(100*time.Millisecond, 20*time.Second, func() (done bool, err error) {
		v1Events, err := client.EventsV1().Events("").List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}

		if len(v1Events.Items) != 2 {
			return false, nil
		}

		events, err := client.CoreV1().Events("").List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}

		if len(events.Items) != 2 {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
}

func TestEventSeries(t *testing.T) {
	result := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer result.TearDownFn()

	client := clientset.NewForConfigOrDie(result.ClientConfig)

	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
			UID:       "bar",
		},
	}

	regarding, err := ref.GetReference(scheme.Scheme, testPod)
	if err != nil {
		t.Fatal(err)
	}

	related, err := ref.GetPartialReference(scheme.Scheme, testPod, ".spec.containers[0]")
	if err != nil {
		t.Fatal(err)
	}

	stopCh := make(chan struct{})
	defer close(stopCh)

	broadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	defer broadcaster.Shutdown()
	recorder := broadcaster.NewRecorder(scheme.Scheme, "k8s.io/kube-scheduler")
	broadcaster.StartRecordingToSink(stopCh)
	recorder.Eventf(regarding, related, v1.EventTypeNormal, "memoryPressure", "killed", "memory pressure")
	recorder.Eventf(regarding, related, v1.EventTypeNormal, "memoryPressure", "killed", "memory pressure")
	err = wait.PollImmediate(100*time.Millisecond, 20*time.Second, func() (done bool, err error) {
		events, err := client.EventsV1().Events("").List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}

		if len(events.Items) != 1 {
			return false, nil
		}

		if events.Items[0].Series == nil {
			return false, nil
		}

		if events.Items[0].Series.Count != 2 {
			return false, fmt.Errorf("expected EventSeries to have a starting count of 2, got: %d", events.Items[0].Series.Count)
		}

		return true, nil
	})
	if err != nil {
		t.Fatalf("error waiting for an Event with a non nil Series to be created: %v", err)
	}

}
