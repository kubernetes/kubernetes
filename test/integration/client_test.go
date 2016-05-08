// +build integration,!no-etcd

/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package integration

import (
	"fmt"
	"log"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestClient(t *testing.T) {
	_, s := framework.RunAMaster(t)
	defer s.Close()

	ns := api.NamespaceDefault
	framework.DeleteAllEtcdKeys()
	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	info, err := client.Discovery().ServerVersion()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e, a := version.Get(), *info; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %#v, got %#v", e, a)
	}

	pods, err := client.Pods(ns).List(api.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(pods.Items) != 0 {
		t.Errorf("expected no pods, got %#v", pods)
	}

	// get a validation error
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "test",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "test",
				},
			},
		},
	}

	got, err := client.Pods(ns).Create(pod)
	if err == nil {
		t.Fatalf("unexpected non-error: %v", got)
	}

	// get a created pod
	pod.Spec.Containers[0].Image = "an-image"
	got, err = client.Pods(ns).Create(pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name == "" {
		t.Errorf("unexpected empty pod Name %v", got)
	}

	// pod is shown, but not scheduled
	pods, err = client.Pods(ns).List(api.ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(pods.Items) != 1 {
		t.Errorf("expected one pod, got %#v", pods)
	}
	actual := pods.Items[0]
	if actual.Name != got.Name {
		t.Errorf("expected pod %#v, got %#v", got, actual)
	}
	if actual.Spec.NodeName != "" {
		t.Errorf("expected pod to be unscheduled, got %#v", actual)
	}
}

func TestSingleWatch(t *testing.T) {
	_, s := framework.RunAMaster(t)
	defer s.Close()

	ns := "blargh"
	deleteAllEtcdKeys()
	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	mkEvent := func(i int) *api.Event {
		name := fmt.Sprintf("event-%v", i)
		return &api.Event{
			ObjectMeta: api.ObjectMeta{
				Namespace: ns,
				Name:      name,
			},
			InvolvedObject: api.ObjectReference{
				Namespace: ns,
				Name:      name,
			},
			Reason: fmt.Sprintf("event %v", i),
		}
	}

	rv1 := ""
	for i := 0; i < 10; i++ {
		event := mkEvent(i)
		got, err := client.Events(ns).Create(event)
		if err != nil {
			t.Fatalf("Failed creating event %#q: %v", event, err)
		}
		if rv1 == "" {
			rv1 = got.ResourceVersion
			if rv1 == "" {
				t.Fatal("did not get a resource version.")
			}
		}
		t.Logf("Created event %#v", got.ObjectMeta)
	}

	w, err := client.Get().
		Prefix("watch").
		NamespaceIfScoped(ns, len(ns) > 0).
		Resource("events").
		Name("event-9").
		Param("resourceVersion", rv1).
		Watch()

	if err != nil {
		t.Fatalf("Failed watch: %v", err)
	}
	defer w.Stop()

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("watch took longer than %s", wait.ForeverTestTimeout.String())
	case got, ok := <-w.ResultChan():
		if !ok {
			t.Fatal("Watch channel closed unexpectedly.")
		}

		// We expect to see an ADD of event-9 and only event-9. (This
		// catches a bug where all the events would have been sent down
		// the channel.)
		if e, a := watch.Added, got.Type; e != a {
			t.Errorf("Wanted %v, got %v", e, a)
		}
		switch o := got.Object.(type) {
		case *api.Event:
			if e, a := "event-9", o.Name; e != a {
				t.Errorf("Wanted %v, got %v", e, a)
			}
		default:
			t.Fatalf("Unexpected watch event containing object %#q", got)
		}
	}
}

func TestMultiWatch(t *testing.T) {
	// Disable this test as long as it demonstrates a problem.
	// TODO: Reenable this test when we get #6059 resolved.
	return
	const watcherCount = 50
	runtime.GOMAXPROCS(watcherCount)

	framework.DeleteAllEtcdKeys()
	defer framework.DeleteAllEtcdKeys()
	_, s := framework.RunAMaster(t)
	defer s.Close()

	ns := api.NamespaceDefault
	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	dummyEvent := func(i int) *api.Event {
		name := fmt.Sprintf("unrelated-%v", i)
		return &api.Event{
			ObjectMeta: api.ObjectMeta{
				Name:      fmt.Sprintf("%v.%x", name, time.Now().UnixNano()),
				Namespace: ns,
			},
			InvolvedObject: api.ObjectReference{
				Name:      name,
				Namespace: ns,
			},
			Reason: fmt.Sprintf("unrelated change %v", i),
		}
	}

	type timePair struct {
		t    time.Time
		name string
	}

	receivedTimes := make(chan timePair, watcherCount*2)
	watchesStarted := sync.WaitGroup{}

	// make a bunch of pods and watch them
	for i := 0; i < watcherCount; i++ {
		watchesStarted.Add(1)
		name := fmt.Sprintf("multi-watch-%v", i)
		got, err := client.Pods(ns).Create(&api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   name,
				Labels: labels.Set{"watchlabel": name},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{{
					Name:  "nothing",
					Image: "gcr.io/google_containers/pause-amd64:3.0",
				}},
			},
		})

		if err != nil {
			t.Fatalf("Couldn't make %v: %v", name, err)
		}
		go func(name, rv string) {
			options := api.ListOptions{
				LabelSelector:   labels.Set{"watchlabel": name}.AsSelector(),
				ResourceVersion: rv,
			}
			w, err := client.Pods(ns).Watch(options)
			if err != nil {
				panic(fmt.Sprintf("watch error for %v: %v", name, err))
			}
			defer w.Stop()
			watchesStarted.Done()
			e, ok := <-w.ResultChan() // should get the update (that we'll do below)
			if !ok {
				panic(fmt.Sprintf("%v ended early?", name))
			}
			if e.Type != watch.Modified {
				panic(fmt.Sprintf("Got unexpected watch notification:\n%v: %+v %+v", name, e, e.Object))
			}
			receivedTimes <- timePair{time.Now(), name}
		}(name, got.ObjectMeta.ResourceVersion)
	}
	log.Printf("%v: %v pods made and watchers started", time.Now(), watcherCount)

	// wait for watches to start before we start spamming the system with
	// objects below, otherwise we'll hit the watch window restriction.
	watchesStarted.Wait()

	const (
		useEventsAsUnrelatedType = false
		usePodsAsUnrelatedType   = true
	)

	// make a bunch of unrelated changes in parallel
	if useEventsAsUnrelatedType {
		const unrelatedCount = 3000
		var wg sync.WaitGroup
		defer wg.Wait()
		changeToMake := make(chan int, unrelatedCount*2)
		changeMade := make(chan int, unrelatedCount*2)
		go func() {
			for i := 0; i < unrelatedCount; i++ {
				changeToMake <- i
			}
			close(changeToMake)
		}()
		for i := 0; i < 50; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					i, ok := <-changeToMake
					if !ok {
						return
					}
					if _, err := client.Events(ns).Create(dummyEvent(i)); err != nil {
						panic(fmt.Sprintf("couldn't make an event: %v", err))
					}
					changeMade <- i
				}
			}()
		}

		for i := 0; i < 2000; i++ {
			<-changeMade
			if (i+1)%50 == 0 {
				log.Printf("%v: %v unrelated changes made", time.Now(), i+1)
			}
		}
	}
	if usePodsAsUnrelatedType {
		const unrelatedCount = 3000
		var wg sync.WaitGroup
		defer wg.Wait()
		changeToMake := make(chan int, unrelatedCount*2)
		changeMade := make(chan int, unrelatedCount*2)
		go func() {
			for i := 0; i < unrelatedCount; i++ {
				changeToMake <- i
			}
			close(changeToMake)
		}()
		for i := 0; i < 50; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					i, ok := <-changeToMake
					if !ok {
						return
					}
					name := fmt.Sprintf("unrelated-%v", i)
					_, err := client.Pods(ns).Create(&api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name: name,
						},
						Spec: api.PodSpec{
							Containers: []api.Container{{
								Name:  "nothing",
								Image: "gcr.io/google_containers/pause-amd64:3.0",
							}},
						},
					})

					if err != nil {
						panic(fmt.Sprintf("couldn't make unrelated pod: %v", err))
					}
					changeMade <- i
				}
			}()
		}

		for i := 0; i < 2000; i++ {
			<-changeMade
			if (i+1)%50 == 0 {
				log.Printf("%v: %v unrelated changes made", time.Now(), i+1)
			}
		}
	}

	// Now we still have changes being made in parallel, but at least 1000 have been made.
	// Make some updates to send down the watches.
	sentTimes := make(chan timePair, watcherCount*2)
	for i := 0; i < watcherCount; i++ {
		go func(i int) {
			name := fmt.Sprintf("multi-watch-%v", i)
			pod, err := client.Pods(ns).Get(name)
			if err != nil {
				panic(fmt.Sprintf("Couldn't get %v: %v", name, err))
			}
			pod.Spec.Containers[0].Image = "gcr.io/google_containers/pause-amd64:3.0"
			sentTimes <- timePair{time.Now(), name}
			if _, err := client.Pods(ns).Update(pod); err != nil {
				panic(fmt.Sprintf("Couldn't make %v: %v", name, err))
			}
		}(i)
	}

	sent := map[string]time.Time{}
	for i := 0; i < watcherCount; i++ {
		tp := <-sentTimes
		sent[tp.name] = tp.t
	}
	log.Printf("all changes made")
	dur := map[string]time.Duration{}
	for i := 0; i < watcherCount; i++ {
		tp := <-receivedTimes
		delta := tp.t.Sub(sent[tp.name])
		dur[tp.name] = delta
		log.Printf("%v: %v", tp.name, delta)
	}
	log.Printf("all watches ended")
	t.Errorf("durations: %v", dur)
}
