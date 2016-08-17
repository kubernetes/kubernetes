// +build integration,!no-etcd

/*
Copyright 2014 The Kubernetes Authors.

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

package client

import (
	"fmt"
	"log"
	"reflect"
	rt "runtime"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/watch"
	e2e "k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestClient(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	ns := framework.CreateTestingNamespace("client", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	info, err := client.Discovery().ServerVersion()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e, a := version.Get(), *info; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %#v, got %#v", e, a)
	}

	pods, err := client.Pods(ns.Name).List(api.ListOptions{})
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
			Namespace:    ns.Name,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "test",
				},
			},
		},
	}

	got, err := client.Pods(ns.Name).Create(pod)
	if err == nil {
		t.Fatalf("unexpected non-error: %v", got)
	}

	// get a created pod
	pod.Spec.Containers[0].Image = "an-image"
	got, err = client.Pods(ns.Name).Create(pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Name == "" {
		t.Errorf("unexpected empty pod Name %v", got)
	}

	// pod is shown, but not scheduled
	pods, err = client.Pods(ns.Name).List(api.ListOptions{})
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

func TestAtomicPut(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	c := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	ns := framework.CreateTestingNamespace("atomic-put", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	rcBody := api.ReplicationController{
		TypeMeta: unversioned.TypeMeta{
			APIVersion: c.APIVersion().String(),
		},
		ObjectMeta: api.ObjectMeta{
			Name:      "atomicrc",
			Namespace: ns.Name,
			Labels: map[string]string{
				"name": "atomicrc",
			},
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 0,
			Selector: map[string]string{
				"foo": "bar",
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "name", Image: "image"},
					},
				},
			},
		},
	}
	rcs := c.ReplicationControllers(ns.Name)
	rc, err := rcs.Create(&rcBody)
	if err != nil {
		t.Fatalf("Failed creating atomicRC: %v", err)
	}
	testLabels := labels.Set{
		"foo": "bar",
	}
	for i := 0; i < 5; i++ {
		// a: z, b: y, etc...
		testLabels[string([]byte{byte('a' + i)})] = string([]byte{byte('z' - i)})
	}
	var wg sync.WaitGroup
	wg.Add(len(testLabels))
	for label, value := range testLabels {
		go func(l, v string) {
			defer wg.Done()
			for {
				tmpRC, err := rcs.Get(rc.Name)
				if err != nil {
					t.Errorf("Error getting atomicRC: %v", err)
					continue
				}
				if tmpRC.Spec.Selector == nil {
					tmpRC.Spec.Selector = map[string]string{l: v}
					tmpRC.Spec.Template.Labels = map[string]string{l: v}
				} else {
					tmpRC.Spec.Selector[l] = v
					tmpRC.Spec.Template.Labels[l] = v
				}
				tmpRC, err = rcs.Update(tmpRC)
				if err != nil {
					if apierrors.IsConflict(err) {
						// This is what we expect.
						continue
					}
					t.Errorf("Unexpected error putting atomicRC: %v", err)
					continue
				}
				return
			}
		}(label, value)
	}
	wg.Wait()
	rc, err = rcs.Get(rc.Name)
	if err != nil {
		t.Fatalf("Failed getting atomicRC after writers are complete: %v", err)
	}
	if !reflect.DeepEqual(testLabels, labels.Set(rc.Spec.Selector)) {
		t.Errorf("Selector PUTs were not atomic: wanted %v, got %v", testLabels, rc.Spec.Selector)
	}
}

func TestPatch(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	c := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	ns := framework.CreateTestingNamespace("patch", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	name := "patchpod"
	resource := "pods"
	podBody := api.Pod{
		TypeMeta: unversioned.TypeMeta{
			APIVersion: c.APIVersion().String(),
		},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns.Name,
			Labels:    map[string]string{},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "name", Image: "image"},
			},
		},
	}
	pods := c.Pods(ns.Name)
	pod, err := pods.Create(&podBody)
	if err != nil {
		t.Fatalf("Failed creating patchpods: %v", err)
	}

	patchBodies := map[unversioned.GroupVersion]map[api.PatchType]struct {
		AddLabelBody        []byte
		RemoveLabelBody     []byte
		RemoveAllLabelsBody []byte
	}{
		v1.SchemeGroupVersion: {
			api.JSONPatchType: {
				[]byte(`[{"op":"add","path":"/metadata/labels","value":{"foo":"bar","baz":"qux"}}]`),
				[]byte(`[{"op":"remove","path":"/metadata/labels/foo"}]`),
				[]byte(`[{"op":"remove","path":"/metadata/labels"}]`),
			},
			api.MergePatchType: {
				[]byte(`{"metadata":{"labels":{"foo":"bar","baz":"qux"}}}`),
				[]byte(`{"metadata":{"labels":{"foo":null}}}`),
				[]byte(`{"metadata":{"labels":null}}`),
			},
			api.StrategicMergePatchType: {
				[]byte(`{"metadata":{"labels":{"foo":"bar","baz":"qux"}}}`),
				[]byte(`{"metadata":{"labels":{"foo":null}}}`),
				[]byte(`{"metadata":{"labels":{"$patch":"replace"}}}`),
			},
		},
	}

	pb := patchBodies[c.APIVersion()]

	execPatch := func(pt api.PatchType, body []byte) error {
		return c.Patch(pt).
			Resource(resource).
			Namespace(ns.Name).
			Name(name).
			Body(body).
			Do().
			Error()
	}
	for k, v := range pb {
		// add label
		err := execPatch(k, v.AddLabelBody)
		if err != nil {
			t.Fatalf("Failed updating patchpod with patch type %s: %v", k, err)
		}
		pod, err = pods.Get(name)
		if err != nil {
			t.Fatalf("Failed getting patchpod: %v", err)
		}
		if len(pod.Labels) != 2 || pod.Labels["foo"] != "bar" || pod.Labels["baz"] != "qux" {
			t.Errorf("Failed updating patchpod with patch type %s: labels are: %v", k, pod.Labels)
		}

		// remove one label
		err = execPatch(k, v.RemoveLabelBody)
		if err != nil {
			t.Fatalf("Failed updating patchpod with patch type %s: %v", k, err)
		}
		pod, err = pods.Get(name)
		if err != nil {
			t.Fatalf("Failed getting patchpod: %v", err)
		}
		if len(pod.Labels) != 1 || pod.Labels["baz"] != "qux" {
			t.Errorf("Failed updating patchpod with patch type %s: labels are: %v", k, pod.Labels)
		}

		// remove all labels
		err = execPatch(k, v.RemoveAllLabelsBody)
		if err != nil {
			t.Fatalf("Failed updating patchpod with patch type %s: %v", k, err)
		}
		pod, err = pods.Get(name)
		if err != nil {
			t.Fatalf("Failed getting patchpod: %v", err)
		}
		if pod.Labels != nil {
			t.Errorf("Failed remove all labels from patchpod with patch type %s: %v", k, pod.Labels)
		}
	}
}

func TestPatchWithCreateOnUpdate(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	c := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	ns := framework.CreateTestingNamespace("patch-with-create", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	endpointTemplate := &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Name:      "patchendpoint",
			Namespace: ns.Name,
		},
		Subsets: []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []api.EndpointPort{{Port: 80, Protocol: api.ProtocolTCP}},
			},
		},
	}

	patchEndpoint := func(json []byte) (runtime.Object, error) {
		return c.Patch(api.MergePatchType).Resource("endpoints").Namespace(ns.Name).Name("patchendpoint").Body(json).Do().Get()
	}

	// Make sure patch doesn't get to CreateOnUpdate
	{
		endpointJSON, err := runtime.Encode(api.Codecs.LegacyCodec(v1.SchemeGroupVersion), endpointTemplate)
		if err != nil {
			t.Fatalf("Failed creating endpoint JSON: %v", err)
		}
		if obj, err := patchEndpoint(endpointJSON); !apierrors.IsNotFound(err) {
			t.Errorf("Expected notfound creating from patch, got error=%v and object: %#v", err, obj)
		}
	}

	// Create the endpoint (endpoints set AllowCreateOnUpdate=true) to get a UID and resource version
	createdEndpoint, err := c.Endpoints(ns.Name).Update(endpointTemplate)
	if err != nil {
		t.Fatalf("Failed creating endpoint: %v", err)
	}

	// Make sure identity patch is accepted
	{
		endpointJSON, err := runtime.Encode(api.Codecs.LegacyCodec(v1.SchemeGroupVersion), createdEndpoint)
		if err != nil {
			t.Fatalf("Failed creating endpoint JSON: %v", err)
		}
		if _, err := patchEndpoint(endpointJSON); err != nil {
			t.Errorf("Failed patching endpoint: %v", err)
		}
	}

	// Make sure patch complains about a mismatched resourceVersion
	{
		endpointTemplate.Name = ""
		endpointTemplate.UID = ""
		endpointTemplate.ResourceVersion = "1"
		endpointJSON, err := runtime.Encode(api.Codecs.LegacyCodec(v1.SchemeGroupVersion), endpointTemplate)
		if err != nil {
			t.Fatalf("Failed creating endpoint JSON: %v", err)
		}
		if _, err := patchEndpoint(endpointJSON); !apierrors.IsConflict(err) {
			t.Errorf("Expected error, got %#v", err)
		}
	}

	// Make sure patch complains about mutating the UID
	{
		endpointTemplate.Name = ""
		endpointTemplate.UID = "abc"
		endpointTemplate.ResourceVersion = ""
		endpointJSON, err := runtime.Encode(api.Codecs.LegacyCodec(v1.SchemeGroupVersion), endpointTemplate)
		if err != nil {
			t.Fatalf("Failed creating endpoint JSON: %v", err)
		}
		if _, err := patchEndpoint(endpointJSON); !apierrors.IsInvalid(err) {
			t.Errorf("Expected error, got %#v", err)
		}
	}

	// Make sure patch complains about a mismatched name
	{
		endpointTemplate.Name = "changedname"
		endpointTemplate.UID = ""
		endpointTemplate.ResourceVersion = ""
		endpointJSON, err := runtime.Encode(api.Codecs.LegacyCodec(v1.SchemeGroupVersion), endpointTemplate)
		if err != nil {
			t.Fatalf("Failed creating endpoint JSON: %v", err)
		}
		if _, err := patchEndpoint(endpointJSON); !apierrors.IsBadRequest(err) {
			t.Errorf("Expected error, got %#v", err)
		}
	}

	// Make sure patch containing originally submitted JSON is accepted
	{
		endpointTemplate.Name = ""
		endpointTemplate.UID = ""
		endpointTemplate.ResourceVersion = ""
		endpointJSON, err := runtime.Encode(api.Codecs.LegacyCodec(v1.SchemeGroupVersion), endpointTemplate)
		if err != nil {
			t.Fatalf("Failed creating endpoint JSON: %v", err)
		}
		if _, err := patchEndpoint(endpointJSON); err != nil {
			t.Errorf("Failed patching endpoint: %v", err)
		}
	}
}

func TestAPIVersions(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	c := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	clientVersion := c.APIVersion().String()
	g, err := c.ServerGroups()
	if err != nil {
		t.Fatalf("Failed to get api versions: %v", err)
	}
	versions := unversioned.ExtractGroupVersions(g)

	// Verify that the server supports the API version used by the client.
	for _, version := range versions {
		if version == clientVersion {
			return
		}
	}
	t.Errorf("Server does not support APIVersion used by client. Server supported APIVersions: '%v', client APIVersion: '%v'", versions, clientVersion)
}

func TestSingleWatch(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	ns := framework.CreateTestingNamespace("single-watch", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	mkEvent := func(i int) *api.Event {
		name := fmt.Sprintf("event-%v", i)
		return &api.Event{
			ObjectMeta: api.ObjectMeta{
				Namespace: ns.Name,
				Name:      name,
			},
			InvolvedObject: api.ObjectReference{
				Namespace: ns.Name,
				Name:      name,
			},
			Reason: fmt.Sprintf("event %v", i),
		}
	}

	rv1 := ""
	for i := 0; i < 10; i++ {
		event := mkEvent(i)
		got, err := client.Events(ns.Name).Create(event)
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
		Namespace(ns.Name).
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
	rt.GOMAXPROCS(watcherCount)

	_, s := framework.RunAMaster(nil)
	defer s.Close()

	ns := framework.CreateTestingNamespace("multi-watch", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	dummyEvent := func(i int) *api.Event {
		name := fmt.Sprintf("unrelated-%v", i)
		return &api.Event{
			ObjectMeta: api.ObjectMeta{
				Name:      fmt.Sprintf("%v.%x", name, time.Now().UnixNano()),
				Namespace: ns.Name,
			},
			InvolvedObject: api.ObjectReference{
				Name:      name,
				Namespace: ns.Name,
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
		got, err := client.Pods(ns.Name).Create(&api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   name,
				Labels: labels.Set{"watchlabel": name},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{{
					Name:  "pause",
					Image: e2e.GetPauseImageName(client),
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
			w, err := client.Pods(ns.Name).Watch(options)
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
					if _, err := client.Events(ns.Name).Create(dummyEvent(i)); err != nil {
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
					_, err := client.Pods(ns.Name).Create(&api.Pod{
						ObjectMeta: api.ObjectMeta{
							Name: name,
						},
						Spec: api.PodSpec{
							Containers: []api.Container{{
								Name:  "nothing",
								Image: e2e.GetPauseImageName(client),
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
			pod, err := client.Pods(ns.Name).Get(name)
			if err != nil {
				panic(fmt.Sprintf("Couldn't get %v: %v", name, err))
			}
			pod.Spec.Containers[0].Image = e2e.GetPauseImageName(client)
			sentTimes <- timePair{time.Now(), name}
			if _, err := client.Pods(ns.Name).Update(pod); err != nil {
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

func runSelfLinkTestOnNamespace(t *testing.T, c *client.Client, namespace string) {
	podBody := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "selflinktest",
			Namespace: namespace,
			Labels: map[string]string{
				"name": "selflinktest",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "name", Image: "image"},
			},
		},
	}
	pod, err := c.Pods(namespace).Create(&podBody)
	if err != nil {
		t.Fatalf("Failed creating selflinktest pod: %v", err)
	}
	if err = c.Get().RequestURI(pod.SelfLink).Do().Into(pod); err != nil {
		t.Errorf("Failed listing pod with supplied self link '%v': %v", pod.SelfLink, err)
	}

	podList, err := c.Pods(namespace).List(api.ListOptions{})
	if err != nil {
		t.Errorf("Failed listing pods: %v", err)
	}

	if err = c.Get().RequestURI(podList.SelfLink).Do().Into(podList); err != nil {
		t.Errorf("Failed listing pods with supplied self link '%v': %v", podList.SelfLink, err)
	}

	found := false
	for i := range podList.Items {
		item := &podList.Items[i]
		if item.Name != "selflinktest" {
			continue
		}
		found = true
		err = c.Get().RequestURI(item.SelfLink).Do().Into(pod)
		if err != nil {
			t.Errorf("Failed listing pod with supplied self link '%v': %v", item.SelfLink, err)
		}
		break
	}
	if !found {
		t.Errorf("never found selflinktest pod in namespace %s", namespace)
	}
}

func TestSelfLinkOnNamespace(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	ns := framework.CreateTestingNamespace("selflink", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	c := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	runSelfLinkTestOnNamespace(t, c, ns.Name)
}
