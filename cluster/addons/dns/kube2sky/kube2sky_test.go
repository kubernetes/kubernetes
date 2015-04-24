/*
Copyright 2015 Google Inc. All rights reserved.

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

package main

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	kapi "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kwatch "github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/coreos/go-etcd/etcd"
)

func TestDNSRecord(t *testing.T) {
	if got, want := newDNSRecord("1.2.3.4", 1500), `{"host":"1.2.3.4","port":1500,"priority":10,"weight":10,"ttl":30}`; got != want {
		t.Errorf("newDNSRecord(1.2.3.4, 1500) = %q, want %q", got, want)
	}
}

func TestDNSNames(t *testing.T) {
	if got, want := buildServiceName("", "svc", "default", "kubernetes.local"), "/skydns/local/kubernetes/default/svc"; got != want {
		t.Errorf("buildServiceName() = %q, want %q", got, want)
	}
	if got, want := buildServiceName("mainport", "svc", "default", "kubernetes.local"), "/skydns/local/kubernetes/default/svc/mainport"; got != want {
		t.Errorf("buildServiceName() = %q, want %q", got, want)
	}
	if got, want := buildPodName("qehzd", "", "svc", "default", "kubernetes.local"), "/skydns/local/kubernetes/default/svc/qehzd"; got != want {
		t.Errorf("buildPodName() = %q, want %q", got, want)
	}
	if got, want := buildPodName("qehzd", "mainport", "svc", "default", "kubernetes.local"), "/skydns/local/kubernetes/default/svc/mainport/qehzd"; got != want {
		t.Errorf("buildPodName() = %q, want %q", got, want)
	}
}

func TestBuildExistingNodes(t *testing.T) {
	hosts := make(map[string]string)
	buildExistingHosts(hosts, &etcd.Node{
		Key:   "/skydns/local/kubernetes/aaa",
		Value: "value1",
		Nodes: []*etcd.Node{
			{
				Key:   "/skydns/local/kubernetes/aaa/123",
				Value: "value2",
			},
			{
				Key:   "/skydns/local/kubernetes/aaa/456",
				Value: "value3",
			},
		},
	})
	if want := map[string]string{
		"/skydns/local/kubernetes/aaa":     "value1",
		"/skydns/local/kubernetes/aaa/123": "value2",
		"/skydns/local/kubernetes/aaa/456": "value3",
	}; !reflect.DeepEqual(hosts, want) {
		t.Errorf("buildExistingHosts(hosts, ...) filled hosts = %q, want %q", hosts, want)
	}
}

type fakeEtcd struct {
	ch chan string
}

func (e *fakeEtcd) Set(key string, value string, ttl uint64) (*etcd.Response, error) {
	e.ch <- fmt.Sprintf("Set(%q, %q, %d)", key, value, ttl)
	return &etcd.Response{}, nil
}

func (e *fakeEtcd) Delete(key string, recursive bool) (*etcd.Response, error) {
	e.ch <- fmt.Sprintf("Delete(%q, %t)", key, recursive)
	return &etcd.Response{}, nil
}

func TestUpdater(t *testing.T) {
	*svcDomain = "svc.kubernetes.local"
	*podDomain = "pod.kubernetes.local"
	e := &fakeEtcd{make(chan string)}
	u := newUpdater(e)
	// This service matches one running pod.
	u.ch <- []kapi.Service{
		{
			ObjectMeta: kapi.ObjectMeta{
				Name:      "svc1",
				Namespace: "default",
			},
			Spec: kapi.ServiceSpec{
				PortalIP: "1.2.3.4",
				Ports:    []kapi.ServicePort{{Port: 1500}},
				Selector: map[string]string{"label1": "value1"},
			},
		},
	}
	// This pod matches the service.
	u.ch <- []kapi.Pod{
		{
			ObjectMeta: kapi.ObjectMeta{
				Name:      "pod1",
				Namespace: "default",
				Labels:    map[string]string{"label1": "value1", "label2": "irrelevant"},
			},
			Status: kapi.PodStatus{
				PodIP: "1.2.1.2",
				Phase: kapi.PodRunning,
			},
		},
	}
	done := make(chan error)
	go func() {
		done <- u.run(map[string]string{
			// This record corresponds to the service above, and should be retained.
			"/skydns/local/kubernetes/svc/default/svc1": newDNSRecord("1.2.3.4", 1500),
			// This record was left from some old service, and should be deleted.
			"/skydns/local/kubernetes/svc/default/svc3": newDNSRecord("1.2.3.7", 1515),
		})
	}()
	for _, T := range []struct {
		description string
		operation   func()
		want        []string
	}{
		{
			"initial update",
			func() {},
			[]string{
				fmt.Sprintf("Delete(%q, false)", "/skydns/local/kubernetes/svc/default/svc3"),
				fmt.Sprintf("Set(%q, %q, 0)", "/skydns/local/kubernetes/pod/default/svc1/pod1", newDNSRecord("1.2.1.2", 1500)),
			},
		},
		{
			"new pod started",
			func() {
				u.ch <- kwatch.Event{Type: kwatch.Added, Object: &kapi.Pod{
					ObjectMeta: kapi.ObjectMeta{
						Name:      "pod2",
						Namespace: "default",
						Labels:    map[string]string{"label1": "value1"},
					},
					Status: kapi.PodStatus{
						PodIP: "1.2.1.3",
						Phase: kapi.PodRunning,
					},
				}}
			},
			[]string{
				fmt.Sprintf("Set(%q, %q, 0)", "/skydns/local/kubernetes/pod/default/svc1/pod2", newDNSRecord("1.2.1.3", 1500)),
			},
		},
		{
			"pod deleted",
			func() {
				u.ch <- kwatch.Event{Type: kwatch.Deleted, Object: &kapi.Pod{
					ObjectMeta: kapi.ObjectMeta{
						Name:      "pod2",
						Namespace: "default",
						Labels:    map[string]string{"label1": "value1"},
					},
					Status: kapi.PodStatus{
						PodIP: "1.2.1.3",
						Phase: kapi.PodRunning,
					},
				}}
			},
			[]string{
				fmt.Sprintf("Delete(%q, false)", "/skydns/local/kubernetes/pod/default/svc1/pod2"),
			},
		},
		{
			"pod stopped",
			func() {
				u.ch <- kwatch.Event{Type: kwatch.Modified, Object: &kapi.Pod{
					ObjectMeta: kapi.ObjectMeta{
						Name:      "pod1",
						Namespace: "default",
						Labels:    map[string]string{"label1": "value1", "label2": "irrelevant"},
					},
					Status: kapi.PodStatus{
						PodIP: "1.2.1.2",
						Phase: kapi.PodFailed,
					},
				}}
			},
			[]string{
				fmt.Sprintf("Delete(%q, false)", "/skydns/local/kubernetes/pod/default/svc1/pod1"),
			},
		},
		{
			"service deleted",
			func() {
				u.ch <- kwatch.Event{Type: kwatch.Deleted, Object: &kapi.Service{
					ObjectMeta: kapi.ObjectMeta{
						Name:      "svc1",
						Namespace: "default",
					},
					Spec: kapi.ServiceSpec{
						PortalIP: "1.2.3.4",
						Ports:    []kapi.ServicePort{{Port: 1500}},
						Selector: map[string]string{"label1": "value1"},
					},
				}}
			},
			[]string{
				fmt.Sprintf("Delete(%q, false)", "/skydns/local/kubernetes/svc/default/svc1"),
			},
		},
	} {
		T.operation()
		for _, want := range T.want {
			select {
			case got := <-e.ch:
				if got != want {
					t.Fatalf("%s: got %s, want %s", T.description, got, want)
				}
			case <-time.After(5 * time.Second):
				t.Fatalf("%s: timeout, want %s", T.description, want)
			}
		}
	}
	u.stop()
	if err := <-done; err != nil {
		t.Error(err)
	}
}
