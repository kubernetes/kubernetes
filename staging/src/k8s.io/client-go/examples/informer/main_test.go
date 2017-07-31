/*
Copyright 2017 The Kubernetes Authors.

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
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/testing/fuzzer"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	kapitesting "k8s.io/kubernetes/pkg/api/testing"
)

func podForTest(r *rand.Rand) *v1.Pod {
	apiObjectFuzzer := fuzzer.FuzzerFor(
		kapitesting.FuzzerFuncs,
		r,
		api.Codecs,
	)
	var p v1.Pod
	apiObjectFuzzer.Fuzz(&p)
	p.Spec.InitContainers = nil
	p.Status.InitContainerStatuses = nil
	return &p
}

func TestMain(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	clientset.PrependWatchReactor(
		"pods",
		clienttesting.DefaultWatchReactor(fakeWatch, nil),
	)
	factory := informers.NewSharedInformerFactory(clientset, 0)
	c := NewPodLoggingController(factory)

	stop := make(chan struct{})
	defer close(stop)

	err := c.Run(stop)
	if err != nil {
		t.Error(err)
	}

	f := func(p *v1.Pod) bool {
		// initialLineCount := glog.Stats.Info.Lines()
		// for _, pod := range tc.initial {
		//	fakeWatch.Delete(pod)
		// }
		// <-time.After(1 * time.Second)
		// actualLineCount := glog.Stats.Info.Lines() - initialLineCount
		// if tc.expectedLineCount != actualLineCount {
		//	t.Errorf(
		//		"Line count mismatch. Expected %d. Got %d",
		//		tc.expectedLineCount,
		//		actualLineCount,
		//	)
		// }
		fakeWatch.Add(p)
		// key, err := cache.MetaNamespaceKeyFunc(p)
		// if err == nil {
		//	t.Log(key)
		// } else {
		//	t.Error(err)
		// }
		return true
	}

	err = quick.Check(
		f,
		&quick.Config{
			MaxCount: 1000,
			Values: func(values []reflect.Value, r *rand.Rand) {
				p := podForTest(r)
				values[0] = reflect.ValueOf(p)
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(time.Second)
}
