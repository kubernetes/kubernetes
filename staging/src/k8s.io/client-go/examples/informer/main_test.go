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
	"fmt"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/testing/fuzzer"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	kapitesting "k8s.io/kubernetes/pkg/api/testing"
)

func podForTest(rand *rand.Rand) *v1.Pod {
	apiObjectFuzzer := fuzzer.FuzzerFor(
		kapitesting.FuzzerFuncs,
		rand,
		api.Codecs,
	)
	var p v1.Pod
	apiObjectFuzzer.Fuzz(&p)

	p.SetName(fmt.Sprintf("Pod%d", rand.Intn(10)))
	p.SetNamespace(fmt.Sprintf("Namespace%d", rand.Intn(10)))
	phases := []v1.PodPhase{
		"creating",
		"starting",
		"running",
		"stopping",
		"deleting",
	}
	p.Status.Phase = phases[rand.Intn(len(phases))]
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

	f := func(p *v1.Pod, operation func(runtime.Object)) bool {
		operation(p)
		return true
	}

	operations := []func(runtime.Object){
		fakeWatch.Add,
		fakeWatch.Delete,
	}

	err = quick.Check(
		f,
		&quick.Config{
			MaxCount: 1000,
			Values: func(values []reflect.Value, r *rand.Rand) {
				p := podForTest(r)
				values[0] = reflect.ValueOf(p)
				operation := operations[rand.Intn(len(operations))]
				values[1] = reflect.ValueOf(operation)
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(time.Second)
}
