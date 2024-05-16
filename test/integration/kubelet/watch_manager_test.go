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

package kubelet

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/kubelet/util/manager"
	"k8s.io/kubernetes/test/integration/framework"
	testingclock "k8s.io/utils/clock/testing"
)

func TestWatchBasedManager(t *testing.T) {
	testNamespace := "test-watch-based-manager"
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	const n = 10
	server.ClientConfig.QPS = 10000
	server.ClientConfig.Burst = 10000
	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := client.CoreV1().Namespaces().Create(context.TODO(), (&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}), metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	listObj := func(namespace string, options metav1.ListOptions) (runtime.Object, error) {
		return client.CoreV1().Secrets(namespace).List(context.TODO(), options)
	}
	watchObj := func(namespace string, options metav1.ListOptions) (watch.Interface, error) {
		return client.CoreV1().Secrets(namespace).Watch(context.TODO(), options)
	}
	newObj := func() runtime.Object { return &v1.Secret{} }
	// We want all watches to be up and running to stress test it.
	// So don't treat any secret as immutable here.
	isImmutable := func(_ runtime.Object) bool { return false }
	fakeClock := testingclock.NewFakeClock(time.Now())

	stopCh := make(chan struct{})
	defer close(stopCh)
	store := manager.NewObjectCache(listObj, watchObj, newObj, isImmutable, schema.GroupResource{Group: "v1", Resource: "secrets"}, fakeClock, time.Minute, stopCh)

	// create 1000 secrets in parallel
	t.Log(time.Now(), "creating 1000 secrets")
	wg := sync.WaitGroup{}
	errCh := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				name := fmt.Sprintf("s%d", i*100+j)
				if _, err := client.CoreV1().Secrets(testNamespace).Create(context.TODO(), &v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: name}}, metav1.CreateOptions{}); err != nil {
					select {
					case errCh <- err:
					default:
					}
				}
			}
			fmt.Print(".")
		}(i)
	}

	wg.Wait()
	select {
	case err := <-errCh:
		t.Fatal(err)
	default:
	}
	t.Log(time.Now(), "finished creating 1000 secrets")

	// fetch all secrets
	wg = sync.WaitGroup{}
	errCh = make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				name := fmt.Sprintf("s%d", i*100+j)
				start := time.Now()
				store.AddReference(testNamespace, name, types.UID(name))
				err := wait.PollImmediate(10*time.Millisecond, 10*time.Second, func() (bool, error) {
					obj, err := store.Get(testNamespace, name)
					if err != nil {
						t.Logf("failed on %s, retrying: %v", name, err)
						return false, nil
					}
					if obj.(*v1.Secret).Name != name {
						return false, fmt.Errorf("wrong object: %v", obj.(*v1.Secret).Name)
					}
					return true, nil
				})
				if err != nil {
					select {
					case errCh <- fmt.Errorf("failed on :%s: %v", name, err):
					default:
					}
				}
				if d := time.Since(start); d > time.Second {
					t.Logf("%s took %v", name, d)
				}
			}
		}(i)
	}

	wg.Wait()
	select {
	case err = <-errCh:
		t.Fatal(err)
	default:
	}
}
