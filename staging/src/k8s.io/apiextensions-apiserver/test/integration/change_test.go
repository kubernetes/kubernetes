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

package integration

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
)

func TestChangeCRD(t *testing.T) {
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()
	config.QPS = 1000
	config.Burst = 1000
	apiExtensionsClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.NamespaceScoped)
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionsClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	ns := "default"
	noxuNamespacedResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, "v1beta1")

	updateCRD := func() {
		noxuDefinitionToUpdate, err := apiExtensionsClient.ApiextensionsV1().CustomResourceDefinitions().Get(context.TODO(), noxuDefinition.Name, metav1.GetOptions{})
		if err != nil {
			t.Error(err)
			return
		}
		if len(noxuDefinitionToUpdate.Spec.Versions) == 1 {
			v2 := noxuDefinitionToUpdate.Spec.Versions[0]
			v2.Name = "v2"
			v2.Served = true
			v2.Storage = false
			noxuDefinitionToUpdate.Spec.Versions = append(noxuDefinitionToUpdate.Spec.Versions, v2)
		} else {
			noxuDefinitionToUpdate.Spec.Versions = noxuDefinitionToUpdate.Spec.Versions[0:1]
		}
		if _, err := apiExtensionsClient.ApiextensionsV1().CustomResourceDefinitions().Update(context.TODO(), noxuDefinitionToUpdate, metav1.UpdateOptions{}); err != nil && !apierrors.IsConflict(err) {
			t.Error(err)
		}
	}

	// Set up 10 watchers for custom resource.
	// We can't exercise them in a loop the same way as get requests, as watchcache
	// can reject them with 429 and Retry-After: 1 if it is uninitialized and even
	// though 429 is automatically retried, with frequent watchcache terminations and
	// reinitializations they could either end-up being rejected N times and fail or
	// or not initialize until the last watchcache reinitialization and then not be
	// terminated. Thus we exercise their termination explicitly at the beginning.
	wg := &sync.WaitGroup{}
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()

			w, err := noxuNamespacedResourceClient.Watch(context.TODO(), metav1.ListOptions{})
			if err != nil {
				t.Errorf("unexpected error establishing watch: %v", err)
				return
			}
			for event := range w.ResultChan() {
				switch event.Type {
				case watch.Added, watch.Modified, watch.Deleted:
					// all expected
				default:
					t.Errorf("unexpected watch event: %#v", event)
				}
			}
		}(i)
	}

	// Let all the established watches soak request loops soak
	time.Sleep(5 * time.Second)

	// Update CRD and ensure that all watches are gracefully terminated.
	updateCRD()

	drained := make(chan struct{})
	go func() {
		defer close(drained)
		wg.Wait()
	}()

	select {
	case <-drained:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("timed out waiting for watchers to be terminated")
	}

	stopChan := make(chan struct{})

	// Set up loop to modify CRD in the background
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stopChan:
				return
			default:
			}

			time.Sleep(10 * time.Millisecond)

			updateCRD()
		}
	}()

	// Set up 10 loops creating and reading custom resources
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			noxuInstanceToCreate := fixtures.NewNoxuInstance(ns, fmt.Sprintf("foo-%d", i))
			if _, err := noxuNamespacedResourceClient.Create(context.TODO(), noxuInstanceToCreate, metav1.CreateOptions{}); err != nil {
				t.Error(err)
				return
			}
			for {
				time.Sleep(10 * time.Millisecond)
				select {
				case <-stopChan:
					return
				default:
					if _, err := noxuNamespacedResourceClient.Get(context.TODO(), noxuInstanceToCreate.GetName(), metav1.GetOptions{}); err != nil {
						t.Error(err)
						continue
					}
				}
			}
		}(i)
	}

	// Let all the established get request loops soak
	time.Sleep(5 * time.Second)

	// Tear down
	close(stopChan)

	// Let loops drain
	drained = make(chan struct{})
	go func() {
		defer close(drained)
		wg.Wait()
	}()

	select {
	case <-drained:
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("timed out waiting for clients to complete")
	}
}
