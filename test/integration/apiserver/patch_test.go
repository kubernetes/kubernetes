// +build integration,!no-etcd

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

package apiserver

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/pborman/uuid"

	kapierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	apiendpointhandlers "k8s.io/apiserver/pkg/endpoints/handlers"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestPatchConflicts(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})

	objName := "myobj"
	ns := "patch-namespace"

	if _, err := clientSet.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}); err != nil {
		t.Fatalf("Error creating namespace:%v", err)
	}
	if _, err := clientSet.CoreV1().Secrets(ns).Create(&v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: objName}}); err != nil {
		t.Fatalf("Error creating k8s resource:%v", err)
	}

	testcases := []struct {
		client   restclient.Interface
		resource string
	}{
		{
			client:   clientSet.CoreV1().RESTClient(),
			resource: "secrets",
		},
	}

	for _, tc := range testcases {
		successes := int32(0)

		// Force patch to deal with resourceVersion conflicts applying non-conflicting patches in parallel.
		// We want to ensure it handles reapplies in a retry loop without internal errors.
		wg := sync.WaitGroup{}
		for i := 0; i < (2 * apiendpointhandlers.MaxRetryWhenPatchConflicts); i++ {
			wg.Add(1)
			go func(labelName string) {
				defer wg.Done()
				labelValue := uuid.NewRandom().String()

				obj, err := tc.client.Patch(types.StrategicMergePatchType).
					Namespace(ns).
					Resource(tc.resource).
					Name(objName).
					Body([]byte(fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, labelName, labelValue))).
					Do().
					Get()

				if kapierrs.IsConflict(err) {
					t.Logf("tolerated %s conflict error patching %s: %v", labelName, tc.resource, err)
					return
				}
				if err != nil {
					t.Errorf("error patching %s: %v", tc.resource, err)
					return
				}

				accessor, err := meta.Accessor(obj)
				if err != nil {
					t.Errorf("error getting object from %s: %v", tc.resource, err)
					return
				}
				if accessor.GetLabels()[labelName] != labelValue {
					t.Errorf("patch of %s was ineffective, expected %s=%s, got labels %#v", tc.resource, labelName, labelValue, accessor.GetLabels())
					return
				}

				atomic.AddInt32(&successes, 1)
			}(fmt.Sprintf("label-%d", i))
		}
		wg.Wait()

		// Each successful patch can cause another patch retry loop to fail at most once. Hence, for a patch
		// retry loop to fail MaxRetryWhenPatchConflicts many times, at least MaxRetryWhenPatchConflicts must
		// have been successful before. Hence, if less patch request fail, something is wrong with the
		// patch handler logic.
		if successes < apiendpointhandlers.MaxRetryWhenPatchConflicts {
			t.Errorf("Expected at least %d successful patches for %s, got %d", apiendpointhandlers.MaxRetryWhenPatchConflicts, tc.resource, successes)
		} else {
			t.Logf("Got %d successful patches for %s", successes, tc.resource)
		}
	}
}
