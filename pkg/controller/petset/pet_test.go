/*
Copyright 2016 The Kubernetes Authors.

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

package petset

import (
	"fmt"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
)

func makeTwoDifferntPCB() (pcb1, pcb2 *pcb) {
	userAdded := api.Volume{
		Name: "test",
		VolumeSource: api.VolumeSource{
			EmptyDir: &api.EmptyDirVolumeSource{Medium: api.StorageMediumMemory},
		},
	}
	ps := newStatefulSet(2)
	pcb1, _ = newPCB("1", ps)
	pcb2, _ = newPCB("2", ps)
	pcb2.pod.Spec.Volumes = append(pcb2.pod.Spec.Volumes, userAdded)
	return pcb1, pcb2
}

func TestUpdatePetWithoutRetry(t *testing.T) {
	body := runtime.EncodeOrDie(testapi.Default.Codec(), &api.Pod{ObjectMeta: api.ObjectMeta{Name: "empty_pod"}})
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: string(body),
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(api.GroupName).GroupVersion}})
	petClient := apiServerPetClient{
		c: client,
	}

	pcb1, pcb2 := makeTwoDifferntPCB()
	// invalid pet with empty pod
	pcb3 := *pcb1
	pcb3.pod = nil

	testCases := []struct {
		realPet     *pcb
		expectedPet *pcb
		errOK       func(error) bool
	}{
		// case 1: error occurs, no need to update
		{
			realPet:     pcb1,
			expectedPet: &pcb3,
			errOK:       func(err error) bool { return err != nil },
		},
		// case 2: identical pet, no need to update
		{
			realPet:     pcb1,
			expectedPet: pcb1,
			errOK:       func(err error) bool { return err == nil },
		},
		// case 3: need to call update once
		{
			realPet:     pcb1,
			expectedPet: pcb2,
			errOK:       func(err error) bool { return err == nil },
		},
	}

	for k, test := range testCases {
		if err := petClient.Update(test.realPet, test.expectedPet); !test.errOK(err) {
			t.Errorf("case %d: unexpected error: %v", k+1, err)
		}
	}
}

func TestUpdatePetWithFailure(t *testing.T) {
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "{}",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(api.GroupName).GroupVersion}})
	petClient := apiServerPetClient{
		c: client,
	}

	pcb1, pcb2 := makeTwoDifferntPCB()

	if err := petClient.Update(pcb1, pcb2); err == nil {
		t.Errorf("expect error, got nil")
	}
	// 1 Update and 1 GET, both of which fail
	fakeHandler.ValidateRequestCount(t, 2)
}

func TestUpdatePetRetrySucceed(t *testing.T) {
	pcb1, pcb2 := makeTwoDifferntPCB()

	fakeClient := &fake.Clientset{}
	fakeClient.AddReactor("get", "pods", func(action core.Action) (bool, runtime.Object, error) {
		return true, pcb2.pod, nil
	})
	fakeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("Fake error")
	})
	petClient := apiServerPetClient{
		c: fakeClient,
	}

	if err := petClient.Update(pcb1, pcb2); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	updates, gets := 0, 0
	for _, a := range fakeClient.Actions() {
		if a.GetResource().Resource != "pods" {
			t.Errorf("Unexpected action %+v", a)
			continue
		}

		switch action := a.(type) {
		case core.GetAction:
			gets++
			// Make sure the get is for the right pod
			if action.GetName() != pcb2.pod.Name {
				t.Errorf("Expected get pod %v, got %q instead", pcb2.pod.Name, action.GetName())
			}
		case core.UpdateAction:
			updates++
		default:
			t.Errorf("Unexpected action %+v", a)
			break
		}
	}
	// 1 GET and 1 Update, both of which succeed
	if gets != 1 || updates != 1 {
		t.Errorf("Expected 1 get and 1 update, got %d gets %d updates", gets, updates)
	}
}
