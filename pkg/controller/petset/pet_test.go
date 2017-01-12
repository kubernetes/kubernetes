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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/testing/core"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
)

func newPetClient(client *clientset.Clientset) *apiServerPetClient {
	return &apiServerPetClient{
		c: client,
	}
}

func makeTwoDifferntPCB() (pcb1, pcb2 *pcb) {
	userAdded := v1.Volume{
		Name: "test",
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory},
		},
	}
	ps := newStatefulSet(2)
	pcb1, _ = newPCB("1", ps)
	pcb2, _ = newPCB("2", ps)
	pcb2.pod.Spec.Volumes = append(pcb2.pod.Spec.Volumes, userAdded)
	return pcb1, pcb2
}

func TestUpdatePetWithoutRetry(t *testing.T) {
	pcb1, pcb2 := makeTwoDifferntPCB()
	// invalid pet with empty pod
	invalidPcb := *pcb1
	invalidPcb.pod = nil

	testCases := []struct {
		realPet     *pcb
		expectedPet *pcb
		expectErr   bool
		requests    int
	}{
		// case 0: error occurs, no need to update
		{
			realPet:     pcb1,
			expectedPet: &invalidPcb,
			expectErr:   true,
			requests:    0,
		},
		// case 1: identical pet, no need to update
		{
			realPet:     pcb1,
			expectedPet: pcb1,
			expectErr:   false,
			requests:    0,
		},
		// case 2: need to call update once
		{
			realPet:     pcb1,
			expectedPet: pcb2,
			expectErr:   false,
			requests:    1,
		},
	}

	for k, tc := range testCases {
		body := runtime.EncodeOrDie(testapi.Default.Codec(), &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "empty_pod"}})
		fakeHandler := utiltesting.FakeHandler{
			StatusCode:   200,
			ResponseBody: string(body),
		}
		testServer := httptest.NewServer(&fakeHandler)

		client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
		petClient := newPetClient(client)
		err := petClient.Update(tc.realPet, tc.expectedPet)

		if tc.expectErr != (err != nil) {
			t.Errorf("case %d: expect error(%v), got err: %v", k, tc.expectErr, err)
		}
		fakeHandler.ValidateRequestCount(t, tc.requests)
		testServer.Close()
	}
}

func TestUpdatePetWithFailure(t *testing.T) {
	fakeHandler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "{}",
	}
	testServer := httptest.NewServer(&fakeHandler)
	defer testServer.Close()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: testServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &api.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	petClient := newPetClient(client)

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

	actions := fakeClient.Actions()
	if len(actions) != 2 {
		t.Errorf("Expect 2 actions, got %d actions", len(actions))
	}
	for i := 0; i < len(actions); i++ {
		a := actions[i]
		if a.GetResource().Resource != "pods" {
			t.Errorf("Unexpected action %+v", a)
			continue
		}

		switch action := a.(type) {
		case core.GetAction:
			if i%2 == 0 {
				t.Errorf("Unexpected Get action")
			}
			// Make sure the get is for the right pod
			if action.GetName() != pcb2.pod.Name {
				t.Errorf("Expected get pod %v, got %q instead", pcb2.pod.Name, action.GetName())
			}
		case core.UpdateAction:
			if i%2 == 1 {
				t.Errorf("Unexpected Update action")
			}
		default:
			t.Errorf("Unexpected action %+v", a)
			break
		}
	}
}
