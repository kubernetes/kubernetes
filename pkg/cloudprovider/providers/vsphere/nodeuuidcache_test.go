/*
Copyright 2018 The Kubernetes Authors.

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

package vsphere

import (
	"fmt"
	"testing"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
)

func TestAddNodeToCache(t *testing.T) {
	fakeClient := &fake.Clientset{}
	configMaps := map[string]*v1.ConfigMap{}
	fakeClient.AddReactor("create", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
		createAction := action.(core.CreateAction)
		configMap := createAction.GetObject().(*v1.ConfigMap)
		configMaps[configMap.Name] = configMap
		return true, createAction.GetObject(), nil
	})
	fakeClient.AddReactor("get", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
		getAction := action.(core.GetAction)
		configMapName := getAction.GetName()
		if configMap, ok := configMaps[configMapName]; ok {
			return true, configMap, nil
		}
		statusError := &apierrors.StatusError{
			ErrStatus: metav1.Status{
				Status: metav1.StatusFailure,
				Code:   int32(404),
				Reason: metav1.StatusReasonNotFound,
			},
		}

		return false, nil, statusError
	})
	vmUUIDCache := newNodeUUIDCache("vsphere-uuid", "kube-system")
	vmUUIDCache.kubeClient = fakeClient
	err := vmUUIDCache.addNode("real-node.local", "abcde-fghe")
	if err != nil {
		t.Errorf("expected error to be nil got %v", err)
	}
	vmUUID, err := vmUUIDCache.getVMUUID("real-node.local")
	if err != nil {
		t.Errorf("expected no error got %v", err)
	}
	if vmUUID != "abcde-fghe" {
		t.Errorf("expected uuid to be %s got %s", "abcde-fghe", vmUUID)
	}

	// test Adding more than 3600 node entries
	for i := 1; i <= (maxDataEntries + 10); i++ {
		err := vmUUIDCache.addNode(fmt.Sprintf("real-node%d.local", i), fmt.Sprintf("node-uuid-%d", i))
		if err != nil {
			t.Errorf("expected no error got %v", err)
		}
	}
	if len(vmUUIDCache.nodeUUIDMap) > maxDataEntries {
		t.Errorf("expected %d items but got %d", maxDataEntries, len(vmUUIDCache.nodeUUIDMap))
	}

	// Mark some nodes as deleted
	for i := 1; i < 100; i++ {
		err := vmUUIDCache.markDelete(fmt.Sprintf("real-node%d.local", i), fmt.Sprintf("node-uuid-%d", i))
		if err != nil {
			t.Errorf("expected no error but got %v", err)
		}
	}
	// Add some more nodes and verify that deleted nodes aren't pruned
	for i := 3700; i < 3750; i++ {
		err := vmUUIDCache.addNode(fmt.Sprintf("real-node%d.local", i), fmt.Sprintf("node-uuid-%d", i))
		if err != nil {
			t.Errorf("expected no error got %v", err)
		}
	}

	for i := 1; i < 100; i++ {
		vmUUID, err := vmUUIDCache.getVMUUID(fmt.Sprintf("real-node%d.local", i))
		if err != nil {
			t.Errorf("expected no error but got %v", err)
		}
		expectedUUID := fmt.Sprintf("node-uuid-%d", i)
		if vmUUID != expectedUUID {
			t.Errorf("expected uuid %s but got %s", expectedUUID, vmUUID)
		}
	}

}
