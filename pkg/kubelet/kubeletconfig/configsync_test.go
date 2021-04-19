/*
Copyright 2021 The Kubernetes Authors.

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

package kubeletconfig

import (
	"testing"

	"github.com/stretchr/testify/assert"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	fakeEventv1 "k8s.io/client-go/kubernetes/typed/core/v1/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint/store"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

var (
	nodeMeta       = metav1.ObjectMeta{Name: "mynode", UID: "mynode-uid"}
	configMapMeta1 = metav1.ObjectMeta{Name: "foo", Namespace: "bar", UID: "myconfigmap-12345", ResourceVersion: "12345"}
	configMapMeta2 = metav1.ObjectMeta{Name: "foo", Namespace: "bar", UID: "myconfigmap-54321", ResourceVersion: "54321"}
	ncs1           = &apiv1.NodeConfigSource{
		ConfigMap: &apiv1.ConfigMapNodeConfigSource{
			Namespace:       configMapMeta1.Namespace,
			Name:            configMapMeta1.Name,
			UID:             configMapMeta1.UID,
			ResourceVersion: configMapMeta1.ResourceVersion,
		}}
	ncs2 = &apiv1.NodeConfigSource{
		ConfigMap: &apiv1.ConfigMapNodeConfigSource{
			Namespace:       configMapMeta2.Namespace,
			Name:            configMapMeta2.Name,
			UID:             configMapMeta2.UID,
			ResourceVersion: configMapMeta2.ResourceVersion,
		}}
	config0 = &apiv1.ConfigMap{}
	config1 = &apiv1.ConfigMap{ObjectMeta: configMapMeta1, Data: map[string]string{
		"--v": "0",
	}}
	config2 = &apiv1.ConfigMap{ObjectMeta: configMapMeta2, Data: map[string]string{
		"--v": "1",
	}}
)

type testSync struct {
	actualNCS       *apiv1.NodeConfigSource
	assignedNCS     *apiv1.NodeConfigSource
	actualConfig    *apiv1.ConfigMap
	actualRestart   bool
	exceptedRestart bool
}

func TestSyncConfigSourceKubelet(t *testing.T) {
	testCases := map[string]testSync{
		"source, assignedNCS is nil": {
			actualNCS:       nil,
			assignedNCS:     nil,
			actualConfig:    config0,
			exceptedRestart: false,
		},
		"source is equal to assignedNCS": {
			actualNCS:       ncs1,
			assignedNCS:     ncs1,
			actualConfig:    config1,
			exceptedRestart: false,
		},
		"source is nil, assigned is non-nil": {
			actualNCS:       nil,
			assignedNCS:     ncs1,
			actualConfig:    config0,
			exceptedRestart: true,
		},
		"source is not nil, assigned is nil": {
			actualNCS:       ncs1,
			assignedNCS:     nil,
			actualConfig:    config1,
			exceptedRestart: true,
		},
		"source is not equal to assigned": {
			actualNCS:       ncs2,
			assignedNCS:     ncs1,
			actualConfig:    config2,
			exceptedRestart: true,
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			tempDir, err := utilfs.DefaultFs{}.TempDir("", "test-syncConfig-")
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			defer cleanupControllerStore(t, tempDir)

			cc, fakeClient, err := newTestController(&testCase, tempDir)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			cc.pokeConfigSourceWorker()
			cc.syncConfigSource(fakeClient, &fakeEventv1.FakeCoreV1{Fake: &fakeClient.Fake}, nodeMeta.Name)
			assertAssigned(t, testCase.actualNCS, cc.checkpointStore)
			assert.Equal(t, testCase.exceptedRestart, testCase.actualRestart)
		})
	}
}

func newTestController(ts *testSync, storeDir string) (*Controller, *fakeclientset.Clientset, error) {
	cc := NewController(storeDir, nil)

	// ensure the filesystem is initialized
	cc.Bootstrap()

	// fake nodeInformer
	fakeClient := fakeclientset.NewSimpleClientset()
	cc.nodeInformer = newSharedNodeInformer(fakeClient, nodeMeta.Name, nil, nil, nil)

	// fake remoteConfigSourceInformer
	rcs, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{
		ConfigMap: &apiv1.ConfigMapNodeConfigSource{Namespace: configMapMeta1.Namespace, Name: configMapMeta1.Name}})
	if err != nil {
		return nil, nil, err
	}
	cc.remoteConfigSourceInformer = rcs.Informer(fakeClient,
		cache.ResourceEventHandlerFuncs{AddFunc: nil, UpdateFunc: nil, DeleteFunc: nil})

	node := apiv1.Node{ObjectMeta: nodeMeta, Spec: apiv1.NodeSpec{ConfigSource: ts.actualNCS}}
	if err := cc.nodeInformer.GetStore().Add(&node); err != nil {
		return nil, nil, err
	}

	if err := cc.remoteConfigSourceInformer.GetStore().Add(ts.actualConfig); err != nil {
		return nil, nil, err
	}

	var remote checkpoint.RemoteConfigSource
	if ts.assignedNCS != nil {
		remote, _, err = checkpoint.NewRemoteConfigSource(ts.assignedNCS)
		if err != nil {
			return nil, nil, err
		}
	}

	if err := cc.checkpointStore.SetAssigned(remote); err != nil {
		return nil, nil, err
	}

	// mock restartForNewConfig to aviod to call os.Exit(0)
	restartForNewConfig = ts.fakeRestartForNewConfig

	return cc, fakeClient, nil
}

func cleanupControllerStore(t *testing.T, storeDir string) {
	err := utilfs.DefaultFs{}.RemoveAll(storeDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func (ts *testSync) fakeRestartForNewConfig(eventClient v1core.EventsGetter, nodeName string, source checkpoint.RemoteConfigSource) {
	ts.actualRestart = true
}

func assertAssigned(t *testing.T, expectedNCS *apiv1.NodeConfigSource, fsstore store.Store) {
	assigned, err := fsstore.Assigned()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if assigned == nil {
		assert.Equal(t, expectedNCS, (*apiv1.NodeConfigSource)(nil))
	} else {
		assert.Equal(t, expectedNCS, assigned.NodeConfigSource())
	}
}
