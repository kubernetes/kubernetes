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

package util

import (
	"fmt"
	"testing"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	fake_kube_release_1_4 "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4/fake"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"

	"github.com/stretchr/testify/assert"
)

// Fake federation view.
type fakeFederationView struct {
}

func (f fakeFederationView) GetClientsetForCluster(clusterName string) (kube_release_1_4.Interface, error) {
	return &fake_kube_release_1_4.Clientset{}, nil
}

func (f *fakeFederationView) GetReadyClusters() ([]*federation_api.Cluster, error) {
	return []*federation_api.Cluster{}, nil
}

func (f *fakeFederationView) GetReadyCluster(name string) (*federation_api.Cluster, bool, error) {
	return nil, false, nil
}

func (f *fakeFederationView) ClustersSynced() bool {
	return true
}

func TestFederatedUpdaterOK(t *testing.T) {
	addChan := make(chan string, 5)
	updateChan := make(chan string, 5)

	updater := NewFederatedUpdater(&fakeFederationView{},
		func(_ kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			service := obj.(*api_v1.Service)
			addChan <- service.Name
			return nil
		},
		func(_ kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			service := obj.(*api_v1.Service)
			updateChan <- service.Name
			return nil
		},
		noop)

	err := updater.Update([]FederatedOperation{
		{
			Type: OperationTypeAdd,
			Obj:  makeService("A", "s1"),
		},
		{
			Type: OperationTypeUpdate,
			Obj:  makeService("B", "s2"),
		},
	}, time.Minute)
	assert.NoError(t, err)
	add := <-addChan
	update := <-updateChan
	assert.Equal(t, "s1", add)
	assert.Equal(t, "s2", update)
}

func TestFederatedUpdaterError(t *testing.T) {
	updater := NewFederatedUpdater(&fakeFederationView{},
		func(_ kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			return fmt.Errorf("boom")
		}, noop, noop)

	err := updater.Update([]FederatedOperation{
		{
			Type: OperationTypeAdd,
			Obj:  makeService("A", "s1"),
		},
		{
			Type: OperationTypeUpdate,
			Obj:  makeService("B", "s1"),
		},
	}, time.Minute)
	assert.Error(t, err)
}

func TestFederatedUpdaterTimeout(t *testing.T) {
	start := time.Now()
	updater := NewFederatedUpdater(&fakeFederationView{},
		func(_ kube_release_1_4.Interface, obj pkg_runtime.Object) error {
			time.Sleep(time.Minute)
			return nil
		},
		noop, noop)

	err := updater.Update([]FederatedOperation{
		{
			Type: OperationTypeAdd,
			Obj:  makeService("A", "s1"),
		},
		{
			Type: OperationTypeUpdate,
			Obj:  makeService("B", "s1"),
		},
	}, time.Second)
	end := time.Now()
	assert.Error(t, err)
	assert.True(t, start.Add(10*time.Second).After(end))
}

func makeService(cluster, name string) *api_v1.Service {
	return &api_v1.Service{
		ObjectMeta: api_v1.ObjectMeta{
			Namespace: "ns1",
			Name:      name,
		},
	}
}

func noop(_ kube_release_1_4.Interface, _ pkg_runtime.Object) error {
	return nil
}
