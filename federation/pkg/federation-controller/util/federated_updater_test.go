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

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	kubeclientset "k8s.io/client-go/kubernetes"
	fakekubeclientset "k8s.io/client-go/kubernetes/fake"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"

	"github.com/stretchr/testify/assert"
)

// Fake federation view.
type fakeFederationView struct {
}

// Verify that fakeFederationView implements FederationView interface
var _ FederationView = &fakeFederationView{}

func (f *fakeFederationView) GetClientsetForCluster(clusterName string) (kubeclientset.Interface, error) {
	return &fakekubeclientset.Clientset{}, nil
}

func (f *fakeFederationView) GetReadyClusters() ([]*federationapi.Cluster, error) {
	return []*federationapi.Cluster{}, nil
}

func (f *fakeFederationView) GetUnreadyClusters() ([]*federationapi.Cluster, error) {
	return []*federationapi.Cluster{}, nil
}

func (f *fakeFederationView) GetReadyCluster(name string) (*federationapi.Cluster, bool, error) {
	return nil, false, nil
}

func (f *fakeFederationView) ClustersSynced() bool {
	return true
}

type fakeEventRecorder struct{}

func (f *fakeEventRecorder) Event(object pkgruntime.Object, eventtype, reason, message string) {}
func (f *fakeEventRecorder) Eventf(object pkgruntime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
}
func (f *fakeEventRecorder) PastEventf(object pkgruntime.Object, timestamp metav1.Time, eventtype, reason, messageFmt string, args ...interface{}) {
}

func TestFederatedUpdaterOK(t *testing.T) {
	addChan := make(chan string, 5)
	updateChan := make(chan string, 5)

	updater := NewFederatedUpdater(&fakeFederationView{}, "foo", time.Minute, &fakeEventRecorder{},
		func(_ kubeclientset.Interface, obj pkgruntime.Object) error {
			service := obj.(*apiv1.Service)
			addChan <- service.Name
			return nil
		},
		func(_ kubeclientset.Interface, obj pkgruntime.Object) error {
			service := obj.(*apiv1.Service)
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
	})
	assert.NoError(t, err)
	add := <-addChan
	update := <-updateChan
	assert.Equal(t, "s1", add)
	assert.Equal(t, "s2", update)
}

func TestFederatedUpdaterError(t *testing.T) {
	updater := NewFederatedUpdater(&fakeFederationView{}, "foo", time.Minute, &fakeEventRecorder{},
		func(_ kubeclientset.Interface, obj pkgruntime.Object) error {
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
	})
	assert.Error(t, err)
}

func TestFederatedUpdaterTimeout(t *testing.T) {
	start := time.Now()
	updater := NewFederatedUpdater(&fakeFederationView{}, "foo", time.Second, &fakeEventRecorder{},
		func(_ kubeclientset.Interface, obj pkgruntime.Object) error {
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
	})
	end := time.Now()
	assert.Error(t, err)
	assert.True(t, start.Add(10*time.Second).After(end))
}

func makeService(cluster, name string) *apiv1.Service {
	return &apiv1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "ns1",
			Name:      name,
		},
	}
}

func noop(_ kubeclientset.Interface, _ pkgruntime.Object) error {
	return nil
}
