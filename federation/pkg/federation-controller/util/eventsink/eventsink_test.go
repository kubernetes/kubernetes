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

package eventsink

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	core "k8s.io/client-go/testing"
	fakefedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	kubev1 "k8s.io/kubernetes/pkg/api/v1"

	"github.com/stretchr/testify/assert"
)

func TestEventSink(t *testing.T) {
	fakeFederationClient := &fakefedclientset.Clientset{}
	createdChan := make(chan runtime.Object, 100)
	fakeFederationClient.AddReactor("create", "events", func(action core.Action) (bool, runtime.Object, error) {
		createAction := action.(core.CreateAction)
		obj := createAction.GetObject()
		createdChan <- obj
		return true, obj, nil
	})
	updateChan := make(chan runtime.Object, 100)
	fakeFederationClient.AddReactor("update", "events", func(action core.Action) (bool, runtime.Object, error) {
		updateAction := action.(core.UpdateAction)
		obj := updateAction.GetObject()
		updateChan <- obj
		return true, obj, nil
	})

	event := clientv1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bzium",
			Namespace: "ns",
		},
	}
	sink := NewFederatedEventSink(fakeFederationClient)
	eventUpdated, err := sink.Create(&event)
	assert.NoError(t, err)
	eventV1 := GetObjectFromChan(createdChan).(*kubev1.Event)
	assert.NotNil(t, eventV1)
	// Just some simple sanity checks.
	assert.Equal(t, event.Name, eventV1.Name)
	assert.Equal(t, event.Name, eventUpdated.Name)

	eventUpdated, err = sink.Update(&event)
	assert.NoError(t, err)
	eventV1 = GetObjectFromChan(updateChan).(*kubev1.Event)
	assert.NotNil(t, eventV1)
	// Just some simple sanity checks.
	assert.Equal(t, event.Name, eventV1.Name)
	assert.Equal(t, event.Name, eventUpdated.Name)
}
