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
	"reflect"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/record"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	kubev1 "k8s.io/kubernetes/pkg/api/v1"
)

// Implements k8s.io/client-go/tools/record.EventSink.
type FederatedEventSink struct {
	clientset fedclientset.Interface
}

// To check if all required functions are implemented.
var _ record.EventSink = &FederatedEventSink{}

func NewFederatedEventSink(clientset fedclientset.Interface) *FederatedEventSink {
	return &FederatedEventSink{
		clientset: clientset,
	}
}

// TODO this is uses a reflection conversion path and is very expensive.  federation should update to use client-go

var scheme = runtime.NewScheme()

func init() {
	// register client-go's and kube's Event type under two different GroupVersions
	// TODO: switch to client-go client for events
	scheme.AddKnownTypes(clientv1.SchemeGroupVersion, &clientv1.Event{})
	scheme.AddKnownTypes(schema.GroupVersion{Group: "fake-kube-" + kubev1.SchemeGroupVersion.Group, Version: kubev1.SchemeGroupVersion.Version}, &kubev1.Event{})

	if err := scheme.AddConversionFuncs(
		metav1.Convert_unversioned_Time_To_unversioned_Time,
	); err != nil {
		panic(err)
	}
	if err := scheme.AddGeneratedDeepCopyFuncs(
		conversion.GeneratedDeepCopyFunc{
			Fn:     metav1.DeepCopy_v1_Time,
			InType: reflect.TypeOf(&metav1.Time{}),
		},
	); err != nil {
		panic(err)
	}
}

func (fes *FederatedEventSink) Create(event *clientv1.Event) (*clientv1.Event, error) {
	kubeEvent := &kubev1.Event{}
	if err := scheme.Convert(event, kubeEvent, nil); err != nil {
		return nil, err
	}

	ret, err := fes.clientset.Core().Events(kubeEvent.Namespace).Create(kubeEvent)
	if err != nil {
		return nil, err
	}

	retEvent := &clientv1.Event{}
	if err := scheme.Convert(ret, retEvent, nil); err != nil {
		return nil, err
	}
	return retEvent, nil
}

func (fes *FederatedEventSink) Update(event *clientv1.Event) (*clientv1.Event, error) {
	kubeEvent := &kubev1.Event{}
	if err := scheme.Convert(event, kubeEvent, nil); err != nil {
		return nil, err
	}

	ret, err := fes.clientset.Core().Events(kubeEvent.Namespace).Update(kubeEvent)
	if err != nil {
		return nil, err
	}

	retEvent := &clientv1.Event{}
	if err := scheme.Convert(ret, retEvent, nil); err != nil {
		return nil, err
	}
	return retEvent, nil
}

func (fes *FederatedEventSink) Patch(event *clientv1.Event, data []byte) (*clientv1.Event, error) {
	kubeEvent := &kubev1.Event{}
	if err := scheme.Convert(event, kubeEvent, nil); err != nil {
		return nil, err
	}

	ret, err := fes.clientset.Core().Events(kubeEvent.Namespace).Patch(kubeEvent.Name, types.StrategicMergePatchType, data)
	if err != nil {
		return nil, err
	}

	retEvent := &clientv1.Event{}
	if err := scheme.Convert(ret, retEvent, nil); err != nil {
		return nil, err
	}
	return retEvent, nil
}
