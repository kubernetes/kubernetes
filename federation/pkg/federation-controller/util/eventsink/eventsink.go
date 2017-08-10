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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
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
	scheme.AddKnownTypes(v1.SchemeGroupVersion, &v1.Event{})
	scheme.AddKnownTypes(schema.GroupVersion{Group: "fake-kube-" + v1.SchemeGroupVersion.Group, Version: v1.SchemeGroupVersion.Version}, &v1.Event{})

	if err := scheme.AddConversionFuncs(
		metav1.Convert_unversioned_Time_To_unversioned_Time,
	); err != nil {
		panic(err)
	}
	if err := scheme.AddGeneratedDeepCopyFuncs(
		conversion.GeneratedDeepCopyFunc{
			Fn: func(in, out interface{}, c *conversion.Cloner) error {
				in.(*metav1.Time).DeepCopyInto(out.(*metav1.Time))
				return nil
			},
			InType: reflect.TypeOf(&metav1.Time{}),
		},
	); err != nil {
		panic(err)
	}
}

func (fes *FederatedEventSink) Create(event *v1.Event) (*v1.Event, error) {
	ret, err := fes.clientset.Core().Events(event.Namespace).Create(event)
	if err != nil {
		return nil, err
	}

	retEvent := &v1.Event{}
	if err := scheme.Convert(ret, retEvent, nil); err != nil {
		return nil, err
	}
	return retEvent, nil
}

func (fes *FederatedEventSink) Update(event *v1.Event) (*v1.Event, error) {
	ret, err := fes.clientset.Core().Events(event.Namespace).Update(event)
	if err != nil {
		return nil, err
	}

	retEvent := &v1.Event{}
	if err := scheme.Convert(ret, retEvent, nil); err != nil {
		return nil, err
	}
	return retEvent, nil
}

func (fes *FederatedEventSink) Patch(event *v1.Event, data []byte) (*v1.Event, error) {
	ret, err := fes.clientset.Core().Events(event.Namespace).Patch(event.Name, types.StrategicMergePatchType, data)
	if err != nil {
		return nil, err
	}

	retEvent := &v1.Event{}
	if err := scheme.Convert(ret, retEvent, nil); err != nil {
		return nil, err
	}
	return retEvent, nil
}
