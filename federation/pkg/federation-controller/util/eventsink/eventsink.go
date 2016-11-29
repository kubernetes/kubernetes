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
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5"
	api "k8s.io/kubernetes/pkg/api"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/record"
)

// Implemnts k8s.io/kubernetes/pkg/client/record.EventSink.
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

func (fes *FederatedEventSink) Create(event *api_v1.Event) (*api_v1.Event, error) {
	return fes.clientset.Core().Events(event.Namespace).Create(event)
}

func (fes *FederatedEventSink) Update(event *api_v1.Event) (*api_v1.Event, error) {
	return fes.clientset.Core().Events(event.Namespace).Update(event)
}

func (fes *FederatedEventSink) Patch(event *api_v1.Event, data []byte) (*api_v1.Event, error) {
	return fes.clientset.Core().Events(event.Namespace).Patch(event.Name, api.StrategicMergePatchType, data)
}
