/*
Copyright 2024 The Kubernetes Authors.

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

package app

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/kubernetes/pkg/apis/core"
	v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	eventstorage "k8s.io/kubernetes/pkg/registry/core/event/storage"
)

// eventRegistrySink wraps an event registry in order to be used as direct event sync, without going through the API.
type eventRegistrySink struct {
	*eventstorage.REST
}

var _ genericapiserver.EventSink = eventRegistrySink{}

func (s eventRegistrySink) Create(v1event *corev1.Event) (*corev1.Event, error) {
	ctx := request.WithNamespace(request.WithRequestInfo(request.NewContext(), &request.RequestInfo{APIVersion: "v1"}), v1event.Namespace)
	// since we are bypassing the API set a hard timeout for the storage layer
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()

	var event core.Event
	if err := v1.Convert_v1_Event_To_core_Event(v1event, &event, nil); err != nil {
		return nil, err
	}

	obj, err := s.REST.Create(ctx, &event, nil, &metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}
	ret, ok := obj.(*core.Event)
	if !ok {
		return nil, fmt.Errorf("expected corev1.Event, got %T", obj)
	}

	var v1ret corev1.Event
	if err := v1.Convert_core_Event_To_v1_Event(ret, &v1ret, nil); err != nil {
		return nil, err
	}

	return &v1ret, nil
}
