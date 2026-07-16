/*
Copyright The Kubernetes Authors.

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

package rest

import (
	"context"
	"fmt"
	"io"
	"net/http"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubelet/client"
	registrypod "k8s.io/kubernetes/pkg/registry/core/pod"
)

// AllocatedREST implements the allocated subresource for a Pod
type AllocatedREST struct {
	Store       registrypod.ResourceGetter
	KubeletConn client.ConnectionInfoGetter
}

// Implement Getter
var _ rest.Getter = &AllocatedREST{}

// New creates a new Pod object.
func (r *AllocatedREST) New() runtime.Object {
	return &api.Pod{}
}

// Destroy cleans up resources on shutdown.
func (r *AllocatedREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// Get retrieves the allocated pod spec from Kubelet on-demand
func (r *AllocatedREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	// Get the pod's node location and connection info
	location, transport, err := registrypod.AllocatedLocation(ctx, r.Store, r.KubeletConn, name)
	if err != nil {
		return nil, err
	}

	// Perform HTTP GET request to Kubelet
	httpClient := &http.Client{Transport: transport}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, location.String(), nil)
	if err != nil {
		return nil, apierrors.NewInternalError(err)
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		// Connection failure -> 503 Service Unavailable
		return nil, apierrors.NewServiceUnavailable(fmt.Sprintf("failed to connect to Kubelet: %v", err))
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Error handling & Translation
		if resp.StatusCode == http.StatusNotFound {
			return nil, apierrors.NewNotFound(schema.GroupResource{Resource: "pods/allocated"}, name)
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024)) // Read small error message
		errMsg := fmt.Sprintf("Kubelet returned status %d: %s", resp.StatusCode, string(body))
		return nil, apierrors.NewGenericServerResponse(resp.StatusCode, "get", schema.GroupResource{Resource: "pods/allocated"}, name, errMsg, 0, true)
	}

	// Decode response into v1.Pod (versioned type)
	const maxResponseSize = 10 * 1024 * 1024 // 10 MB
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSize))
	if err != nil {
		return nil, apierrors.NewInternalError(err)
	}

	versionedPod := &v1.Pod{}
	obj, _, err := legacyscheme.Codecs.UniversalDecoder().Decode(body, nil, versionedPod)
	if err != nil {
		return nil, apierrors.NewInternalError(fmt.Errorf("failed to decode Kubelet response: %w", err))
	}
	v1Pod, ok := obj.(*v1.Pod)
	if !ok {
		return nil, apierrors.NewInternalError(fmt.Errorf("expected v1.Pod, got %T", obj))
	}

	// Convert v1.Pod to api.Pod (internal type)
	internalPod := &api.Pod{}
	if err := legacyscheme.Scheme.Convert(v1Pod, internalPod, nil); err != nil {
		return nil, apierrors.NewInternalError(fmt.Errorf("failed to convert versioned pod to internal: %w", err))
	}

	return internalPod, nil
}
