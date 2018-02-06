/*
Copyright 2017 The Kubernetes Authors.

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

// Package mock encapsulates mocks for testing GCE provider functionality.
// These methods are used to override the mock objects' methods in order to
// intercept the standard processing and to add custom logic for test purposes.
//
//  // Example usage:
// cloud := cloud.NewMockGCE()
// cloud.MockTargetPools.AddInstanceHook = mock.AddInstanceHook

package mock

import (
	"context"
	"fmt"
	"net/http"

	ga "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	cloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func AddInstanceHook(m *cloud.MockTargetPools, ctx context.Context, key *meta.Key, req *ga.TargetPoolsAddInstanceRequest) error {
	pool, err := m.Get(ctx, key)
	if err != nil {
		return &googleapi.Error{
			Code:    http.StatusNotFound,
			Message: fmt.Sprintf("Key: %s was not found in TargetPools", key.String()),
		}
	}

	for _, instance := range req.Instances {
		pool.Instances = append(pool.Instances, instance.Instance)
	}

	return nil
}
