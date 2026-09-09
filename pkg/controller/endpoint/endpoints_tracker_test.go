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

package endpoint

import (
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestStaleEndpointsTracker(t *testing.T) {
	ns := metav1.NamespaceDefault
	tracker := newStaleEndpointsTracker()

	endpoints := &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{IP: "6.7.8.9", NodeName: &emptyNodeName}},
			Ports:     []v1.EndpointPort{{Port: 1000}},
		}},
	}

	assert.False(t, tracker.IsStale(endpoints), "IsStale should return false before the endpoint is staled")

	tracker.Stale(endpoints)
	assert.True(t, tracker.IsStale(endpoints), "IsStale should return true after the endpoint is staled")

	endpoints.ResourceVersion = "2"
	assert.False(t, tracker.IsStale(endpoints), "IsStale should return false after the endpoint is updated")

	tracker.Delete(endpoints.Namespace, endpoints.Name)
	assert.Empty(t, tracker.staleResourceVersionByEndpoints)
}
