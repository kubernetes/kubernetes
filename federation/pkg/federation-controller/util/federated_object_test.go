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
	"testing"

	api_v1 "k8s.io/kubernetes/pkg/api/v1"

	"github.com/stretchr/testify/assert"
)

func TestGetClusterName(t *testing.T) {
	// There is a single service ns1/s1 in cluster mycluster.
	service := api_v1.Service{
		ObjectMeta: api_v1.ObjectMeta{
			Namespace: "ns1",
			Name:      "s1",
			Annotations: map[string]string{
				ClusterNameAnnotation: "mycluster",
			},
		},
	}
	name, err := GetClusterName(&service)
	assert.NoError(t, err)
	assert.Equal(t, "mycluster", name)
}

func TestSetClusterName(t *testing.T) {
	// There is a single service ns1/s1 in cluster mycluster.
	service := api_v1.Service{
		ObjectMeta: api_v1.ObjectMeta{
			Namespace: "ns1",
			Name:      "s1",
		},
	}
	err := SetClusterName(&service, "mytestname")
	assert.NoError(t, err)
	clusterName := service.Annotations[ClusterNameAnnotation]
	assert.Equal(t, "mytestname", clusterName)
}
