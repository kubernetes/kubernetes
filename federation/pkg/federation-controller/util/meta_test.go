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

func TestObjectMeta(t *testing.T) {
	o1 := api_v1.ObjectMeta{
		Namespace:       "ns1",
		Name:            "s1",
		UID:             "1231231412",
		ResourceVersion: "999",
	}
	o2 := CopyObjectMeta(o1)
	o3 := api_v1.ObjectMeta{
		Namespace:   "ns1",
		Name:        "s1",
		UID:         "1231231412",
		Annotations: map[string]string{"A": "B"},
	}
	o4 := api_v1.ObjectMeta{
		Namespace:   "ns1",
		Name:        "s1",
		UID:         "1231255531412",
		Annotations: map[string]string{"A": "B"},
	}
	o5 := api_v1.ObjectMeta{
		Namespace:       "ns1",
		Name:            "s1",
		ResourceVersion: "1231231412",
		Annotations:     map[string]string{"A": "B"},
	}
	o6 := api_v1.ObjectMeta{
		Namespace:       "ns1",
		Name:            "s1",
		ResourceVersion: "1231255531412",
		Annotations:     map[string]string{"A": "B"},
	}
	assert.Equal(t, 0, len(o2.UID))
	assert.Equal(t, 0, len(o2.ResourceVersion))
	assert.Equal(t, o1.Name, o2.Name)
	assert.True(t, ObjectMetaEquivalent(o1, o2))
	assert.False(t, ObjectMetaEquivalent(o1, o3))
	assert.True(t, ObjectMetaEquivalent(o3, o4))
	assert.True(t, ObjectMetaEquivalent(o5, o6))
	assert.True(t, ObjectMetaEquivalent(o3, o5))
}
