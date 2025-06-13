/*
Copyright 2025 The Kubernetes Authors.

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

package storage

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/storage/testresource"
)

func TestObjectSizeWithoutLabels(t *testing.T) {
	obj := &testresource.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}}
	accessor := meta.NewAccessor()
	err := SetObjectSizeLabel(accessor, obj, 42)
	require.NoError(t, err)
	assert.Equal(t, &testresource.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5", Labels: map[string]string{objectSizeLabel: "42"}}}, obj)
	size, err := ReadAndRemoveObjectSizeLabel(accessor, obj)
	require.NoError(t, err)
	assert.Equal(t, int64(42), size)
	assert.Equal(t, &testresource.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}}, obj)
}

func TestObjectSize(t *testing.T) {
	obj := &testresource.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5", Labels: map[string]string{"env": "prod"}}}
	accessor := meta.NewAccessor()
	err := SetObjectSizeLabel(accessor, obj, 42)
	require.NoError(t, err)
	assert.Equal(t, &testresource.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5", Labels: map[string]string{objectSizeLabel: "42", "env": "prod"}}}, obj)
	size, err := ReadAndRemoveObjectSizeLabel(accessor, obj)
	require.NoError(t, err)
	assert.Equal(t, int64(42), size)
	assert.Equal(t, &testresource.TestResource{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5", Labels: map[string]string{"env": "prod"}}}, obj)
}
