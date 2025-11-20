/*
Copyright 2015 The Kubernetes Authors.

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

package cache

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
)

func TestNewIdentifierDuplicate(t *testing.T) {
	resetIdentity()
	aPod := NewIdentifier("a", &v1.Pod{})
	aPod1 := NewIdentifier("a", &v1.Pod{})
	aPod2 := NewIdentifier("a", &v1.Pod{})
	aNode := NewIdentifier("a", &v1.Node{})

	assert.Equal(t, "a", aPod.Name())
	assert.Equal(t, "v1.Pod", aPod.ItemType())
	assert.True(t, aPod.IsUnique())

	assert.Equal(t, "a-1", aPod1.Name())
	assert.Equal(t, "v1.Pod", aPod1.ItemType())
	assert.True(t, aPod1.IsUnique())

	assert.Equal(t, "a-2", aPod2.Name())
	assert.Equal(t, "v1.Pod", aPod2.ItemType())
	assert.True(t, aPod2.IsUnique())

	assert.Equal(t, "a", aNode.Name())
	assert.Equal(t, "v1.Node", aNode.ItemType())
	assert.True(t, aNode.IsUnique())
}

func TestNewIdentifierEMpty(t *testing.T) {
	resetIdentity()
	empty := NewIdentifier("", &v1.Pod{})
	assert.Equal(t, "", empty.Name())
	assert.Equal(t, "v1.Pod", empty.ItemType())
	assert.False(t, empty.IsUnique())
}

func TestNewIdentifierDash(t *testing.T) {
	resetIdentity()
	a := NewIdentifier("a", &v1.Pod{})
	aDashOne := NewIdentifier("a-1", &v1.Pod{})
	aDuplicate := NewIdentifier("a", &v1.Pod{})

	assert.Equal(t, "a", a.Name())
	assert.Equal(t, "v1.Pod", a.ItemType())
	assert.True(t, a.IsUnique())

	assert.Equal(t, "a-1", aDashOne.Name())
	assert.Equal(t, "v1.Pod", aDashOne.ItemType())
	assert.True(t, aDashOne.IsUnique())

	assert.Equal(t, "a-1", aDuplicate.Name())
	assert.Equal(t, "v1.Pod", aDuplicate.ItemType())
	assert.False(t, aDuplicate.IsUnique())
}

func TestNewIdentifierWithItemType(t *testing.T) {
	resetIdentity()
	aPod := NewIdentifier("a", &v1.Pod{})

	assert.Equal(t, "a", aPod.Name())
	assert.Equal(t, "v1.Pod", aPod.ItemType())
	assert.True(t, aPod.IsUnique())

	assert.Equal(t, aPod, aPod.WithObjectType(&v1.Pod{}))
	aNode := aPod.WithObjectType(&v1.Node{})

	assert.Equal(t, "a", aNode.Name())
	assert.Equal(t, "v1.Node", aNode.ItemType())
	assert.True(t, aNode.IsUnique())

	assert.Equal(t, "a", aPod.Name())
	assert.Equal(t, "v1.Pod", aPod.ItemType())
	assert.True(t, aPod.IsUnique())
}
