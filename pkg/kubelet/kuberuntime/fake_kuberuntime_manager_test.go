/*
Copyright 2021 The Kubernetes Authors.

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

package kuberuntime

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
)

func TestGet(t *testing.T) {
	fakeUrl := "fakeUrl.com"
	fakeError := fmt.Errorf("fake error")

	fake := &fakeHTTP{
		url: "",
		err: fakeError,
	}

	resultFake := &fakeHTTP{
		url: fakeUrl,
		err: fakeError,
	}

	response, err := fake.Get(fakeUrl)
	assert.Nil(t, response)
	assert.Equal(t, err, fakeError)
	assert.Equal(t, fake, resultFake)
}

func TestIsPodDeleted(t *testing.T) {
	var podId1 types.UID = "id1"
	var podId2 types.UID = "id2"

	fakeProvider := &fakePodStateProvider{
		existingPods: map[types.UID]struct{}{podId1: {}},
	}

	assert.False(t, fakeProvider.IsPodDeleted(podId1))
	assert.True(t, fakeProvider.IsPodDeleted(podId2))
}

func TestIsPodTerminated(t *testing.T) {
	var podId1 types.UID = "id1"
	var podId2 types.UID = "id2"

	fakeProvider := &fakePodStateProvider{
		runningPods: map[types.UID]struct{}{podId1: {}},
	}

	assert.False(t, fakeProvider.IsPodTerminated(podId1))
	assert.True(t, fakeProvider.IsPodTerminated(podId2))
}
