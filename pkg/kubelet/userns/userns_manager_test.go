/*
Copyright 2022 The Kubernetes Authors.

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

package userns

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
)

type testUserNsPodsManager struct {
}

func (m *testUserNsPodsManager) GetPodDir(podUID types.UID) string {
	return "/tmp/non-existant-dir.This-is-not-used-in-tests"
}

func (m *testUserNsPodsManager) ListPodsFromDisk() ([]types.UID, error) {
	return nil, nil
}

func TestUserNsManagerAllocate(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesStatelessPodsSupport, true)()

	testUserNsPodsManager := &testUserNsPodsManager{}
	m, err := MakeUserNsManager(testUserNsPodsManager)
	require.NoError(t, err)

	assert.Equal(t, true, m.isSet(0*65536), "m.isSet(0) should be true")

	allocated, length, err := m.allocateOne("one")
	assert.NoError(t, err)
	assert.Equal(t, userNsLength, int(length), "m.isSet(%d).length=%v", allocated, length)
	assert.Equal(t, true, m.isSet(allocated), "m.isSet(%d)", allocated)

	allocated2, length2, err := m.allocateOne("two")
	assert.NoError(t, err)
	assert.NotEqual(t, allocated, allocated2, "allocated != allocated2")
	assert.Equal(t, length, length2, "length == length2")

	// verify that re-adding the same pod with the same settings won't fail
	err = m.record("two", allocated2, length2)
	assert.NoError(t, err)
	// but it fails if anyting is different
	err = m.record("two", allocated2+1, length2)
	assert.Error(t, err)

	m.Release("one")
	m.Release("two")
	assert.Equal(t, false, m.isSet(allocated), "m.isSet(%d)", allocated)
	assert.Equal(t, false, m.isSet(allocated2), "m.nsSet(%d)", allocated2)

	var allocs []uint32
	for i := 0; i < 1000; i++ {
		allocated, length, err = m.allocateOne(types.UID(fmt.Sprintf("%d", i)))
		assert.Equal(t, userNsLength, int(length), "length is not the expected. iter: %v", i)
		assert.NoError(t, err)
		allocs = append(allocs, allocated)
	}
	for i, v := range allocs {
		assert.Equal(t, true, m.isSet(v), "m.isSet(%d) should be true", v)
		m.Release(types.UID(fmt.Sprintf("%d", i)))
		assert.Equal(t, false, m.isSet(v), "m.isSet(%d) should be false", v)

		err = m.record(types.UID(fmt.Sprintf("%d", i)), v, userNsLength)
		assert.NoError(t, err)
		m.Release(types.UID(fmt.Sprintf("%d", i)))
		assert.Equal(t, false, m.isSet(v), "m.isSet(%d) should be false", v)
	}
}

func TestUserNsManagerParseUserNsFile(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesStatelessPodsSupport, true)()

	cases := []struct {
		name    string
		file    string
		success bool
	}{
		{
			name: "basic",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":0, "length":65536 } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":0, "length":65536 } ]
                               }`,
			success: true,
		},
		{
			name: "invalid length",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":0, "length":0 } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":0, "length":0 } ]
                               }`,
			success: false,
		},
		{
			name: "wrong offset",
			file: `{
	                        "uidMappings":[ {"hostId":131072, "containerId":0, "length":65536 } ],
	                        "gidMappings":[ {"hostId":1, "containerId":0, "length":65536 } ]
                               }`,
			success: false,
		},
		{
			name: "two GID mappings",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":0, "length":userNsLength } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":0, "length":userNsLength }, { "hostId":196608, "containerId":0, "length":65536 } ]
                               }`,
			success: false,
		},
		{
			name: "two UID mappings",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":0, "length":65536 }, { "hostId":196608, "containerId":0, "length":65536 } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":0, "length":65536 } ]
                               }`,
			success: false,
		},
		{
			name: "no root UID",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":1, "length":65536 } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":0, "length":65536 } ]
                               }`,
			success: false,
		},
		{
			name: "no root GID",
			file: `{
	                        "uidMappings":[ { "hostId":131072, "containerId":0, "length":65536 } ],
	                        "gidMappings":[ { "hostId":131072, "containerId":1, "length":65536 } ]
                               }`,
			success: false,
		},
	}

	testUserNsPodsManager := &testUserNsPodsManager{}
	m, err := MakeUserNsManager(testUserNsPodsManager)
	assert.NoError(t, err)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// We don't validate the result. It was parsed with the json parser, we trust that.
			_, err = m.parseUserNsFileAndRecord(types.UID(tc.name), []byte(tc.file))
			if (tc.success && err == nil) || (!tc.success && err != nil) {
				return
			}

			t.Errorf("expected success: %v but got error: %v", tc.success, err)
		})
	}
}
