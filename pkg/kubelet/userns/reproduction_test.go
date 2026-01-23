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

package userns

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestMakeUserNsManagerWithMismatchingPodLength(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.UserNamespacesSupport, true)

	podUID := types.UID("pod-mismatch")
	podDir := t.TempDir()

	// Create a mappings file with length 131072
	usernsDir := podDir // The manager uses GetPodDir(podUID) as the base
	err := os.MkdirAll(usernsDir, 0755)
	require.NoError(t, err)

	content := `{
		"uidMappings":[ { "hostId":131072, "containerId":0, "length":131072 } ],
		"gidMappings":[ { "hostId":131072, "containerId":0, "length":131072 } ]
	}`
	err = os.WriteFile(filepath.Join(usernsDir, mappingsFile), []byte(content), 0644)
	require.NoError(t, err)

	testUserNsPodsManager := &testUserNsPodsManager{
		podDir:  podDir,
		podList: []types.UID{podUID},
	}

	// Initialize manager with default length (65536)
	idsPerPod := int64(65536)
	_, err = MakeUserNsManager(logger, testUserNsPodsManager, &idsPerPod)

	// Now, this should SUCCEED
	_, err = MakeUserNsManager(logger, testUserNsPodsManager, &idsPerPod)
	assert.NoError(t, err, "Expected no error after fix")
}
