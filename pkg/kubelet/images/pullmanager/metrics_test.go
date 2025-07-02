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

package pullmanager

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-base/metrics/legacyregistry"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/kubelet/apis/config"
)

func TestFSPullRecordsMetrics(t *testing.T) {
	tempDir := t.TempDir()
	legacyregistry.Reset()
	defer legacyregistry.Reset()

	fsAccessor, err := NewFSPullRecordsAccessor(tempDir)
	if err != nil {
		t.Fatal(err)
	}

	cmpIntents(t, 0)
	require.NoError(t, fsAccessor.WriteImagePullIntent("test-image:latest"))
	cmpIntents(t, 1)

	// Test that writing the same record does not increase the count
	require.NoError(t, fsAccessor.WriteImagePullIntent("test-image:latest"))
	require.NoError(t, fsAccessor.WriteImagePullIntent("test-image:latest"))
	cmpIntents(t, 1)

	// Test adding more records
	require.NoError(t, fsAccessor.WriteImagePullIntent("test-image:v1"))
	require.NoError(t, fsAccessor.WriteImagePullIntent("test-image:v1.1"))
	cmpIntents(t, 3)

	cmpPulledRecords(t, 0)
	require.NoError(t, fsAccessor.WriteImagePulledRecord(&config.ImagePulledRecord{
		ImageRef:        "test-image-latest-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	cmpPulledRecords(t, 1)

	// Test that writing the same record does not increase the count
	require.NoError(t, fsAccessor.WriteImagePulledRecord(&config.ImagePulledRecord{
		ImageRef:        "test-image-latest-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	require.NoError(t, fsAccessor.WriteImagePulledRecord(&config.ImagePulledRecord{
		ImageRef:        "test-image-latest-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	cmpPulledRecords(t, 1)

	// Test adding more records
	require.NoError(t, fsAccessor.WriteImagePulledRecord(&config.ImagePulledRecord{
		ImageRef:        "test-image-v1-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	require.NoError(t, fsAccessor.WriteImagePulledRecord(&config.ImagePulledRecord{
		ImageRef:        "test-image-v1.1-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	require.NoError(t, fsAccessor.WriteImagePulledRecord(&config.ImagePulledRecord{
		ImageRef:        "test-image-v1.2-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	cmpPulledRecords(t, 4)

	cmpIntents(t, 3) // double-check that intents count is not affected

	// Test deletions
	require.NoError(t, fsAccessor.DeleteImagePullIntent("test-image:latest"))
	cmpIntents(t, 2)

	require.NoError(t, fsAccessor.DeleteImagePullIntent("test-image:latest"))
	require.NoError(t, fsAccessor.DeleteImagePullIntent("test-image:latest"))
	cmpIntents(t, 2)

	require.NoError(t, fsAccessor.DeleteImagePullIntent("test-image:v1"))
	require.NoError(t, fsAccessor.DeleteImagePullIntent("test-image:v1.1"))
	cmpIntents(t, 0)

	cmpPulledRecords(t, 4) // double-check that pulled records count is not affected

	// Test image pulled record deletions
	require.NoError(t, fsAccessor.DeleteImagePulledRecord("test-image-v1.1-ref"))
	cmpPulledRecords(t, 3)

	require.NoError(t, fsAccessor.DeleteImagePulledRecord("test-image-v1.1-ref"))
	require.NoError(t, fsAccessor.DeleteImagePulledRecord("test-image-v1.1-ref"))
	cmpPulledRecords(t, 3)

	require.NoError(t, fsAccessor.DeleteImagePulledRecord("test-image-v1.2-ref"))
	require.NoError(t, fsAccessor.DeleteImagePulledRecord("test-image-v1-ref"))
	require.NoError(t, fsAccessor.DeleteImagePulledRecord("test-image-latest-ref"))
	cmpPulledRecords(t, 0)

	require.NoError(t, fsAccessor.DeleteImagePulledRecord("test-image-v1-ref"))
	require.NoError(t, fsAccessor.DeleteImagePulledRecord("test-image-latest-ref"))
	cmpPulledRecords(t, 0)
}

func cmpIntents(t *testing.T, expected uint) {
	t.Helper()
	const metricFormat = `
# HELP kubelet_imagemanager_ondisk_pullintents [ALPHA] Number of ImagePullIntents stored on disk.
# TYPE kubelet_imagemanager_ondisk_pullintents gauge
kubelet_imagemanager_ondisk_pullintents %d
`

	err := metricstestutil.GatherAndCompare(
		legacyregistry.DefaultGatherer, strings.NewReader(fmt.Sprintf(metricFormat, expected)), "kubelet_imagemanager_ondisk_pullintents",
	)
	if err != nil {
		t.Errorf("failed to gather metrics: %v", err)
	}
}
func cmpPulledRecords(t *testing.T, expected uint) {
	t.Helper()
	const metricFormat = `
# HELP kubelet_imagemanager_ondisk_pulledrecords [ALPHA] Number of ImagePulledRecords stored on disk.
# TYPE kubelet_imagemanager_ondisk_pulledrecords gauge
kubelet_imagemanager_ondisk_pulledrecords %d
`

	err := metricstestutil.GatherAndCompare(
		legacyregistry.DefaultGatherer, strings.NewReader(fmt.Sprintf(metricFormat, expected)), "kubelet_imagemanager_ondisk_pulledrecords",
	)
	if err != nil {
		t.Errorf("failed to gather metrics: %v", err)
	}
}
