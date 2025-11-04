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
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-base/metrics/legacyregistry"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestFSPullRecordsMetrics(t *testing.T) {
	tempDir := t.TempDir()
	legacyregistry.Reset()
	defer legacyregistry.Reset()

	fsAccessor, err := NewFSPullRecordsAccessor(tempDir)
	if err != nil {
		t.Fatal(err)
	}

	const pullIntentsCacheSize, pulledRecordsCacheSize, stripedSetLocksSize int32 = 5, 10, 1
	inMemoryAccessor := NewCachedPullRecordsAccessor(fsAccessor, pullIntentsCacheSize, pulledRecordsCacheSize, stripedSetLocksSize)

	cmpFSIntents(t, 0)
	cmpMemIntents(t, 0)
	require.NoError(t, inMemoryAccessor.WriteImagePullIntent("test-image:latest"))
	cmpFSIntents(t, 1)
	cmpMemIntents(t, 20)

	// Test that writing the same record does not increase the count
	require.NoError(t, inMemoryAccessor.WriteImagePullIntent("test-image:latest"))
	require.NoError(t, inMemoryAccessor.WriteImagePullIntent("test-image:latest"))
	cmpFSIntents(t, 1)
	cmpMemIntents(t, 20)

	// Test adding more records
	require.NoError(t, inMemoryAccessor.WriteImagePullIntent("test-image:v1"))
	require.NoError(t, inMemoryAccessor.WriteImagePullIntent("test-image:v1.1"))
	cmpFSIntents(t, 3)
	cmpMemIntents(t, 60)

	cmpFSPulledRecords(t, 0)
	cmpMemPulledRecords(t, 0)
	require.NoError(t, inMemoryAccessor.WriteImagePulledRecord(&kubeletconfig.ImagePulledRecord{
		ImageRef:        "test-image-latest-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	cmpFSPulledRecords(t, 1)
	cmpMemPulledRecords(t, 10)

	// Test that writing the same record does not increase the count
	require.NoError(t, inMemoryAccessor.WriteImagePulledRecord(&kubeletconfig.ImagePulledRecord{
		ImageRef:        "test-image-latest-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	require.NoError(t, inMemoryAccessor.WriteImagePulledRecord(&kubeletconfig.ImagePulledRecord{
		ImageRef:        "test-image-latest-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	cmpFSPulledRecords(t, 1)
	cmpMemPulledRecords(t, 10)

	// Test adding more records
	require.NoError(t, inMemoryAccessor.WriteImagePulledRecord(&kubeletconfig.ImagePulledRecord{
		ImageRef:        "test-image-v1-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	require.NoError(t, inMemoryAccessor.WriteImagePulledRecord(&kubeletconfig.ImagePulledRecord{
		ImageRef:        "test-image-v1.1-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	require.NoError(t, inMemoryAccessor.WriteImagePulledRecord(&kubeletconfig.ImagePulledRecord{
		ImageRef:        "test-image-v1.2-ref",
		LastUpdatedTime: metav1.NewTime(time.Now()),
	}))
	cmpFSPulledRecords(t, 4)
	cmpMemPulledRecords(t, 40)

	// double-check that intents count is not affected
	cmpFSIntents(t, 3)
	cmpMemIntents(t, 60)

	// Test deletions
	require.NoError(t, inMemoryAccessor.DeleteImagePullIntent("test-image:latest"))
	cmpFSIntents(t, 2)
	cmpMemIntents(t, 40)

	require.NoError(t, inMemoryAccessor.DeleteImagePullIntent("test-image:latest"))
	require.NoError(t, inMemoryAccessor.DeleteImagePullIntent("test-image:latest"))
	cmpFSIntents(t, 2)
	cmpMemIntents(t, 40)

	require.NoError(t, inMemoryAccessor.DeleteImagePullIntent("test-image:v1"))
	require.NoError(t, inMemoryAccessor.DeleteImagePullIntent("test-image:v1.1"))
	cmpFSIntents(t, 0)
	cmpMemIntents(t, 0)

	// double-check that pulled records count is not affected
	cmpFSPulledRecords(t, 4)
	cmpMemPulledRecords(t, 40)

	// Test image pulled record deletions
	require.NoError(t, inMemoryAccessor.DeleteImagePulledRecord("test-image-v1.1-ref"))
	cmpFSPulledRecords(t, 3)
	cmpMemPulledRecords(t, 30)

	require.NoError(t, inMemoryAccessor.DeleteImagePulledRecord("test-image-v1.1-ref"))
	require.NoError(t, inMemoryAccessor.DeleteImagePulledRecord("test-image-v1.1-ref"))
	cmpFSPulledRecords(t, 3)
	cmpMemPulledRecords(t, 30)

	require.NoError(t, inMemoryAccessor.DeleteImagePulledRecord("test-image-v1.2-ref"))
	require.NoError(t, inMemoryAccessor.DeleteImagePulledRecord("test-image-v1-ref"))
	require.NoError(t, inMemoryAccessor.DeleteImagePulledRecord("test-image-latest-ref"))
	cmpFSPulledRecords(t, 0)
	cmpMemPulledRecords(t, 0)

	require.NoError(t, inMemoryAccessor.DeleteImagePulledRecord("test-image-v1-ref"))
	require.NoError(t, inMemoryAccessor.DeleteImagePulledRecord("test-image-latest-ref"))
	cmpFSPulledRecords(t, 0)
	cmpMemPulledRecords(t, 0)

	// test exceeding memory cache sizes
	for i := range 20 {
		require.NoError(t, inMemoryAccessor.WriteImagePullIntent(fmt.Sprintf("test-image-%d:latest", i)))
		require.NoError(t, inMemoryAccessor.WriteImagePulledRecord(&kubeletconfig.ImagePulledRecord{
			ImageRef:        fmt.Sprintf("test-image-v%d-ref", i),
			LastUpdatedTime: metav1.NewTime(time.Now()),
		}))
	}
	cmpFSIntents(t, 20)
	cmpFSPulledRecords(t, 20)
	cmpMemIntents(t, 100)
	cmpMemPulledRecords(t, 100)

	// test removing some of the latest records from the cache
	require.NoError(t, inMemoryAccessor.DeleteImagePullIntent("test-image-19:latest"))
	cmpFSIntents(t, 19)
	cmpMemIntents(t, 80)

	require.NoError(t, inMemoryAccessor.DeleteImagePulledRecord("test-image-v19-ref"))
	cmpFSPulledRecords(t, 19)
	cmpMemPulledRecords(t, 90)

}

func TestMustAttemptPullMetrics(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	tempDir := t.TempDir()
	pulledDir := filepath.Join(tempDir, "image_manager", "pulled")
	legacyregistry.Reset()
	defer legacyregistry.Reset()

	fsAccessor, err := NewFSPullRecordsAccessor(tempDir)
	if err != nil {
		t.Fatal(err)
	}

	fakeRuntime := &containertest.FakeRuntime{}
	imageManager, err := NewImagePullManager(ctx,
		fsAccessor,
		&NeverVerifyAllowlistedImages{absoluteURLs: sets.New("docker.io/testing/policyexempt")},
		fakeRuntime,
		1,
	)
	require.NoError(t, err)

	copyTestData(t, pulledDir, "pulled", []string{
		"sha256-e766e4624f9bc4d3847d6c5b470d9d5362cc8d0c250bd9572daf95278e044263",
		"sha256-a2eace2182b24cdbbb730798e47b10709b9ef5e0f0c1624a3bc06c8ca987727a",
		"sha256-f4058727984875eb66ddbf289f7013096d4cbaa167e591fbcb2e447e36391c1f",
		"sha256-b3c0cc4278800b03a308ceb2611161430df571ca733122f0a40ac8b9792a9064",
	})

	expectedMetrics := make(map[string]int)
	mustPull, err := imageManager.MustAttemptImagePull(ctx, "docker.io/testing/broken", "testbrokenrecord", nil)
	require.NoError(t, err)
	require.True(t, mustPull)
	expectedMetrics[string(checkResultError)]++
	cmpMustAttemptPullMetrics(t, expectedMetrics)

	mustPull, err = imageManager.MustAttemptImagePull(ctx, "docker.io/testing/broken", "testbrokenrecord", nil)
	require.NoError(t, err)
	require.True(t, mustPull)
	mustPull, err = imageManager.MustAttemptImagePull(ctx, "docker.io/testing/broken", "testbrokenrecord", nil)
	require.NoError(t, err)
	require.True(t, mustPull)
	expectedMetrics[string(checkResultError)] += 2

	mustPull, err = imageManager.MustAttemptImagePull(ctx, "docker.io/testing/test", "testimage-anonpull", nil)
	require.NoError(t, err)
	require.False(t, mustPull)
	expectedMetrics[string(checkResultCredentialRecordFound)]++
	cmpMustAttemptPullMetrics(t, expectedMetrics)

	mustPull, err = imageManager.MustAttemptImagePull(ctx, "docker.io/testing/test", "testimage-anonpull", nil)
	require.NoError(t, err)
	require.False(t, mustPull)
	mustPull, err = imageManager.MustAttemptImagePull(ctx, "docker.io/testing/test", "testimage-anonpull", nil)
	require.NoError(t, err)
	require.False(t, mustPull)
	mustPull, err = imageManager.MustAttemptImagePull(ctx, "docker.io/testing/test", "testimage-anonpull", nil)
	require.NoError(t, err)
	require.False(t, mustPull)
	expectedMetrics[string(checkResultCredentialRecordFound)] += 3
	cmpMustAttemptPullMetrics(t, expectedMetrics)

	mustPull, err = imageManager.MustAttemptImagePull(ctx, "docker.io/testing/policyexempt", "policyallowed", nil)
	require.NoError(t, err)
	require.False(t, mustPull)
	expectedMetrics[string(checkResultCredentialPolicyAllowed)]++
	cmpMustAttemptPullMetrics(t, expectedMetrics)

	mustPull, err = imageManager.MustAttemptImagePull(ctx, "docker.io/testing/policyexempt", "policyallowed", nil)
	require.NoError(t, err)
	require.False(t, mustPull)
	expectedMetrics[string(checkResultCredentialPolicyAllowed)]++
	cmpMustAttemptPullMetrics(t, expectedMetrics)

	mustPull, err = imageManager.MustAttemptImagePull(ctx, "docker.io/testing/norecords", "somewhatunknown", nil)
	require.NoError(t, err)
	require.True(t, mustPull)
	expectedMetrics[string(checkResultMustAuthenticate)]++
	cmpMustAttemptPullMetrics(t, expectedMetrics)

	mustPull, err = imageManager.MustAttemptImagePull(ctx, "docker.io/testing/test", "testimageref",
		func() ([]kubeletconfig.ImagePullSecret, *kubeletconfig.ImagePullServiceAccount, error) {
			return []kubeletconfig.ImagePullSecret{
				{UID: "testsecretuid", Namespace: "default", Name: "pull-secret", CredentialHash: "testsecrethash"},
			}, nil, nil
		})
	require.NoError(t, err)
	require.False(t, mustPull)
	expectedMetrics[string(checkResultCredentialRecordFound)]++
	cmpMustAttemptPullMetrics(t, expectedMetrics)
}

func cmpFSIntents(t *testing.T, expected uint) {
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
func cmpFSPulledRecords(t *testing.T, expected uint) {
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

func cmpMemIntents(t *testing.T, expected uint) {
	t.Helper()
	const metricFormat = `
# HELP kubelet_imagemanager_inmemory_pullintents_usage_percent [ALPHA] The ImagePullIntents in-memory cache usage in percent.
# TYPE kubelet_imagemanager_inmemory_pullintents_usage_percent gauge
kubelet_imagemanager_inmemory_pullintents_usage_percent %d
`

	err := metricstestutil.GatherAndCompare(
		legacyregistry.DefaultGatherer, strings.NewReader(fmt.Sprintf(metricFormat, expected)), "kubelet_imagemanager_inmemory_pullintents_usage_percent",
	)
	if err != nil {
		t.Errorf("failed to gather metrics: %v", err)
	}
}
func cmpMemPulledRecords(t *testing.T, expected uint) {
	t.Helper()
	const metricFormat = `
# HELP kubelet_imagemanager_inmemory_pulledrecords_usage_percent [ALPHA] The ImagePulledRecords in-memory cache usage in percent.
# TYPE kubelet_imagemanager_inmemory_pulledrecords_usage_percent gauge
kubelet_imagemanager_inmemory_pulledrecords_usage_percent %d
`

	err := metricstestutil.GatherAndCompare(
		legacyregistry.DefaultGatherer, strings.NewReader(fmt.Sprintf(metricFormat, expected)), "kubelet_imagemanager_inmemory_pulledrecords_usage_percent",
	)
	if err != nil {
		t.Errorf("failed to gather metrics: %v", err)
	}
}

func cmpMustAttemptPullMetrics(t *testing.T, labelMap map[string]int) {
	t.Helper()
	const metricFormat = `
# HELP kubelet_imagemanager_image_mustpull_checks_total [ALPHA] Counter for how many times kubelet checked whether credentials need to be re-verified to access an image
# TYPE kubelet_imagemanager_image_mustpull_checks_total counter
`
	expected := metricFormat
	for label, val := range labelMap {
		expected += fmt.Sprintf("kubelet_imagemanager_image_mustpull_checks_total{result=\"%s\"} %d\n", label, val)
	}

	err := metricstestutil.GatherAndCompare(
		legacyregistry.DefaultGatherer, strings.NewReader(expected), "kubelet_imagemanager_image_mustpull_checks_total",
	)
	if err != nil {
		t.Errorf("failed to gather metrics: %v", err)
	}
}
