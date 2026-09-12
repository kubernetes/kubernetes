/*
Copyright The Kubernetes Authors.

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

package metrics

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/discovery"
	coretesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	metricsv1 "k8s.io/metrics/pkg/apis/metrics/v1"
	metricsv1beta1 "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

// fakeDiscovery serves a canned APIGroupList and counts calls.
type fakeDiscovery struct {
	discovery.DiscoveryInterface

	groups *metav1.APIGroupList
	err    error
	calls  int
}

func (d *fakeDiscovery) ServerGroups() (*metav1.APIGroupList, error) {
	d.calls++
	return d.groups, d.err
}

func (d *fakeDiscovery) ServerGroupsWithContext(ctx context.Context) (*metav1.APIGroupList, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return d.ServerGroups()
}

func discoveryFor(versions ...string) *fakeDiscovery {
	groupList := &metav1.APIGroupList{}
	if len(versions) > 0 {
		group := metav1.APIGroup{Name: "metrics.k8s.io"}
		for _, version := range versions {
			group.Versions = append(group.Versions, metav1.GroupVersionForDiscovery{
				GroupVersion: "metrics.k8s.io/" + version,
				Version:      version,
			})
		}
		// deliberately report the oldest version as the server-preferred one:
		// client-side preference must win over it
		group.PreferredVersion = group.Versions[len(group.Versions)-1]
		groupList.Groups = append(groupList.Groups, group)
	}
	return &fakeDiscovery{groups: groupList}
}

func v1PodMetrics(name string) *metricsv1.PodMetrics {
	return &metricsv1.PodMetrics{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "test-ns"},
		Timestamp:  metav1.Time{Time: fixedTimestamp},
		Window:     metav1.Duration{Duration: 30 * time.Second},
		Containers: []metricsv1.ContainerMetrics{{
			Name:  "app",
			Usage: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
		}},
	}
}

func v1beta1PodMetrics(name string) *metricsv1beta1.PodMetrics {
	return &metricsv1beta1.PodMetrics{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "test-ns"},
		Timestamp:  metav1.Time{Time: fixedTimestamp},
		Window:     metav1.Duration{Duration: 30 * time.Second},
		Containers: []metricsv1beta1.ContainerMetrics{{
			Name:  "app",
			Usage: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
		}},
	}
}

// newVersionedGetter backs the v1 and v1beta1 clients with separate fake
// clientsets serving differently-named objects, so tests can tell which API
// version actually served a request.
func newVersionedGetter(d *fakeDiscovery) VersionedPodMetricsGetter {
	v1Clientset := &metricsfake.Clientset{}
	v1Clientset.AddReactor("get", "pods", func(action coretesting.Action) (bool, runtime.Object, error) {
		return true, v1PodMetrics("served-by-v1"), nil
	})
	v1Clientset.AddReactor("list", "pods", func(action coretesting.Action) (bool, runtime.Object, error) {
		return true, &metricsv1.PodMetricsList{Items: []metricsv1.PodMetrics{*v1PodMetrics("served-by-v1")}}, nil
	})

	v1beta1Clientset := &metricsfake.Clientset{}
	v1beta1Clientset.AddReactor("get", "pods", func(action coretesting.Action) (bool, runtime.Object, error) {
		return true, v1beta1PodMetrics("served-by-v1beta1"), nil
	})
	v1beta1Clientset.AddReactor("list", "pods", func(action coretesting.Action) (bool, runtime.Object, error) {
		return true, &metricsv1beta1.PodMetricsList{Items: []metricsv1beta1.PodMetrics{*v1beta1PodMetrics("served-by-v1beta1")}}, nil
	})

	return NewVersionedPodMetricsGetter(v1Clientset.MetricsV1(), v1beta1Clientset.MetricsV1beta1(), d)
}

func TestVersionedPodMetricsGetterVersionSelection(t *testing.T) {
	testCases := []struct {
		name            string
		servedVersions  []string
		expectedPodName string
		expectedErr     string
	}{
		{
			name:            "prefers v1 when both are served",
			servedVersions:  []string{"v1", "v1beta1"},
			expectedPodName: "served-by-v1",
		},
		{
			name:            "prefers v1 even when the server prefers v1beta1",
			servedVersions:  []string{"v1beta1", "v1"},
			expectedPodName: "served-by-v1",
		},
		{
			name:            "falls back to v1beta1 when v1 is not served",
			servedVersions:  []string{"v1beta1"},
			expectedPodName: "served-by-v1beta1",
		},
		{
			name:            "uses v1 when it is the only version served",
			servedVersions:  []string{"v1"},
			expectedPodName: "served-by-v1",
		},
		{
			name:           "fails when the group is not registered",
			servedVersions: nil,
			expectedErr:    "no resource metrics API (metrics.k8s.io) registered",
		},
		{
			name:           "fails when only unknown versions are served",
			servedVersions: []string{"v1alpha1"},
			expectedErr:    "no known available resource metrics API versions found",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			getter := newVersionedGetter(discoveryFor(tc.servedVersions...))

			metrics, err := getter.PodMetricses("test-ns").List(tCtx, metav1.ListOptions{})
			if tc.expectedErr != "" {
				require.ErrorContains(t, err, tc.expectedErr)
				return
			}
			require.NoError(t, err)
			require.Len(t, metrics.Items, 1)
			assert.Equal(t, tc.expectedPodName, metrics.Items[0].Name)
		})
	}
}

func TestVersionedPodMetricsGetterConversion(t *testing.T) {
	tCtx := ktesting.Init(t)
	getter := newVersionedGetter(discoveryFor("v1"))

	metrics, err := getter.PodMetricses("test-ns").Get(tCtx, "served-by-v1", metav1.GetOptions{})
	require.NoError(t, err)

	// the v1 response must round-trip losslessly into the v1beta1 type
	expected := v1beta1PodMetrics("served-by-v1")
	assert.Equal(t, expected.ObjectMeta, metrics.ObjectMeta)
	assert.Equal(t, expected.Timestamp, metrics.Timestamp)
	assert.Equal(t, expected.Window, metrics.Window)
	assert.Equal(t, expected.Containers, metrics.Containers)
}

func TestVersionedPodMetricsGetterCachesAndInvalidates(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeDisc := discoveryFor("v1beta1")
	getter := newVersionedGetter(fakeDisc)
	podMetrics := getter.PodMetricses("test-ns")

	for range 3 {
		metrics, err := podMetrics.List(tCtx, metav1.ListOptions{})
		require.NoError(t, err)
		assert.Equal(t, "served-by-v1beta1", metrics.Items[0].Name)
	}
	assert.Equal(t, 1, fakeDisc.calls, "discovery should only be hit once while the cache is warm")

	// the metrics server got upgraded: it now serves v1 as well
	fakeDisc.groups = discoveryFor("v1", "v1beta1").groups
	getter.Invalidate()

	metrics, err := podMetrics.List(tCtx, metav1.ListOptions{})
	require.NoError(t, err)
	assert.Equal(t, "served-by-v1", metrics.Items[0].Name)
	assert.Equal(t, 2, fakeDisc.calls, "discovery should be re-queried after invalidation")
}

func TestVersionedPodMetricsGetterNegotiationErrorsAreNotCached(t *testing.T) {
	testCases := []struct {
		name         string
		versions     []string
		discoveryErr error
		expectedErr  string
	}{
		{
			name:        "metrics API group is missing",
			expectedErr: "no resource metrics API (metrics.k8s.io) registered",
		},
		{
			name:         "discovery request fails",
			versions:     []string{"v1beta1"},
			discoveryErr: errors.New("discovery failed"),
			expectedErr:  "discovery failed",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)

			fakeDisc := discoveryFor(tc.versions...)
			fakeDisc.err = tc.discoveryErr

			getter := newVersionedGetter(fakeDisc)
			podMetrics := getter.PodMetricses("test-ns")

			_, err := podMetrics.List(tCtx, metav1.ListOptions{})
			require.ErrorContains(t, err, tc.expectedErr)

			// Discovery recovers and serves v1beta1. No explicit cache
			// invalidation should be necessary after a negotiation error.
			fakeDisc.groups = discoveryFor("v1beta1").groups
			fakeDisc.err = nil

			metrics, err := podMetrics.List(tCtx, metav1.ListOptions{})
			require.NoError(t, err)
			require.Len(t, metrics.Items, 1)
			assert.Equal(t, "served-by-v1beta1", metrics.Items[0].Name)
			assert.Equal(t, 2, fakeDisc.calls)
		})
	}
}

func TestVersionedPodMetricsGetterHonorsDiscoveryCancellation(t *testing.T) {
	tCtx := ktesting.Init(t)
	cancelCtx := tCtx.WithCancel()
	cancelCtx.Cancel("testing discovery cancellation")

	getter := newVersionedGetter(discoveryFor("v1"))

	_, err := getter.PodMetricses("test-ns").List(
		cancelCtx,
		metav1.ListOptions{},
	)
	require.ErrorIs(t, err, context.Canceled)
}
