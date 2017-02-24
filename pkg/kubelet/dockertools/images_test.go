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

package dockertools

import (
	"testing"

	dockertypes "github.com/docker/engine-api/types"
	"github.com/stretchr/testify/assert"
)

func TestImageStatsNoImages(t *testing.T) {
	fakeDockerClient := NewFakeDockerClient().WithVersion("1.2.3", "1.2")
	isp := newImageStatsProvider(fakeDockerClient)
	st, err := isp.ImageStats()
	as := assert.New(t)
	as.NoError(err)
	as.NoError(fakeDockerClient.AssertCalls([]string{"list_images"}))
	as.Equal(st.TotalStorageBytes, uint64(0))
}

func TestImageStatsWithImages(t *testing.T) {
	fakeDockerClient := NewFakeDockerClient().WithVersion("1.2.3", "1.2")
	fakeHistoryData := map[string][]dockertypes.ImageHistory{
		"busybox": {
			{
				ID:        "0123456",
				CreatedBy: "foo",
				Size:      100,
			},
			{
				ID:        "0123457",
				CreatedBy: "duplicate",
				Size:      200,
			},
			{
				ID:        "<missing>",
				CreatedBy: "baz",
				Size:      300,
			},
		},
		"kubelet": {
			{
				ID:        "1123456",
				CreatedBy: "foo",
				Size:      200,
			},
			{
				ID:        "<missing>",
				CreatedBy: "1baz",
				Size:      400,
			},
		},
		"busybox-new": {
			{
				ID:        "01234567",
				CreatedBy: "foo",
				Size:      100,
			},
			{
				ID:        "0123457",
				CreatedBy: "duplicate",
				Size:      200,
			},
			{
				ID:        "<missing>",
				CreatedBy: "baz",
				Size:      300,
			},
		},
	}
	fakeDockerClient.InjectImageHistory(fakeHistoryData)
	fakeDockerClient.InjectImages([]dockertypes.Image{
		{
			ID: "busybox",
		},
		{
			ID: "kubelet",
		},
		{
			ID: "busybox-new",
		},
	})
	isp := newImageStatsProvider(fakeDockerClient)
	st, err := isp.ImageStats()
	as := assert.New(t)
	as.NoError(err)
	as.NoError(fakeDockerClient.AssertCalls([]string{"list_images", "image_history", "image_history", "image_history"}))
	const expectedOutput uint64 = 1300
	as.Equal(expectedOutput, st.TotalStorageBytes, "expected %d, got %d", expectedOutput, st.TotalStorageBytes)
}

func TestImageStatsWithCachedImages(t *testing.T) {
	for _, test := range []struct {
		oldLayers                map[string]*dockertypes.ImageHistory
		oldImageToLayerIDs       map[string][]string
		images                   []dockertypes.Image
		history                  map[string][]dockertypes.ImageHistory
		expectedCalls            []string
		expectedLayers           map[string]*dockertypes.ImageHistory
		expectedImageToLayerIDs  map[string][]string
		expectedTotalStorageSize uint64
	}{
		{
			// No cache
			oldLayers:          make(map[string]*dockertypes.ImageHistory),
			oldImageToLayerIDs: make(map[string][]string),
			images: []dockertypes.Image{
				{
					ID: "busybox",
				},
				{
					ID: "kubelet",
				},
			},
			history: map[string][]dockertypes.ImageHistory{
				"busybox": {
					{
						ID:        "0123456",
						CreatedBy: "foo",
						Size:      100,
					},
					{
						ID:        "<missing>",
						CreatedBy: "baz",
						Size:      300,
					},
				},
				"kubelet": {
					{
						ID:        "1123456",
						CreatedBy: "foo",
						Size:      200,
					},
					{
						ID:        "<missing>",
						CreatedBy: "1baz",
						Size:      400,
					},
				},
			},
			expectedCalls: []string{"list_images", "image_history", "image_history"},
			expectedLayers: map[string]*dockertypes.ImageHistory{
				"0123456": {
					ID:        "0123456",
					CreatedBy: "foo",
					Size:      100,
				},
				"1123456": {
					ID:        "1123456",
					CreatedBy: "foo",
					Size:      200,
				},
				"<missing>baz": {
					ID:        "<missing>",
					CreatedBy: "baz",
					Size:      300,
				},
				"<missing>1baz": {
					ID:        "<missing>",
					CreatedBy: "1baz",
					Size:      400,
				},
			},
			expectedImageToLayerIDs: map[string][]string{
				"busybox": {"0123456", "<missing>baz"},
				"kubelet": {"1123456", "<missing>1baz"},
			},
			expectedTotalStorageSize: 1000,
		},
		{
			// Use cache value
			oldLayers: map[string]*dockertypes.ImageHistory{
				"0123456": {
					ID:        "0123456",
					CreatedBy: "foo",
					Size:      100,
				},
				"<missing>baz": {
					ID:        "<missing>",
					CreatedBy: "baz",
					Size:      300,
				},
			},
			oldImageToLayerIDs: map[string][]string{
				"busybox": {"0123456", "<missing>baz"},
			},
			images: []dockertypes.Image{
				{
					ID: "busybox",
				},
				{
					ID: "kubelet",
				},
			},
			history: map[string][]dockertypes.ImageHistory{
				"busybox": {
					{
						ID:        "0123456",
						CreatedBy: "foo",
						Size:      100,
					},
					{
						ID:        "<missing>",
						CreatedBy: "baz",
						Size:      300,
					},
				},
				"kubelet": {
					{
						ID:        "1123456",
						CreatedBy: "foo",
						Size:      200,
					},
					{
						ID:        "<missing>",
						CreatedBy: "1baz",
						Size:      400,
					},
				},
			},
			expectedCalls: []string{"list_images", "image_history"},
			expectedLayers: map[string]*dockertypes.ImageHistory{
				"0123456": {
					ID:        "0123456",
					CreatedBy: "foo",
					Size:      100,
				},
				"1123456": {
					ID:        "1123456",
					CreatedBy: "foo",
					Size:      200,
				},
				"<missing>baz": {
					ID:        "<missing>",
					CreatedBy: "baz",
					Size:      300,
				},
				"<missing>1baz": {
					ID:        "<missing>",
					CreatedBy: "1baz",
					Size:      400,
				},
			},
			expectedImageToLayerIDs: map[string][]string{
				"busybox": {"0123456", "<missing>baz"},
				"kubelet": {"1123456", "<missing>1baz"},
			},
			expectedTotalStorageSize: 1000,
		},
		{
			// Unused cache value
			oldLayers: map[string]*dockertypes.ImageHistory{
				"0123456": {
					ID:        "0123456",
					CreatedBy: "foo",
					Size:      100,
				},
				"<missing>baz": {
					ID:        "<missing>",
					CreatedBy: "baz",
					Size:      300,
				},
			},
			oldImageToLayerIDs: map[string][]string{
				"busybox": {"0123456", "<missing>baz"},
			},
			images: []dockertypes.Image{
				{
					ID: "kubelet",
				},
			},
			history: map[string][]dockertypes.ImageHistory{
				"kubelet": {
					{
						ID:        "1123456",
						CreatedBy: "foo",
						Size:      200,
					},
					{
						ID:        "<missing>",
						CreatedBy: "1baz",
						Size:      400,
					},
				},
			},
			expectedCalls: []string{"list_images", "image_history"},
			expectedLayers: map[string]*dockertypes.ImageHistory{
				"1123456": {
					ID:        "1123456",
					CreatedBy: "foo",
					Size:      200,
				},
				"<missing>1baz": {
					ID:        "<missing>",
					CreatedBy: "1baz",
					Size:      400,
				},
			},
			expectedImageToLayerIDs: map[string][]string{
				"kubelet": {"1123456", "<missing>1baz"},
			},
			expectedTotalStorageSize: 600,
		},
	} {
		fakeDockerClient := NewFakeDockerClient().WithVersion("1.2.3", "1.2")
		fakeDockerClient.InjectImages(test.images)
		fakeDockerClient.InjectImageHistory(test.history)
		isp := newImageStatsProvider(fakeDockerClient)
		isp.layers = test.oldLayers
		isp.imageToLayerIDs = test.oldImageToLayerIDs
		st, err := isp.ImageStats()
		as := assert.New(t)
		as.NoError(err)
		as.NoError(fakeDockerClient.AssertCalls(test.expectedCalls))
		as.Equal(test.expectedLayers, isp.layers, "expected %+v, got %+v", test.expectedLayers, isp.layers)
		as.Equal(test.expectedImageToLayerIDs, isp.imageToLayerIDs, "expected %+v, got %+v", test.expectedImageToLayerIDs, isp.imageToLayerIDs)
		as.Equal(test.expectedTotalStorageSize, st.TotalStorageBytes, "expected %d, got %d", test.expectedTotalStorageSize, st.TotalStorageBytes)
	}
}
