/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	fakeDockerClient := NewFakeDockerClientWithVersion("1.2.3", "1.2")
	isp := &imageStatsProvider{fakeDockerClient}
	st, err := isp.ImageStats()
	as := assert.New(t)
	as.NoError(err)
	as.Equal(st.TotalStorageBytes, uint64(0))
}

func TestImageStatsWithImages(t *testing.T) {
	fakeDockerClient := NewFakeDockerClientWithVersion("1.2.3", "1.2")
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
	isp := &imageStatsProvider{fakeDockerClient}
	st, err := isp.ImageStats()
	as := assert.New(t)
	as.NoError(err)
	const expectedOutput uint64 = 1300
	as.Equal(expectedOutput, st.TotalStorageBytes, "expected %d, got %d", expectedOutput, st.TotalStorageBytes)
}
