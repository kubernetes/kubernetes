/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package flocker

import (
	"io/ioutil"
	"os"
	"testing"

	flockerClient "github.com/ClusterHQ/flocker-go"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
)

const pluginName = "kubernetes.io/flocker"

func newInitializedVolumePlugMgr(t *testing.T) (volume.VolumePluginMgr, string) {
	plugMgr := volume.VolumePluginMgr{}
	dir, err := ioutil.TempDir("", "flocker")
	assert.NoError(t, err)
	plugMgr.InitPlugins(ProbeVolumePlugins(), volume.NewFakeVolumeHost(dir, nil, nil))
	return plugMgr, dir
}

func TestGetByName(t *testing.T) {
	assert := assert.New(t)
	plugMgr, _ := newInitializedVolumePlugMgr(t)

	plug, err := plugMgr.FindPluginByName(pluginName)
	assert.NotNil(plug, "Can't find the plugin by name")
	assert.NoError(err)
}

func TestCanSupport(t *testing.T) {
	assert := assert.New(t)
	plugMgr, _ := newInitializedVolumePlugMgr(t)

	plug, err := plugMgr.FindPluginByName(pluginName)
	assert.NoError(err)

	specs := map[*volume.Spec]bool{
		&volume.Spec{
			Volume: &api.Volume{
				VolumeSource: api.VolumeSource{
					Flocker: &api.FlockerVolumeSource{},
				},
			},
		}: true,
		&volume.Spec{
			PersistentVolume: &api.PersistentVolume{
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						Flocker: &api.FlockerVolumeSource{},
					},
				},
			},
		}: true,
		&volume.Spec{
			Volume: &api.Volume{
				VolumeSource: api.VolumeSource{},
			},
		}: false,
	}

	for spec, expected := range specs {
		actual := plug.CanSupport(spec)
		assert.Equal(expected, actual)
	}
}

func TestGetFlockerVolumeSource(t *testing.T) {
	assert := assert.New(t)

	p := flockerPlugin{}

	spec := &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				Flocker: &api.FlockerVolumeSource{},
			},
		},
	}
	vs, ro := p.getFlockerVolumeSource(spec)
	assert.False(ro)
	assert.Equal(spec.Volume.Flocker, vs)

	spec = &volume.Spec{
		PersistentVolume: &api.PersistentVolume{
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					Flocker: &api.FlockerVolumeSource{},
				},
			},
		},
	}
	vs, ro = p.getFlockerVolumeSource(spec)
	assert.False(ro)
	assert.Equal(spec.PersistentVolume.Spec.Flocker, vs)
}

func TestNewBuilder(t *testing.T) {
	assert := assert.New(t)

	plugMgr, _ := newInitializedVolumePlugMgr(t)
	plug, err := plugMgr.FindPluginByName(pluginName)
	assert.NoError(err)

	spec := &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				Flocker: &api.FlockerVolumeSource{
					DatasetName: "something",
				},
			},
		},
	}

	_, err = plug.NewBuilder(spec, &api.Pod{}, volume.VolumeOptions{})
	assert.NoError(err)
}

func TestNewCleaner(t *testing.T) {
	assert := assert.New(t)

	p := flockerPlugin{}

	cleaner, err := p.NewCleaner("", types.UID(""))
	assert.Nil(cleaner)
	assert.NoError(err)
}

func TestIsReadOnly(t *testing.T) {
	b := flockerBuilder{readOnly: true}
	assert.True(t, b.IsReadOnly())
}

func TestGetPath(t *testing.T) {
	const expectedPath = "/flocker/expected"

	assert := assert.New(t)

	b := flockerBuilder{flocker: &flocker{path: expectedPath}}
	assert.Equal(expectedPath, b.GetPath())
}

type mockFlockerClient struct {
	datasetID, primaryUUID, path string
	datasetState                 *flockerClient.DatasetState
}

func newMockFlockerClient(mockDatasetID, mockPrimaryUUID, mockPath string) *mockFlockerClient {
	return &mockFlockerClient{
		datasetID:   mockDatasetID,
		primaryUUID: mockPrimaryUUID,
		path:        mockPath,
		datasetState: &flockerClient.DatasetState{
			Path:      mockPath,
			DatasetID: mockDatasetID,
			Primary:   mockPrimaryUUID,
		},
	}
}

func (m mockFlockerClient) CreateDataset(metaName string) (*flockerClient.DatasetState, error) {
	return m.datasetState, nil
}
func (m mockFlockerClient) GetDatasetState(datasetID string) (*flockerClient.DatasetState, error) {
	return m.datasetState, nil
}
func (m mockFlockerClient) GetDatasetID(metaName string) (string, error) {
	return m.datasetID, nil
}
func (m mockFlockerClient) GetPrimaryUUID() (string, error) {
	return m.primaryUUID, nil
}
func (m mockFlockerClient) UpdatePrimaryForDataset(primaryUUID, datasetID string) (*flockerClient.DatasetState, error) {
	return m.datasetState, nil
}

func TestSetUpAtInternal(t *testing.T) {
	const dir = "dir"
	mockPath := "expected-to-be-set-properly" // package var
	expectedPath := mockPath

	assert := assert.New(t)

	plugMgr, rootDir := newInitializedVolumePlugMgr(t)
	if rootDir != "" {
		defer os.RemoveAll(rootDir)
	}
	plug, err := plugMgr.FindPluginByName(flockerPluginName)
	assert.NoError(err)

	pod := &api.Pod{ObjectMeta: api.ObjectMeta{UID: types.UID("poduid")}}
	b := flockerBuilder{flocker: &flocker{pod: pod, plugin: plug.(*flockerPlugin)}}
	b.client = newMockFlockerClient("dataset-id", "primary-uid", mockPath)

	assert.NoError(b.SetUpAt(dir))
	assert.Equal(expectedPath, b.flocker.path)
}
