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

package flocker

import (
	"fmt"
	"os"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	flockerApi "github.com/ClusterHQ/flocker-go"
	"github.com/stretchr/testify/assert"
)

const pluginName = "kubernetes.io/flocker"

type fakeFlockerUtil struct {
}

func (fake *fakeFlockerUtil) CreateVolume(c *flockerVolumeProvisioner) (datasetUUID string, volumeSizeGB int, labels map[string]string, err error) {
	labels = make(map[string]string)
	labels["fakeflockerutil"] = "yes"
	return "test-flocker-volume-uuid", 100, labels, nil
}

func (fake *fakeFlockerUtil) DeleteVolume(cd *flockerVolumeDeleter) error {
	if cd.datasetUUID != "test-flocker-volume-uuid" {
		return fmt.Errorf("Deleter got unexpected datasetUUID: %s", cd.datasetUUID)
	}
	return nil
}

func newInitializedVolumePlugMgr(t *testing.T) (*volume.VolumePluginMgr, string) {
	plugMgr := &volume.VolumePluginMgr{}
	dir, err := utiltesting.MkTmpdir("flocker")
	assert.NoError(t, err)
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(dir, nil, nil, "" /* rootContext */))
	return plugMgr, dir
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("flockerTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil, "" /* rootContext */))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/flocker")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &api.Volume{
		Name: "vol1",
		VolumeSource: api.VolumeSource{
			Flocker: &api.FlockerVolumeSource{
				DatasetUUID: "uuid1",
			},
		},
	}
	fakeManager := &fakeFlockerUtil{}
	fakeMounter := &mount.FakeMounter{}
	mounter, err := plug.(*flockerPlugin).newMounterInternal(volume.NewSpecFromVolume(spec), types.UID("poduid"), fakeManager, fakeMounter)
	if err != nil {
		t.Errorf("Failed to make a new Mounter: %v", err)
	}
	if mounter == nil {
		t.Errorf("Got a nil Mounter")
	}
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

func TestNewMounterDatasetName(t *testing.T) {
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

	_, err = plug.NewMounter(spec, &api.Pod{}, volume.VolumeOptions{})
	assert.NoError(err)
}

func TestNewMounterDatasetUUID(t *testing.T) {
	assert := assert.New(t)

	plugMgr, _ := newInitializedVolumePlugMgr(t)
	plug, err := plugMgr.FindPluginByName(pluginName)
	assert.NoError(err)

	spec := &volume.Spec{
		Volume: &api.Volume{
			VolumeSource: api.VolumeSource{
				Flocker: &api.FlockerVolumeSource{
					DatasetUUID: "uuid1",
				},
			},
		},
	}

	mounter, err := plug.NewMounter(spec, &api.Pod{}, volume.VolumeOptions{})
	assert.NoError(err)
	assert.NotNil(mounter, "got a nil mounter")

}

func TestNewUnmounter(t *testing.T) {
	assert := assert.New(t)

	p := flockerPlugin{}

	unmounter, err := p.NewUnmounter("", types.UID(""))
	assert.Nil(unmounter)
	assert.NoError(err)
}

func TestIsReadOnly(t *testing.T) {
	b := &flockerVolumeMounter{readOnly: true}
	assert.True(t, b.GetAttributes().ReadOnly)
}

type mockFlockerClient struct {
	datasetID, primaryUUID, path string
	datasetState                 *flockerApi.DatasetState
}

func newMockFlockerClient(mockDatasetID, mockPrimaryUUID, mockPath string) *mockFlockerClient {
	return &mockFlockerClient{
		datasetID:   mockDatasetID,
		primaryUUID: mockPrimaryUUID,
		path:        mockPath,
		datasetState: &flockerApi.DatasetState{
			Path:      mockPath,
			DatasetID: mockDatasetID,
			Primary:   mockPrimaryUUID,
		},
	}
}

func (m mockFlockerClient) CreateDataset(metaName string) (*flockerApi.DatasetState, error) {
	return m.datasetState, nil
}
func (m mockFlockerClient) GetDatasetState(datasetID string) (*flockerApi.DatasetState, error) {
	return m.datasetState, nil
}
func (m mockFlockerClient) GetDatasetID(metaName string) (string, error) {
	return m.datasetID, nil
}
func (m mockFlockerClient) GetPrimaryUUID() (string, error) {
	return m.primaryUUID, nil
}
func (m mockFlockerClient) UpdatePrimaryForDataset(primaryUUID, datasetID string) (*flockerApi.DatasetState, error) {
	return m.datasetState, nil
}

/*
TODO: reenable after refactor
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
	b := flockerVolumeMounter{flockerVolume: &flockerVolume{pod: pod, plugin: plug.(*flockerPlugin)}}
	b.client = newMockFlockerClient("dataset-id", "primary-uid", mockPath)

	assert.NoError(b.SetUpAt(dir, nil))
	assert.Equal(expectedPath, b.flocker.path)
}
*/
