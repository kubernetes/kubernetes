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

	"k8s.io/apimachinery/pkg/types"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"

	flockerapi "github.com/clusterhq/flocker-go"
	"github.com/stretchr/testify/assert"
)

const pluginName = "kubernetes.io/flocker"
const datasetOneID = "11111111-1111-1111-1111-111111111100"
const nodeOneID = "11111111-1111-1111-1111-111111111111"
const nodeTwoID = "22222222-2222-2222-2222-222222222222"

var _ flockerapi.Clientable = &fakeFlockerClient{}

type fakeFlockerClient struct {
	DatasetID string
	Primary   string
	Deleted   bool
	Metadata  map[string]string
	Nodes     []flockerapi.NodeState
	Error     error
}

func newFakeFlockerClient() *fakeFlockerClient {
	return &fakeFlockerClient{
		DatasetID: datasetOneID,
		Primary:   nodeOneID,
		Deleted:   false,
		Metadata:  map[string]string{"Name": "dataset-one"},
		Nodes: []flockerapi.NodeState{
			{
				Host: "1.2.3.4",
				UUID: nodeOneID,
			},
			{
				Host: "4.5.6.7",
				UUID: nodeTwoID,
			},
		},
	}
}

func (c *fakeFlockerClient) CreateDataset(options *flockerapi.CreateDatasetOptions) (*flockerapi.DatasetState, error) {

	if c.Error != nil {
		return nil, c.Error
	}

	return &flockerapi.DatasetState{
		DatasetID: c.DatasetID,
	}, nil
}

func (c *fakeFlockerClient) DeleteDataset(datasetID string) error {
	c.DatasetID = datasetID
	c.Deleted = true
	return nil
}

func (c *fakeFlockerClient) GetDatasetState(datasetID string) (*flockerapi.DatasetState, error) {
	return &flockerapi.DatasetState{}, nil
}

func (c *fakeFlockerClient) GetDatasetID(metaName string) (datasetID string, err error) {
	if val, ok := c.Metadata["Name"]; !ok {
		return val, nil
	}
	return "", fmt.Errorf("No dataset with metadata X found")
}

func (c *fakeFlockerClient) GetPrimaryUUID() (primaryUUID string, err error) {
	return
}

func (c *fakeFlockerClient) ListNodes() (nodes []flockerapi.NodeState, err error) {
	return c.Nodes, nil
}

func (c *fakeFlockerClient) UpdatePrimaryForDataset(primaryUUID, datasetID string) (*flockerapi.DatasetState, error) {
	return &flockerapi.DatasetState{}, nil
}

type fakeFlockerUtil struct {
}

func (fake *fakeFlockerUtil) CreateVolume(c *flockerVolumeProvisioner) (datasetUUID string, volumeSizeGB int, labels map[string]string, err error) {
	labels = make(map[string]string)
	labels["fakeflockerutil"] = "yes"
	return "test-flocker-volume-uuid", 3, labels, nil
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
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(dir, nil, nil))
	return plugMgr, dir
}

func TestPlugin(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("flockerTest")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), volumetest.NewFakeVolumeHost(tmpDir, nil, nil))

	plug, err := plugMgr.FindPluginByName("kubernetes.io/flocker")
	if err != nil {
		t.Errorf("Can't find the plugin by name")
	}
	spec := &v1.Volume{
		Name: "vol1",
		VolumeSource: v1.VolumeSource{
			Flocker: &v1.FlockerVolumeSource{
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
	plugMgr, dir := newInitializedVolumePlugMgr(t)
	defer os.RemoveAll(dir)

	plug, err := plugMgr.FindPluginByName(pluginName)
	assert.NotNil(plug, "Can't find the plugin by name")
	assert.NoError(err)
}

func TestCanSupport(t *testing.T) {
	assert := assert.New(t)
	plugMgr, dir := newInitializedVolumePlugMgr(t)
	defer os.RemoveAll(dir)

	plug, err := plugMgr.FindPluginByName(pluginName)
	assert.NoError(err)

	specs := map[*volume.Spec]bool{
		{
			Volume: &v1.Volume{
				VolumeSource: v1.VolumeSource{
					Flocker: &v1.FlockerVolumeSource{},
				},
			},
		}: true,
		{
			PersistentVolume: &v1.PersistentVolume{
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						Flocker: &v1.FlockerVolumeSource{},
					},
				},
			},
		}: true,
		{
			Volume: &v1.Volume{
				VolumeSource: v1.VolumeSource{},
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
		Volume: &v1.Volume{
			VolumeSource: v1.VolumeSource{
				Flocker: &v1.FlockerVolumeSource{},
			},
		},
	}
	vs, ro := p.getFlockerVolumeSource(spec)
	assert.False(ro)
	assert.Equal(spec.Volume.Flocker, vs)

	spec = &volume.Spec{
		PersistentVolume: &v1.PersistentVolume{
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					Flocker: &v1.FlockerVolumeSource{},
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

	plugMgr, dir := newInitializedVolumePlugMgr(t)
	defer os.RemoveAll(dir)
	plug, err := plugMgr.FindPluginByName(pluginName)
	assert.NoError(err)

	spec := &volume.Spec{
		Volume: &v1.Volume{
			VolumeSource: v1.VolumeSource{
				Flocker: &v1.FlockerVolumeSource{
					DatasetName: "something",
				},
			},
		},
	}

	_, err = plug.NewMounter(spec, &v1.Pod{}, volume.VolumeOptions{})
	assert.NoError(err)
}

func TestNewMounterDatasetUUID(t *testing.T) {
	assert := assert.New(t)

	plugMgr, dir := newInitializedVolumePlugMgr(t)
	defer os.RemoveAll(dir)
	plug, err := plugMgr.FindPluginByName(pluginName)
	assert.NoError(err)

	spec := &volume.Spec{
		Volume: &v1.Volume{
			VolumeSource: v1.VolumeSource{
				Flocker: &v1.FlockerVolumeSource{
					DatasetUUID: "uuid1",
				},
			},
		},
	}

	mounter, err := plug.NewMounter(spec, &v1.Pod{}, volume.VolumeOptions{})
	assert.NoError(err)
	assert.NotNil(mounter, "got a nil mounter")

}

func TestNewUnmounter(t *testing.T) {
	t.Skip("broken")
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
	datasetState                 *flockerapi.DatasetState
}

func newMockFlockerClient(mockDatasetID, mockPrimaryUUID, mockPath string) *mockFlockerClient {
	return &mockFlockerClient{
		datasetID:   mockDatasetID,
		primaryUUID: mockPrimaryUUID,
		path:        mockPath,
		datasetState: &flockerapi.DatasetState{
			Path:      mockPath,
			DatasetID: mockDatasetID,
			Primary:   mockPrimaryUUID,
		},
	}
}

func (m mockFlockerClient) CreateDataset(metaName string) (*flockerapi.DatasetState, error) {
	return m.datasetState, nil
}
func (m mockFlockerClient) GetDatasetState(datasetID string) (*flockerapi.DatasetState, error) {
	return m.datasetState, nil
}
func (m mockFlockerClient) GetDatasetID(metaName string) (string, error) {
	return m.datasetID, nil
}
func (m mockFlockerClient) GetPrimaryUUID() (string, error) {
	return m.primaryUUID, nil
}
func (m mockFlockerClient) UpdatePrimaryForDataset(primaryUUID, datasetID string) (*flockerapi.DatasetState, error) {
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

	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("poduid")}}
	b := flockerVolumeMounter{flockerVolume: &flockerVolume{pod: pod, plugin: plug.(*flockerPlugin)}}
	b.client = newMockFlockerClient("dataset-id", "primary-uid", mockPath)

	assert.NoError(b.SetUpAt(dir, nil))
	assert.Equal(expectedPath, b.flocker.path)
}
*/
