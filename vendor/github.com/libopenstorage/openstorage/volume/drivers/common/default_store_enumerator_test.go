package common

import (
	"testing"

	"go.pedge.io/dlog"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/portworx/kvdb"
	"github.com/portworx/kvdb/mem"
	"github.com/stretchr/testify/assert"
)

var (
	testEnumerator volume.StoreEnumerator
	testLabels     = map[string]string{"Foo": "DEADBEEF"}
)

func init() {
	kv, err := kvdb.New(mem.Name, "driver_test", []string{}, nil, dlog.Panicf)
	if err != nil {
		dlog.Panicf("Failed to initialize KVDB")
	}
	if err := kvdb.SetInstance(kv); err != nil {
		dlog.Panicf("Failed to set KVDB instance")
	}
	testEnumerator = NewDefaultStoreEnumerator("enumerator_test", kv)
}

func TestInspect(t *testing.T) {
	volume := newTestVolume("TestVolume")
	err := testEnumerator.CreateVol(volume)
	assert.NoError(t, err, "Failed in CreateVol")
	volumes, err := testEnumerator.Inspect([]string{volume.Id})
	assert.NoError(t, err, "Failed in Inspect")
	assert.Equal(t, len(volumes), 1, "Number of volumes returned in inspect should be 1")
	if len(volumes) == 1 {
		assert.Equal(t, volumes[0].Id, volume.Id, "Invalid volume returned in Inspect")
	}
	err = testEnumerator.DeleteVol(volume.Id)
	assert.NoError(t, err, "Failed in Delete")
	volumes, err = testEnumerator.Inspect([]string{volume.Id})
	assert.NotNil(t, volumes, "Inspect returned nil volumes")
	assert.Equal(t, len(volumes), 0, "Number of volumes returned in inspect should be 0")
}

func TestEnumerate(t *testing.T) {
	volume := newTestVolume("TestVolume")
	err := testEnumerator.CreateVol(volume)
	assert.NoError(t, err, "Failed in CreateVol")
	volumes, err := testEnumerator.Enumerate(&api.VolumeLocator{}, nil)
	assert.NoError(t, err, "Failed in Enumerate")
	assert.Equal(t, 1, len(volumes), "Number of volumes returned in enumerate should be 1")

	volumes, err = testEnumerator.Enumerate(&api.VolumeLocator{Name: volume.Id}, nil)
	assert.NoError(t, err, "Failed in Enumerate")
	assert.Equal(t, 1, len(volumes), "Number of volumes returned in enumerate should be 1")
	if len(volumes) == 1 {
		assert.Equal(t, volumes[0].Id, volume.Id, "Invalid volume returned in Enumerate")
	}
	volumes, err = testEnumerator.Enumerate(&api.VolumeLocator{VolumeLabels: testLabels}, nil)
	assert.NoError(t, err, "Failed in Enumerate")
	assert.Equal(t, len(volumes), 1, "Number of volumes returned in enumerate should be 1")
	if len(volumes) == 1 {
		assert.Equal(t, volumes[0].Id, volume.Id, "Invalid volume returned in Enumerate")
	}
	err = testEnumerator.DeleteVol(volume.Id)
	assert.NoError(t, err, "Failed in Delete")
	volumes, err = testEnumerator.Enumerate(&api.VolumeLocator{Name: volume.Id}, nil)
	assert.Equal(t, len(volumes), 0, "Number of volumes returned in enumerate should be 0")
}

func TestSnapEnumerate(t *testing.T) {
	vol := newTestVolume("TestVolume")
	err := testEnumerator.CreateVol(vol)
	assert.NoError(t, err, "Failed in CreateVol")
	snap := newSnapVolume("SnapVolume", "TestVolume")
	err = testEnumerator.CreateVol(snap)
	assert.NoError(t, err, "Failed in CreateSnap")

	snaps, err := testEnumerator.SnapEnumerate([]string{vol.Id}, nil)
	assert.NoError(t, err, "Failed in Enumerate")
	assert.Equal(t, len(snaps), 1, "Number of snaps returned in enumerate should be 1")
	if len(snaps) == 1 {
		assert.Equal(t, snaps[0].Id, snap.Id, "Invalid snap returned in Enumerate")
	}
	snaps, err = testEnumerator.SnapEnumerate([]string{vol.Id}, testLabels)
	assert.NoError(t, err, "Failed in Enumerate")
	assert.Equal(t, len(snaps), 1, "Number of snaps returned in enumerate should be 1")
	if len(snaps) == 1 {
		assert.Equal(t, snaps[0].Id, snap.Id, "Invalid snap returned in Enumerate")
	}

	snaps, err = testEnumerator.SnapEnumerate(nil, testLabels)
	assert.NoError(t, err, "Failed in Enumerate")
	assert.True(t, len(snaps) >= 1, "Number of snaps returned in enumerate should be at least 1")
	if len(snaps) == 1 {
		assert.Equal(t, snaps[0].Id, snap.Id, "Invalid snap returned in Enumerate")
	}

	snaps, err = testEnumerator.SnapEnumerate(nil, nil)
	assert.NoError(t, err, "Failed in Enumerate")
	assert.True(t, len(snaps) >= 1, "Number of snaps returned in enumerate should be at least 1")
	if len(snaps) == 1 {
		assert.Equal(t, snaps[0].Id, snap.Id, "Invalid snap returned in Enumerate")
	}

	err = testEnumerator.DeleteVol(snap.Id)
	assert.NoError(t, err, "Failed in Delete")
	snaps, err = testEnumerator.SnapEnumerate([]string{vol.Id}, testLabels)
	assert.NotNil(t, snaps, "Inspect returned nil snaps")
	assert.Equal(t, len(snaps), 0, "Number of snaps returned in enumerate should be 0")

	err = testEnumerator.DeleteVol(vol.Id)
	assert.NoError(t, err, "Failed in Delete")
}

func newTestVolume(id string) *api.Volume {
	return &api.Volume{
		Id:      id,
		Locator: &api.VolumeLocator{Name: id, VolumeLabels: testLabels},
		State:   api.VolumeState_VOLUME_STATE_AVAILABLE,
		Spec:    &api.VolumeSpec{},
	}
}

func newSnapVolume(snapID string, volumeID string) *api.Volume {
	return &api.Volume{
		Id:      snapID,
		Locator: &api.VolumeLocator{Name: volumeID, VolumeLabels: testLabels},
		State:   api.VolumeState_VOLUME_STATE_AVAILABLE,
		Spec:    &api.VolumeSpec{},
		Source:  &api.Source{Parent: volumeID},
	}
}
