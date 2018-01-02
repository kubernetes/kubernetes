package testutils

import (
	"fmt"
	"time"

	"github.com/docker/docker/volume"
)

// NoopVolume is a volume that doesn't perform any operation
type NoopVolume struct{}

// Name is the name of the volume
func (NoopVolume) Name() string { return "noop" }

// DriverName is the name of the driver
func (NoopVolume) DriverName() string { return "noop" }

// Path is the filesystem path to the volume
func (NoopVolume) Path() string { return "noop" }

// Mount mounts the volume in the container
func (NoopVolume) Mount(_ string) (string, error) { return "noop", nil }

// Unmount unmounts the volume from the container
func (NoopVolume) Unmount(_ string) error { return nil }

// Status provides low-level details about the volume
func (NoopVolume) Status() map[string]interface{} { return nil }

// CreatedAt provides the time the volume (directory) was created at
func (NoopVolume) CreatedAt() (time.Time, error) { return time.Now(), nil }

// FakeVolume is a fake volume with a random name
type FakeVolume struct {
	name       string
	driverName string
}

// NewFakeVolume creates a new fake volume for testing
func NewFakeVolume(name string, driverName string) volume.Volume {
	return FakeVolume{name: name, driverName: driverName}
}

// Name is the name of the volume
func (f FakeVolume) Name() string { return f.name }

// DriverName is the name of the driver
func (f FakeVolume) DriverName() string { return f.driverName }

// Path is the filesystem path to the volume
func (FakeVolume) Path() string { return "fake" }

// Mount mounts the volume in the container
func (FakeVolume) Mount(_ string) (string, error) { return "fake", nil }

// Unmount unmounts the volume from the container
func (FakeVolume) Unmount(_ string) error { return nil }

// Status provides low-level details about the volume
func (FakeVolume) Status() map[string]interface{} { return nil }

// CreatedAt provides the time the volume (directory) was created at
func (FakeVolume) CreatedAt() (time.Time, error) { return time.Now(), nil }

// FakeDriver is a driver that generates fake volumes
type FakeDriver struct {
	name string
	vols map[string]volume.Volume
}

// NewFakeDriver creates a new FakeDriver with the specified name
func NewFakeDriver(name string) volume.Driver {
	return &FakeDriver{
		name: name,
		vols: make(map[string]volume.Volume),
	}
}

// Name is the name of the driver
func (d *FakeDriver) Name() string { return d.name }

// Create initializes a fake volume.
// It returns an error if the options include an "error" key with a message
func (d *FakeDriver) Create(name string, opts map[string]string) (volume.Volume, error) {
	if opts != nil && opts["error"] != "" {
		return nil, fmt.Errorf(opts["error"])
	}
	v := NewFakeVolume(name, d.name)
	d.vols[name] = v
	return v, nil
}

// Remove deletes a volume.
func (d *FakeDriver) Remove(v volume.Volume) error {
	if _, exists := d.vols[v.Name()]; !exists {
		return fmt.Errorf("no such volume")
	}
	delete(d.vols, v.Name())
	return nil
}

// List lists the volumes
func (d *FakeDriver) List() ([]volume.Volume, error) {
	var vols []volume.Volume
	for _, v := range d.vols {
		vols = append(vols, v)
	}
	return vols, nil
}

// Get gets the volume
func (d *FakeDriver) Get(name string) (volume.Volume, error) {
	if v, exists := d.vols[name]; exists {
		return v, nil
	}
	return nil, fmt.Errorf("no such volume")
}

// Scope returns the local scope
func (*FakeDriver) Scope() string {
	return "local"
}
