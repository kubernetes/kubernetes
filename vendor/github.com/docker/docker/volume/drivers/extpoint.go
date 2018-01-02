//go:generate pluginrpc-gen -i $GOFILE -o proxy.go -type volumeDriver -name VolumeDriver

package volumedrivers

import (
	"fmt"
	"sort"
	"sync"

	"github.com/docker/docker/pkg/locker"
	getter "github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/docker/volume"
)

// currently created by hand. generation tool would generate this like:
// $ extpoint-gen Driver > volume/extpoint.go

var drivers = &driverExtpoint{
	extensions: make(map[string]volume.Driver),
	driverLock: &locker.Locker{},
}

const extName = "VolumeDriver"

// NewVolumeDriver returns a driver has the given name mapped on the given client.
func NewVolumeDriver(name string, baseHostPath string, c client) volume.Driver {
	proxy := &volumeDriverProxy{c}
	return &volumeDriverAdapter{name: name, baseHostPath: baseHostPath, proxy: proxy}
}

// volumeDriver defines the available functions that volume plugins must implement.
// This interface is only defined to generate the proxy objects.
// It's not intended to be public or reused.
type volumeDriver interface {
	// Create a volume with the given name
	Create(name string, opts map[string]string) (err error)
	// Remove the volume with the given name
	Remove(name string) (err error)
	// Get the mountpoint of the given volume
	Path(name string) (mountpoint string, err error)
	// Mount the given volume and return the mountpoint
	Mount(name, id string) (mountpoint string, err error)
	// Unmount the given volume
	Unmount(name, id string) (err error)
	// List lists all the volumes known to the driver
	List() (volumes []*proxyVolume, err error)
	// Get retrieves the volume with the requested name
	Get(name string) (volume *proxyVolume, err error)
	// Capabilities gets the list of capabilities of the driver
	Capabilities() (capabilities volume.Capability, err error)
}

type driverExtpoint struct {
	extensions map[string]volume.Driver
	sync.Mutex
	driverLock   *locker.Locker
	plugingetter getter.PluginGetter
}

// RegisterPluginGetter sets the plugingetter
func RegisterPluginGetter(plugingetter getter.PluginGetter) {
	drivers.plugingetter = plugingetter
}

// Register associates the given driver to the given name, checking if
// the name is already associated
func Register(extension volume.Driver, name string) bool {
	if name == "" {
		return false
	}

	drivers.Lock()
	defer drivers.Unlock()

	_, exists := drivers.extensions[name]
	if exists {
		return false
	}

	if err := validateDriver(extension); err != nil {
		return false
	}

	drivers.extensions[name] = extension

	return true
}

// Unregister dissociates the name from its driver, if the association exists.
func Unregister(name string) bool {
	drivers.Lock()
	defer drivers.Unlock()

	_, exists := drivers.extensions[name]
	if !exists {
		return false
	}
	delete(drivers.extensions, name)
	return true
}

// lookup returns the driver associated with the given name. If a
// driver with the given name has not been registered it checks if
// there is a VolumeDriver plugin available with the given name.
func lookup(name string, mode int) (volume.Driver, error) {
	drivers.driverLock.Lock(name)
	defer drivers.driverLock.Unlock(name)

	drivers.Lock()
	ext, ok := drivers.extensions[name]
	drivers.Unlock()
	if ok {
		return ext, nil
	}
	if drivers.plugingetter != nil {
		p, err := drivers.plugingetter.Get(name, extName, mode)
		if err != nil {
			return nil, fmt.Errorf("Error looking up volume plugin %s: %v", name, err)
		}

		d := NewVolumeDriver(p.Name(), p.BasePath(), p.Client())
		if err := validateDriver(d); err != nil {
			return nil, err
		}

		if p.IsV1() {
			drivers.Lock()
			drivers.extensions[name] = d
			drivers.Unlock()
		}
		return d, nil
	}
	return nil, fmt.Errorf("Error looking up volume plugin %s", name)
}

func validateDriver(vd volume.Driver) error {
	scope := vd.Scope()
	if scope != volume.LocalScope && scope != volume.GlobalScope {
		return fmt.Errorf("Driver %q provided an invalid capability scope: %s", vd.Name(), scope)
	}
	return nil
}

// GetDriver returns a volume driver by its name.
// If the driver is empty, it looks for the local driver.
func GetDriver(name string) (volume.Driver, error) {
	if name == "" {
		name = volume.DefaultDriverName
	}
	return lookup(name, getter.Lookup)
}

// CreateDriver returns a volume driver by its name and increments RefCount.
// If the driver is empty, it looks for the local driver.
func CreateDriver(name string) (volume.Driver, error) {
	if name == "" {
		name = volume.DefaultDriverName
	}
	return lookup(name, getter.Acquire)
}

// RemoveDriver returns a volume driver by its name and decrements RefCount..
// If the driver is empty, it looks for the local driver.
func RemoveDriver(name string) (volume.Driver, error) {
	if name == "" {
		name = volume.DefaultDriverName
	}
	return lookup(name, getter.Release)
}

// GetDriverList returns list of volume drivers registered.
// If no driver is registered, empty string list will be returned.
func GetDriverList() []string {
	var driverList []string
	drivers.Lock()
	for driverName := range drivers.extensions {
		driverList = append(driverList, driverName)
	}
	drivers.Unlock()
	sort.Strings(driverList)
	return driverList
}

// GetAllDrivers lists all the registered drivers
func GetAllDrivers() ([]volume.Driver, error) {
	var plugins []getter.CompatPlugin
	if drivers.plugingetter != nil {
		var err error
		plugins, err = drivers.plugingetter.GetAllByCap(extName)
		if err != nil {
			return nil, fmt.Errorf("error listing plugins: %v", err)
		}
	}
	var ds []volume.Driver

	drivers.Lock()
	defer drivers.Unlock()

	for _, d := range drivers.extensions {
		ds = append(ds, d)
	}

	for _, p := range plugins {
		name := p.Name()

		if _, ok := drivers.extensions[name]; ok {
			continue
		}

		ext := NewVolumeDriver(name, p.BasePath(), p.Client())
		if p.IsV1() {
			drivers.extensions[name] = ext
		}
		ds = append(ds, ext)
	}
	return ds, nil
}
