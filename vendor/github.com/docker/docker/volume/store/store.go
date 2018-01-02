package store

import (
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/pkg/errors"

	"github.com/boltdb/bolt"
	"github.com/docker/docker/pkg/locker"
	"github.com/docker/docker/volume"
	"github.com/docker/docker/volume/drivers"
	"github.com/sirupsen/logrus"
)

const (
	volumeDataDir = "volumes"
)

type volumeWrapper struct {
	volume.Volume
	labels  map[string]string
	scope   string
	options map[string]string
}

func (v volumeWrapper) Options() map[string]string {
	options := map[string]string{}
	for key, value := range v.options {
		options[key] = value
	}
	return options
}

func (v volumeWrapper) Labels() map[string]string {
	return v.labels
}

func (v volumeWrapper) Scope() string {
	return v.scope
}

func (v volumeWrapper) CachedPath() string {
	if vv, ok := v.Volume.(interface {
		CachedPath() string
	}); ok {
		return vv.CachedPath()
	}
	return v.Volume.Path()
}

// New initializes a VolumeStore to keep
// reference counting of volumes in the system.
func New(rootPath string) (*VolumeStore, error) {
	vs := &VolumeStore{
		locks:   &locker.Locker{},
		names:   make(map[string]volume.Volume),
		refs:    make(map[string]map[string]struct{}),
		labels:  make(map[string]map[string]string),
		options: make(map[string]map[string]string),
	}

	if rootPath != "" {
		// initialize metadata store
		volPath := filepath.Join(rootPath, volumeDataDir)
		if err := os.MkdirAll(volPath, 750); err != nil {
			return nil, err
		}

		dbPath := filepath.Join(volPath, "metadata.db")

		var err error
		vs.db, err = bolt.Open(dbPath, 0600, &bolt.Options{Timeout: 1 * time.Second})
		if err != nil {
			return nil, errors.Wrap(err, "error while opening volume store metadata database")
		}

		// initialize volumes bucket
		if err := vs.db.Update(func(tx *bolt.Tx) error {
			if _, err := tx.CreateBucketIfNotExists(volumeBucketName); err != nil {
				return errors.Wrap(err, "error while setting up volume store metadata database")
			}
			return nil
		}); err != nil {
			return nil, err
		}
	}

	vs.restore()

	return vs, nil
}

func (s *VolumeStore) getNamed(name string) (volume.Volume, bool) {
	s.globalLock.RLock()
	v, exists := s.names[name]
	s.globalLock.RUnlock()
	return v, exists
}

func (s *VolumeStore) setNamed(v volume.Volume, ref string) {
	name := v.Name()

	s.globalLock.Lock()
	s.names[name] = v
	if len(ref) > 0 {
		if s.refs[name] == nil {
			s.refs[name] = make(map[string]struct{})
		}
		s.refs[name][ref] = struct{}{}
	}
	s.globalLock.Unlock()
}

// hasRef returns true if the given name has at least one ref.
// Callers of this function are expected to hold the name lock.
func (s *VolumeStore) hasRef(name string) bool {
	s.globalLock.RLock()
	l := len(s.refs[name])
	s.globalLock.RUnlock()
	return l > 0
}

// getRefs gets the list of refs for a given name
// Callers of this function are expected to hold the name lock.
func (s *VolumeStore) getRefs(name string) []string {
	s.globalLock.RLock()
	defer s.globalLock.RUnlock()

	refs := make([]string, 0, len(s.refs[name]))
	for r := range s.refs[name] {
		refs = append(refs, r)
	}

	return refs
}

// Purge allows the cleanup of internal data on docker in case
// the internal data is out of sync with volumes driver plugins.
func (s *VolumeStore) Purge(name string) {
	s.globalLock.Lock()
	v, exists := s.names[name]
	if exists {
		if _, err := volumedrivers.RemoveDriver(v.DriverName()); err != nil {
			logrus.Errorf("Error dereferencing volume driver: %v", err)
		}
	}
	if err := s.removeMeta(name); err != nil {
		logrus.Errorf("Error removing volume metadata for volume %q: %v", name, err)
	}
	delete(s.names, name)
	delete(s.refs, name)
	delete(s.labels, name)
	delete(s.options, name)
	s.globalLock.Unlock()
}

// VolumeStore is a struct that stores the list of volumes available and keeps track of their usage counts
type VolumeStore struct {
	// locks ensures that only one action is being performed on a particular volume at a time without locking the entire store
	// since actions on volumes can be quite slow, this ensures the store is free to handle requests for other volumes.
	locks *locker.Locker
	// globalLock is used to protect access to mutable structures used by the store object
	globalLock sync.RWMutex
	// names stores the volume name -> volume relationship.
	// This is used for making lookups faster so we don't have to probe all drivers
	names map[string]volume.Volume
	// refs stores the volume name and the list of things referencing it
	refs map[string]map[string]struct{}
	// labels stores volume labels for each volume
	labels map[string]map[string]string
	// options stores volume options for each volume
	options map[string]map[string]string
	db      *bolt.DB
}

// List proxies to all registered volume drivers to get the full list of volumes
// If a driver returns a volume that has name which conflicts with another volume from a different driver,
// the first volume is chosen and the conflicting volume is dropped.
func (s *VolumeStore) List() ([]volume.Volume, []string, error) {
	vols, warnings, err := s.list()
	if err != nil {
		return nil, nil, &OpErr{Err: err, Op: "list"}
	}
	var out []volume.Volume

	for _, v := range vols {
		name := normaliseVolumeName(v.Name())

		s.locks.Lock(name)
		storedV, exists := s.getNamed(name)
		// Note: it's not safe to populate the cache here because the volume may have been
		// deleted before we acquire a lock on its name
		if exists && storedV.DriverName() != v.DriverName() {
			logrus.Warnf("Volume name %s already exists for driver %s, not including volume returned by %s", v.Name(), storedV.DriverName(), v.DriverName())
			s.locks.Unlock(v.Name())
			continue
		}

		out = append(out, v)
		s.locks.Unlock(v.Name())
	}
	return out, warnings, nil
}

// list goes through each volume driver and asks for its list of volumes.
func (s *VolumeStore) list() ([]volume.Volume, []string, error) {
	var (
		ls       []volume.Volume
		warnings []string
	)

	drivers, err := volumedrivers.GetAllDrivers()
	if err != nil {
		return nil, nil, err
	}

	type vols struct {
		vols       []volume.Volume
		err        error
		driverName string
	}
	chVols := make(chan vols, len(drivers))

	for _, vd := range drivers {
		go func(d volume.Driver) {
			vs, err := d.List()
			if err != nil {
				chVols <- vols{driverName: d.Name(), err: &OpErr{Err: err, Name: d.Name(), Op: "list"}}
				return
			}
			for i, v := range vs {
				s.globalLock.RLock()
				vs[i] = volumeWrapper{v, s.labels[v.Name()], d.Scope(), s.options[v.Name()]}
				s.globalLock.RUnlock()
			}

			chVols <- vols{vols: vs}
		}(vd)
	}

	badDrivers := make(map[string]struct{})
	for i := 0; i < len(drivers); i++ {
		vs := <-chVols

		if vs.err != nil {
			warnings = append(warnings, vs.err.Error())
			badDrivers[vs.driverName] = struct{}{}
			logrus.Warn(vs.err)
		}
		ls = append(ls, vs.vols...)
	}

	if len(badDrivers) > 0 {
		s.globalLock.RLock()
		for _, v := range s.names {
			if _, exists := badDrivers[v.DriverName()]; exists {
				ls = append(ls, v)
			}
		}
		s.globalLock.RUnlock()
	}
	return ls, warnings, nil
}

// CreateWithRef creates a volume with the given name and driver and stores the ref
// This ensures there's no race between creating a volume and then storing a reference.
func (s *VolumeStore) CreateWithRef(name, driverName, ref string, opts, labels map[string]string) (volume.Volume, error) {
	name = normaliseVolumeName(name)
	s.locks.Lock(name)
	defer s.locks.Unlock(name)

	v, err := s.create(name, driverName, opts, labels)
	if err != nil {
		if _, ok := err.(*OpErr); ok {
			return nil, err
		}
		return nil, &OpErr{Err: err, Name: name, Op: "create"}
	}

	s.setNamed(v, ref)
	return v, nil
}

// Create creates a volume with the given name and driver.
// This is just like CreateWithRef() except we don't store a reference while holding the lock.
func (s *VolumeStore) Create(name, driverName string, opts, labels map[string]string) (volume.Volume, error) {
	return s.CreateWithRef(name, driverName, "", opts, labels)
}

// checkConflict checks the local cache for name collisions with the passed in name,
// for existing volumes with the same name but in a different driver.
// This is used by `Create` as a best effort to prevent name collisions for volumes.
// If a matching volume is found that is not a conflict that is returned so the caller
// does not need to perform an additional lookup.
// When no matching volume is found, both returns will be nil
//
// Note: This does not probe all the drivers for name collisions because v1 plugins
// are very slow, particularly if the plugin is down, and cause other issues,
// particularly around locking the store.
// TODO(cpuguy83): With v2 plugins this shouldn't be a problem. Could also potentially
// use a connect timeout for this kind of check to ensure we aren't blocking for a
// long time.
func (s *VolumeStore) checkConflict(name, driverName string) (volume.Volume, error) {
	// check the local cache
	v, _ := s.getNamed(name)
	if v == nil {
		return nil, nil
	}

	vDriverName := v.DriverName()
	var conflict bool
	if driverName != "" {
		// Retrieve canonical driver name to avoid inconsistencies (for example
		// "plugin" vs. "plugin:latest")
		vd, err := volumedrivers.GetDriver(driverName)
		if err != nil {
			return nil, err
		}

		if vDriverName != vd.Name() {
			conflict = true
		}
	}

	// let's check if the found volume ref
	// is stale by checking with the driver if it still exists
	exists, err := volumeExists(v)
	if err != nil {
		return nil, errors.Wrapf(errNameConflict, "found reference to volume '%s' in driver '%s', but got an error while checking the driver: %v", name, vDriverName, err)
	}

	if exists {
		if conflict {
			return nil, errors.Wrapf(errNameConflict, "driver '%s' already has volume '%s'", vDriverName, name)
		}
		return v, nil
	}

	if s.hasRef(v.Name()) {
		// Containers are referencing this volume but it doesn't seem to exist anywhere.
		// Return a conflict error here, the user can fix this with `docker volume rm -f`
		return nil, errors.Wrapf(errNameConflict, "found references to volume '%s' in driver '%s' but the volume was not found in the driver -- you may need to remove containers referencing this volume or force remove the volume to re-create it", name, vDriverName)
	}

	// doesn't exist, so purge it from the cache
	s.Purge(name)
	return nil, nil
}

// volumeExists returns if the volume is still present in the driver.
// An error is returned if there was an issue communicating with the driver.
func volumeExists(v volume.Volume) (bool, error) {
	exists, err := lookupVolume(v.DriverName(), v.Name())
	if err != nil {
		return false, err
	}
	return exists != nil, nil
}

// create asks the given driver to create a volume with the name/opts.
// If a volume with the name is already known, it will ask the stored driver for the volume.
// If the passed in driver name does not match the driver name which is stored
//  for the given volume name, an error is returned after checking if the reference is stale.
// If the reference is stale, it will be purged and this create can continue.
// It is expected that callers of this function hold any necessary locks.
func (s *VolumeStore) create(name, driverName string, opts, labels map[string]string) (volume.Volume, error) {
	// Validate the name in a platform-specific manner
	valid, err := volume.IsVolumeNameValid(name)
	if err != nil {
		return nil, err
	}
	if !valid {
		return nil, &OpErr{Err: errInvalidName, Name: name, Op: "create"}
	}

	v, err := s.checkConflict(name, driverName)
	if err != nil {
		return nil, err
	}

	if v != nil {
		return v, nil
	}

	// Since there isn't a specified driver name, let's see if any of the existing drivers have this volume name
	if driverName == "" {
		v, _ := s.getVolume(name)
		if v != nil {
			return v, nil
		}
	}

	vd, err := volumedrivers.CreateDriver(driverName)

	if err != nil {
		return nil, &OpErr{Op: "create", Name: name, Err: err}
	}

	logrus.Debugf("Registering new volume reference: driver %q, name %q", vd.Name(), name)

	if v, _ := vd.Get(name); v != nil {
		return v, nil
	}
	v, err = vd.Create(name, opts)
	if err != nil {
		return nil, err
	}
	s.globalLock.Lock()
	s.labels[name] = labels
	s.options[name] = opts
	s.refs[name] = make(map[string]struct{})
	s.globalLock.Unlock()

	metadata := volumeMetadata{
		Name:    name,
		Driver:  vd.Name(),
		Labels:  labels,
		Options: opts,
	}

	if err := s.setMeta(name, metadata); err != nil {
		return nil, err
	}
	return volumeWrapper{v, labels, vd.Scope(), opts}, nil
}

// GetWithRef gets a volume with the given name from the passed in driver and stores the ref
// This is just like Get(), but we store the reference while holding the lock.
// This makes sure there are no races between checking for the existence of a volume and adding a reference for it
func (s *VolumeStore) GetWithRef(name, driverName, ref string) (volume.Volume, error) {
	name = normaliseVolumeName(name)
	s.locks.Lock(name)
	defer s.locks.Unlock(name)

	vd, err := volumedrivers.GetDriver(driverName)
	if err != nil {
		return nil, &OpErr{Err: err, Name: name, Op: "get"}
	}

	v, err := vd.Get(name)
	if err != nil {
		return nil, &OpErr{Err: err, Name: name, Op: "get"}
	}

	s.setNamed(v, ref)

	s.globalLock.RLock()
	defer s.globalLock.RUnlock()
	return volumeWrapper{v, s.labels[name], vd.Scope(), s.options[name]}, nil
}

// Get looks if a volume with the given name exists and returns it if so
func (s *VolumeStore) Get(name string) (volume.Volume, error) {
	name = normaliseVolumeName(name)
	s.locks.Lock(name)
	defer s.locks.Unlock(name)

	v, err := s.getVolume(name)
	if err != nil {
		return nil, &OpErr{Err: err, Name: name, Op: "get"}
	}
	s.setNamed(v, "")
	return v, nil
}

// getVolume requests the volume, if the driver info is stored it just accesses that driver,
// if the driver is unknown it probes all drivers until it finds the first volume with that name.
// it is expected that callers of this function hold any necessary locks
func (s *VolumeStore) getVolume(name string) (volume.Volume, error) {
	var meta volumeMetadata
	meta, err := s.getMeta(name)
	if err != nil {
		return nil, err
	}

	driverName := meta.Driver
	if driverName == "" {
		s.globalLock.RLock()
		v, exists := s.names[name]
		s.globalLock.RUnlock()
		if exists {
			meta.Driver = v.DriverName()
			if err := s.setMeta(name, meta); err != nil {
				return nil, err
			}
		}
	}

	if meta.Driver != "" {
		vol, err := lookupVolume(meta.Driver, name)
		if err != nil {
			return nil, err
		}
		if vol == nil {
			s.Purge(name)
			return nil, errNoSuchVolume
		}

		var scope string
		vd, err := volumedrivers.GetDriver(meta.Driver)
		if err == nil {
			scope = vd.Scope()
		}
		return volumeWrapper{vol, meta.Labels, scope, meta.Options}, nil
	}

	logrus.Debugf("Probing all drivers for volume with name: %s", name)
	drivers, err := volumedrivers.GetAllDrivers()
	if err != nil {
		return nil, err
	}

	for _, d := range drivers {
		v, err := d.Get(name)
		if err != nil || v == nil {
			continue
		}
		meta.Driver = v.DriverName()
		if err := s.setMeta(name, meta); err != nil {
			return nil, err
		}
		return volumeWrapper{v, meta.Labels, d.Scope(), meta.Options}, nil
	}
	return nil, errNoSuchVolume
}

// lookupVolume gets the specified volume from the specified driver.
// This will only return errors related to communications with the driver.
// If the driver returns an error that is not communication related the
//   error is logged but not returned.
// If the volume is not found it will return `nil, nil``
func lookupVolume(driverName, volumeName string) (volume.Volume, error) {
	vd, err := volumedrivers.GetDriver(driverName)
	if err != nil {
		return nil, errors.Wrapf(err, "error while checking if volume %q exists in driver %q", volumeName, driverName)
	}
	v, err := vd.Get(volumeName)
	if err != nil {
		err = errors.Cause(err)
		if _, ok := err.(net.Error); ok {
			if v != nil {
				volumeName = v.Name()
				driverName = v.DriverName()
			}
			return nil, errors.Wrapf(err, "error while checking if volume %q exists in driver %q", volumeName, driverName)
		}

		// At this point, the error could be anything from the driver, such as "no such volume"
		// Let's not check an error here, and instead check if the driver returned a volume
		logrus.WithError(err).WithField("driver", driverName).WithField("volume", volumeName).Warnf("Error while looking up volume")
	}
	return v, nil
}

// Remove removes the requested volume. A volume is not removed if it has any refs
func (s *VolumeStore) Remove(v volume.Volume) error {
	name := normaliseVolumeName(v.Name())
	s.locks.Lock(name)
	defer s.locks.Unlock(name)

	if s.hasRef(name) {
		return &OpErr{Err: errVolumeInUse, Name: v.Name(), Op: "remove", Refs: s.getRefs(name)}
	}

	vd, err := volumedrivers.GetDriver(v.DriverName())
	if err != nil {
		return &OpErr{Err: err, Name: v.DriverName(), Op: "remove"}
	}

	logrus.Debugf("Removing volume reference: driver %s, name %s", v.DriverName(), name)
	vol := unwrapVolume(v)
	if err := vd.Remove(vol); err != nil {
		return &OpErr{Err: err, Name: name, Op: "remove"}
	}

	s.Purge(name)
	return nil
}

// Dereference removes the specified reference to the volume
func (s *VolumeStore) Dereference(v volume.Volume, ref string) {
	name := v.Name()

	s.locks.Lock(name)
	defer s.locks.Unlock(name)

	s.globalLock.Lock()
	defer s.globalLock.Unlock()

	if s.refs[name] != nil {
		delete(s.refs[name], ref)
	}
}

// Refs gets the current list of refs for the given volume
func (s *VolumeStore) Refs(v volume.Volume) []string {
	name := v.Name()

	s.locks.Lock(name)
	defer s.locks.Unlock(name)

	return s.getRefs(name)
}

// FilterByDriver returns the available volumes filtered by driver name
func (s *VolumeStore) FilterByDriver(name string) ([]volume.Volume, error) {
	vd, err := volumedrivers.GetDriver(name)
	if err != nil {
		return nil, &OpErr{Err: err, Name: name, Op: "list"}
	}
	ls, err := vd.List()
	if err != nil {
		return nil, &OpErr{Err: err, Name: name, Op: "list"}
	}
	for i, v := range ls {
		options := map[string]string{}
		s.globalLock.RLock()
		for key, value := range s.options[v.Name()] {
			options[key] = value
		}
		ls[i] = volumeWrapper{v, s.labels[v.Name()], vd.Scope(), options}
		s.globalLock.RUnlock()
	}
	return ls, nil
}

// FilterByUsed returns the available volumes filtered by if they are in use or not.
// `used=true` returns only volumes that are being used, while `used=false` returns
// only volumes that are not being used.
func (s *VolumeStore) FilterByUsed(vols []volume.Volume, used bool) []volume.Volume {
	return s.filter(vols, func(v volume.Volume) bool {
		s.locks.Lock(v.Name())
		hasRef := s.hasRef(v.Name())
		s.locks.Unlock(v.Name())
		return used == hasRef
	})
}

// filterFunc defines a function to allow filter volumes in the store
type filterFunc func(vol volume.Volume) bool

// filter returns the available volumes filtered by a filterFunc function
func (s *VolumeStore) filter(vols []volume.Volume, f filterFunc) []volume.Volume {
	var ls []volume.Volume
	for _, v := range vols {
		if f(v) {
			ls = append(ls, v)
		}
	}
	return ls
}

func unwrapVolume(v volume.Volume) volume.Volume {
	if vol, ok := v.(volumeWrapper); ok {
		return vol.Volume
	}

	return v
}

// Shutdown releases all resources used by the volume store
// It does not make any changes to volumes, drivers, etc.
func (s *VolumeStore) Shutdown() error {
	return s.db.Close()
}
