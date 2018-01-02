package volume

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	mounttypes "github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/stringid"
	"github.com/opencontainers/selinux/go-selinux/label"
	"github.com/pkg/errors"
)

// DefaultDriverName is the driver name used for the driver
// implemented in the local package.
const DefaultDriverName = "local"

// Scopes define if a volume has is cluster-wide (global) or local only.
// Scopes are returned by the volume driver when it is queried for capabilities and then set on a volume
const (
	LocalScope  = "local"
	GlobalScope = "global"
)

// Driver is for creating and removing volumes.
type Driver interface {
	// Name returns the name of the volume driver.
	Name() string
	// Create makes a new volume with the given name.
	Create(name string, opts map[string]string) (Volume, error)
	// Remove deletes the volume.
	Remove(vol Volume) (err error)
	// List lists all the volumes the driver has
	List() ([]Volume, error)
	// Get retrieves the volume with the requested name
	Get(name string) (Volume, error)
	// Scope returns the scope of the driver (e.g. `global` or `local`).
	// Scope determines how the driver is handled at a cluster level
	Scope() string
}

// Capability defines a set of capabilities that a driver is able to handle.
type Capability struct {
	// Scope is the scope of the driver, `global` or `local`
	// A `global` scope indicates that the driver manages volumes across the cluster
	// A `local` scope indicates that the driver only manages volumes resources local to the host
	// Scope is declared by the driver
	Scope string
}

// Volume is a place to store data. It is backed by a specific driver, and can be mounted.
type Volume interface {
	// Name returns the name of the volume
	Name() string
	// DriverName returns the name of the driver which owns this volume.
	DriverName() string
	// Path returns the absolute path to the volume.
	Path() string
	// Mount mounts the volume and returns the absolute path to
	// where it can be consumed.
	Mount(id string) (string, error)
	// Unmount unmounts the volume when it is no longer in use.
	Unmount(id string) error
	// CreatedAt returns Volume Creation time
	CreatedAt() (time.Time, error)
	// Status returns low-level status information about a volume
	Status() map[string]interface{}
}

// DetailedVolume wraps a Volume with user-defined labels, options, and cluster scope (e.g., `local` or `global`)
type DetailedVolume interface {
	Labels() map[string]string
	Options() map[string]string
	Scope() string
	Volume
}

// MountPoint is the intersection point between a volume and a container. It
// specifies which volume is to be used and where inside a container it should
// be mounted.
type MountPoint struct {
	// Source is the source path of the mount.
	// E.g. `mount --bind /foo /bar`, `/foo` is the `Source`.
	Source string
	// Destination is the path relative to the container root (`/`) to the mount point
	// It is where the `Source` is mounted to
	Destination string
	// RW is set to true when the mountpoint should be mounted as read-write
	RW bool
	// Name is the name reference to the underlying data defined by `Source`
	// e.g., the volume name
	Name string
	// Driver is the volume driver used to create the volume (if it is a volume)
	Driver string
	// Type of mount to use, see `Type<foo>` definitions in github.com/docker/docker/api/types/mount
	Type mounttypes.Type `json:",omitempty"`
	// Volume is the volume providing data to this mountpoint.
	// This is nil unless `Type` is set to `TypeVolume`
	Volume Volume `json:"-"`

	// Mode is the comma separated list of options supplied by the user when creating
	// the bind/volume mount.
	// Note Mode is not used on Windows
	Mode string `json:"Relabel,omitempty"` // Originally field was `Relabel`"

	// Propagation describes how the mounts are propagated from the host into the
	// mount point, and vice-versa.
	// See https://www.kernel.org/doc/Documentation/filesystems/sharedsubtree.txt
	// Note Propagation is not used on Windows
	Propagation mounttypes.Propagation `json:",omitempty"` // Mount propagation string

	// Specifies if data should be copied from the container before the first mount
	// Use a pointer here so we can tell if the user set this value explicitly
	// This allows us to error out when the user explicitly enabled copy but we can't copy due to the volume being populated
	CopyData bool `json:"-"`
	// ID is the opaque ID used to pass to the volume driver.
	// This should be set by calls to `Mount` and unset by calls to `Unmount`
	ID string `json:",omitempty"`

	// Sepc is a copy of the API request that created this mount.
	Spec mounttypes.Mount

	// Track usage of this mountpoint
	// Specifically needed for containers which are running and calls to `docker cp`
	// because both these actions require mounting the volumes.
	active int
}

// Cleanup frees resources used by the mountpoint
func (m *MountPoint) Cleanup() error {
	if m.Volume == nil || m.ID == "" {
		return nil
	}

	if err := m.Volume.Unmount(m.ID); err != nil {
		return errors.Wrapf(err, "error unmounting volume %s", m.Volume.Name())
	}

	m.active--
	if m.active == 0 {
		m.ID = ""
	}
	return nil
}

// Setup sets up a mount point by either mounting the volume if it is
// configured, or creating the source directory if supplied.
// The, optional, checkFun parameter allows doing additional checking
// before creating the source directory on the host.
func (m *MountPoint) Setup(mountLabel string, rootIDs idtools.IDPair, checkFun func(m *MountPoint) error) (path string, err error) {
	defer func() {
		if err != nil || !label.RelabelNeeded(m.Mode) {
			return
		}

		err = label.Relabel(m.Source, mountLabel, label.IsShared(m.Mode))
		if err == syscall.ENOTSUP {
			err = nil
		}
		if err != nil {
			path = ""
			err = errors.Wrapf(err, "error setting label on mount source '%s'", m.Source)
		}
	}()

	if m.Volume != nil {
		id := m.ID
		if id == "" {
			id = stringid.GenerateNonCryptoID()
		}
		path, err := m.Volume.Mount(id)
		if err != nil {
			return "", errors.Wrapf(err, "error while mounting volume '%s'", m.Source)
		}

		m.ID = id
		m.active++
		return path, nil
	}

	if len(m.Source) == 0 {
		return "", fmt.Errorf("Unable to setup mount point, neither source nor volume defined")
	}

	// system.MkdirAll() produces an error if m.Source exists and is a file (not a directory),
	if m.Type == mounttypes.TypeBind {
		// Before creating the source directory on the host, invoke checkFun if it's not nil. One of
		// the use case is to forbid creating the daemon socket as a directory if the daemon is in
		// the process of shutting down.
		if checkFun != nil {
			if err := checkFun(m); err != nil {
				return "", err
			}
		}
		// idtools.MkdirAllNewAs() produces an error if m.Source exists and is a file (not a directory)
		// also, makes sure that if the directory is created, the correct remapped rootUID/rootGID will own it
		if err := idtools.MkdirAllAndChownNew(m.Source, 0755, rootIDs); err != nil {
			if perr, ok := err.(*os.PathError); ok {
				if perr.Err != syscall.ENOTDIR {
					return "", errors.Wrapf(err, "error while creating mount source path '%s'", m.Source)
				}
			}
		}
	}
	return m.Source, nil
}

// Path returns the path of a volume in a mount point.
func (m *MountPoint) Path() string {
	if m.Volume != nil {
		return m.Volume.Path()
	}
	return m.Source
}

// ParseVolumesFrom ensures that the supplied volumes-from is valid.
func ParseVolumesFrom(spec string) (string, string, error) {
	if len(spec) == 0 {
		return "", "", fmt.Errorf("volumes-from specification cannot be an empty string")
	}

	specParts := strings.SplitN(spec, ":", 2)
	id := specParts[0]
	mode := "rw"

	if len(specParts) == 2 {
		mode = specParts[1]
		if !ValidMountMode(mode) {
			return "", "", errInvalidMode(mode)
		}
		// For now don't allow propagation properties while importing
		// volumes from data container. These volumes will inherit
		// the same propagation property as of the original volume
		// in data container. This probably can be relaxed in future.
		if HasPropagation(mode) {
			return "", "", errInvalidMode(mode)
		}
		// Do not allow copy modes on volumes-from
		if _, isSet := getCopyMode(mode); isSet {
			return "", "", errInvalidMode(mode)
		}
	}
	return id, mode, nil
}

// ParseMountRaw parses a raw volume spec (e.g. `-v /foo:/bar:shared`) into a
// structured spec. Once the raw spec is parsed it relies on `ParseMountSpec` to
// validate the spec and create a MountPoint
func ParseMountRaw(raw, volumeDriver string) (*MountPoint, error) {
	arr, err := splitRawSpec(convertSlash(raw))
	if err != nil {
		return nil, err
	}

	var spec mounttypes.Mount
	var mode string
	switch len(arr) {
	case 1:
		// Just a destination path in the container
		spec.Target = arr[0]
	case 2:
		if ValidMountMode(arr[1]) {
			// Destination + Mode is not a valid volume - volumes
			// cannot include a mode. e.g. /foo:rw
			return nil, errInvalidSpec(raw)
		}
		// Host Source Path or Name + Destination
		spec.Source = arr[0]
		spec.Target = arr[1]
	case 3:
		// HostSourcePath+DestinationPath+Mode
		spec.Source = arr[0]
		spec.Target = arr[1]
		mode = arr[2]
	default:
		return nil, errInvalidSpec(raw)
	}

	if !ValidMountMode(mode) {
		return nil, errInvalidMode(mode)
	}

	if filepath.IsAbs(spec.Source) {
		spec.Type = mounttypes.TypeBind
	} else {
		spec.Type = mounttypes.TypeVolume
	}

	spec.ReadOnly = !ReadWrite(mode)

	// cannot assume that if a volume driver is passed in that we should set it
	if volumeDriver != "" && spec.Type == mounttypes.TypeVolume {
		spec.VolumeOptions = &mounttypes.VolumeOptions{
			DriverConfig: &mounttypes.Driver{Name: volumeDriver},
		}
	}

	if copyData, isSet := getCopyMode(mode); isSet {
		if spec.VolumeOptions == nil {
			spec.VolumeOptions = &mounttypes.VolumeOptions{}
		}
		spec.VolumeOptions.NoCopy = !copyData
	}
	if HasPropagation(mode) {
		spec.BindOptions = &mounttypes.BindOptions{
			Propagation: GetPropagation(mode),
		}
	}

	mp, err := ParseMountSpec(spec, platformRawValidationOpts...)
	if mp != nil {
		mp.Mode = mode
	}
	if err != nil {
		err = fmt.Errorf("%v: %v", errInvalidSpec(raw), err)
	}
	return mp, err
}

// ParseMountSpec reads a mount config, validates it, and configures a mountpoint from it.
func ParseMountSpec(cfg mounttypes.Mount, options ...func(*validateOpts)) (*MountPoint, error) {
	if err := validateMountConfig(&cfg, options...); err != nil {
		return nil, err
	}
	mp := &MountPoint{
		RW:          !cfg.ReadOnly,
		Destination: clean(convertSlash(cfg.Target)),
		Type:        cfg.Type,
		Spec:        cfg,
	}

	switch cfg.Type {
	case mounttypes.TypeVolume:
		if cfg.Source == "" {
			mp.Name = stringid.GenerateNonCryptoID()
		} else {
			mp.Name = cfg.Source
		}
		mp.CopyData = DefaultCopyMode

		if cfg.VolumeOptions != nil {
			if cfg.VolumeOptions.DriverConfig != nil {
				mp.Driver = cfg.VolumeOptions.DriverConfig.Name
			}
			if cfg.VolumeOptions.NoCopy {
				mp.CopyData = false
			}
		}
	case mounttypes.TypeBind:
		mp.Source = clean(convertSlash(cfg.Source))
		if cfg.BindOptions != nil && len(cfg.BindOptions.Propagation) > 0 {
			mp.Propagation = cfg.BindOptions.Propagation
		} else {
			// If user did not specify a propagation mode, get
			// default propagation mode.
			mp.Propagation = DefaultPropagationMode
		}
	case mounttypes.TypeTmpfs:
		// NOP
	}
	return mp, nil
}

func errInvalidMode(mode string) error {
	return fmt.Errorf("invalid mode: %v", mode)
}

func errInvalidSpec(spec string) error {
	return fmt.Errorf("invalid volume specification: '%s'", spec)
}
