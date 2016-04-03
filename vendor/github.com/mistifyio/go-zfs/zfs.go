// Package zfs provides wrappers around the ZFS command line tools.
package zfs

import (
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// ZFS dataset types, which can indicate if a dataset is a filesystem,
// snapshot, or volume.
const (
	DatasetFilesystem = "filesystem"
	DatasetSnapshot   = "snapshot"
	DatasetVolume     = "volume"
)

// Dataset is a ZFS dataset.  A dataset could be a clone, filesystem, snapshot,
// or volume.  The Type struct member can be used to determine a dataset's type.
//
// The field definitions can be found in the ZFS manual:
// http://www.freebsd.org/cgi/man.cgi?zfs(8).
type Dataset struct {
	Name          string
	Origin        string
	Used          uint64
	Avail         uint64
	Mountpoint    string
	Compression   string
	Type          string
	Written       uint64
	Volsize       uint64
	Usedbydataset uint64
	Logicalused   uint64
	Quota         uint64
}

// InodeType is the type of inode as reported by Diff
type InodeType int

// Types of Inodes
const (
	_                     = iota // 0 == unknown type
	BlockDevice InodeType = iota
	CharacterDevice
	Directory
	Door
	NamedPipe
	SymbolicLink
	EventPort
	Socket
	File
)

// ChangeType is the type of inode change as reported by Diff
type ChangeType int

// Types of Changes
const (
	_                  = iota // 0 == unknown type
	Removed ChangeType = iota
	Created
	Modified
	Renamed
)

// DestroyFlag is the options flag passed to Destroy
type DestroyFlag int

// Valid destroy options
const (
	DestroyDefault         DestroyFlag = 1 << iota
	DestroyRecursive                   = 1 << iota
	DestroyRecursiveClones             = 1 << iota
	DestroyDeferDeletion               = 1 << iota
	DestroyForceUmount                 = 1 << iota
)

// InodeChange represents a change as reported by Diff
type InodeChange struct {
	Change               ChangeType
	Type                 InodeType
	Path                 string
	NewPath              string
	ReferenceCountChange int
}

// Logger can be used to log commands/actions
type Logger interface {
	Log(cmd []string)
}

type defaultLogger struct{}

func (*defaultLogger) Log(cmd []string) {
	return
}

var logger Logger = &defaultLogger{}

// SetLogger set a log handler to log all commands including arguments before
// they are executed
func SetLogger(l Logger) {
	if l != nil {
		logger = l
	}
}

// zfs is a helper function to wrap typical calls to zfs.
func zfs(arg ...string) ([][]string, error) {
	c := command{Command: "zfs"}
	return c.Run(arg...)
}

// Datasets returns a slice of ZFS datasets, regardless of type.
// A filter argument may be passed to select a dataset with the matching name,
// or empty string ("") may be used to select all datasets.
func Datasets(filter string) ([]*Dataset, error) {
	return listByType("all", filter)
}

// Snapshots returns a slice of ZFS snapshots.
// A filter argument may be passed to select a snapshot with the matching name,
// or empty string ("") may be used to select all snapshots.
func Snapshots(filter string) ([]*Dataset, error) {
	return listByType(DatasetSnapshot, filter)
}

// Filesystems returns a slice of ZFS filesystems.
// A filter argument may be passed to select a filesystem with the matching name,
// or empty string ("") may be used to select all filesystems.
func Filesystems(filter string) ([]*Dataset, error) {
	return listByType(DatasetFilesystem, filter)
}

// Volumes returns a slice of ZFS volumes.
// A filter argument may be passed to select a volume with the matching name,
// or empty string ("") may be used to select all volumes.
func Volumes(filter string) ([]*Dataset, error) {
	return listByType(DatasetVolume, filter)
}

// GetDataset retrieves a single ZFS dataset by name.  This dataset could be
// any valid ZFS dataset type, such as a clone, filesystem, snapshot, or volume.
func GetDataset(name string) (*Dataset, error) {
	out, err := zfs("get", "-Hp", "all", name)
	if err != nil {
		return nil, err
	}

	ds := &Dataset{Name: name}
	for _, line := range out {
		if err := ds.parseLine(line); err != nil {
			return nil, err
		}
	}

	return ds, nil
}

// Clone clones a ZFS snapshot and returns a clone dataset.
// An error will be returned if the input dataset is not of snapshot type.
func (d *Dataset) Clone(dest string, properties map[string]string) (*Dataset, error) {
	if d.Type != DatasetSnapshot {
		return nil, errors.New("can only clone snapshots")
	}
	args := make([]string, 2, 4)
	args[0] = "clone"
	args[1] = "-p"
	if properties != nil {
		args = append(args, propsSlice(properties)...)
	}
	args = append(args, []string{d.Name, dest}...)
	_, err := zfs(args...)
	if err != nil {
		return nil, err
	}
	return GetDataset(dest)
}

// ReceiveSnapshot receives a ZFS stream from the input io.Reader, creates a
// new snapshot with the specified name, and streams the input data into the
// newly-created snapshot.
func ReceiveSnapshot(input io.Reader, name string) (*Dataset, error) {
	c := command{Command: "zfs", Stdin: input}
	_, err := c.Run("receive", name)
	if err != nil {
		return nil, err
	}
	return GetDataset(name)
}

// SendSnapshot sends a ZFS stream of a snapshot to the input io.Writer.
// An error will be returned if the input dataset is not of snapshot type.
func (d *Dataset) SendSnapshot(output io.Writer) error {
	if d.Type != DatasetSnapshot {
		return errors.New("can only send snapshots")
	}

	c := command{Command: "zfs", Stdout: output}
	_, err := c.Run("send", d.Name)
	return err
}

// CreateVolume creates a new ZFS volume with the specified name, size, and
// properties.
// A full list of available ZFS properties may be found here:
// https://www.freebsd.org/cgi/man.cgi?zfs(8).
func CreateVolume(name string, size uint64, properties map[string]string) (*Dataset, error) {
	args := make([]string, 4, 5)
	args[0] = "create"
	args[1] = "-p"
	args[2] = "-V"
	args[3] = strconv.FormatUint(size, 10)
	if properties != nil {
		args = append(args, propsSlice(properties)...)
	}
	args = append(args, name)
	_, err := zfs(args...)
	if err != nil {
		return nil, err
	}
	return GetDataset(name)
}

// Destroy destroys a ZFS dataset. If the destroy bit flag is set, any
// descendents of the dataset will be recursively destroyed, including snapshots.
// If the deferred bit flag is set, the snapshot is marked for deferred
// deletion.
func (d *Dataset) Destroy(flags DestroyFlag) error {
	args := make([]string, 1, 3)
	args[0] = "destroy"
	if flags&DestroyRecursive != 0 {
		args = append(args, "-r")
	}

	if flags&DestroyRecursiveClones != 0 {
		args = append(args, "-R")
	}

	if flags&DestroyDeferDeletion != 0 {
		args = append(args, "-d")
	}

	if flags&DestroyForceUmount != 0 {
		args = append(args, "-f")
	}

	args = append(args, d.Name)
	_, err := zfs(args...)
	return err
}

// SetProperty sets a ZFS property on the receiving dataset.
// A full list of available ZFS properties may be found here:
// https://www.freebsd.org/cgi/man.cgi?zfs(8).
func (d *Dataset) SetProperty(key, val string) error {
	prop := strings.Join([]string{key, val}, "=")
	_, err := zfs("set", prop, d.Name)
	return err
}

// GetProperty returns the current value of a ZFS property from the
// receiving dataset.
// A full list of available ZFS properties may be found here:
// https://www.freebsd.org/cgi/man.cgi?zfs(8).
func (d *Dataset) GetProperty(key string) (string, error) {
	out, err := zfs("get", key, d.Name)
	if err != nil {
		return "", err
	}

	return out[0][2], nil
}

// Snapshots returns a slice of all ZFS snapshots of a given dataset.
func (d *Dataset) Snapshots() ([]*Dataset, error) {
	return Snapshots(d.Name)
}

// CreateFilesystem creates a new ZFS filesystem with the specified name and
// properties.
// A full list of available ZFS properties may be found here:
// https://www.freebsd.org/cgi/man.cgi?zfs(8).
func CreateFilesystem(name string, properties map[string]string) (*Dataset, error) {
	args := make([]string, 1, 4)
	args[0] = "create"

	if properties != nil {
		args = append(args, propsSlice(properties)...)
	}

	args = append(args, name)
	_, err := zfs(args...)
	if err != nil {
		return nil, err
	}
	return GetDataset(name)
}

// Snapshot creates a new ZFS snapshot of the receiving dataset, using the
// specified name.  Optionally, the snapshot can be taken recursively, creating
// snapshots of all descendent filesystems in a single, atomic operation.
func (d *Dataset) Snapshot(name string, recursive bool) (*Dataset, error) {
	args := make([]string, 1, 4)
	args[0] = "snapshot"
	if recursive {
		args = append(args, "-r")
	}
	snapName := fmt.Sprintf("%s@%s", d.Name, name)
	args = append(args, snapName)
	_, err := zfs(args...)
	if err != nil {
		return nil, err
	}
	return GetDataset(snapName)
}

// Rollback rolls back the receiving ZFS dataset to a previous snapshot.
// Optionally, intermediate snapshots can be destroyed.  A ZFS snapshot
// rollback cannot be completed without this option, if more recent
// snapshots exist.
// An error will be returned if the input dataset is not of snapshot type.
func (d *Dataset) Rollback(destroyMoreRecent bool) error {
	if d.Type != DatasetSnapshot {
		return errors.New("can only rollback snapshots")
	}

	args := make([]string, 1, 3)
	args[0] = "rollback"
	if destroyMoreRecent {
		args = append(args, "-r")
	}
	args = append(args, d.Name)

	_, err := zfs(args...)
	return err
}

// Children returns a slice of children of the receiving ZFS dataset.
// A recursion depth may be specified, or a depth of 0 allows unlimited
// recursion.
func (d *Dataset) Children(depth uint64) ([]*Dataset, error) {
	args := []string{"get", "-t", "all", "-Hp", "all"}
	if depth > 0 {
		args = append(args, "-d")
		args = append(args, strconv.FormatUint(depth, 10))
	} else {
		args = append(args, "-r")
	}
	args = append(args, d.Name)

	out, err := zfs(args...)
	if err != nil {
		return nil, err
	}

	var datasets []*Dataset
	name := ""
	var ds *Dataset
	for _, line := range out {
		if name != line[0] {
			name = line[0]
			ds = &Dataset{Name: name}
			datasets = append(datasets, ds)
		}
		if err := ds.parseLine(line); err != nil {
			return nil, err
		}
	}
	return datasets[1:], nil
}

// Diff returns changes between a snapshot and the given ZFS dataset.
// The snapshot name must include the filesystem part as it is possible to
// compare clones with their origin snapshots.
func (d *Dataset) Diff(snapshot string) ([]*InodeChange, error) {
	args := []string{"diff", "-FH", snapshot, d.Name}[:]
	out, err := zfs(args...)
	if err != nil {
		return nil, err
	}
	inodeChanges, err := parseInodeChanges(out)
	if err != nil {
		return nil, err
	}
	return inodeChanges, nil
}
