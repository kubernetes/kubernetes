package daemon

import (
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/container"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/system"
	"github.com/pkg/errors"
)

// ErrExtractPointNotDirectory is used to convey that the operation to extract
// a tar archive to a directory in a container has failed because the specified
// path does not refer to a directory.
var ErrExtractPointNotDirectory = errors.New("extraction point is not a directory")

// ContainerCopy performs a deprecated operation of archiving the resource at
// the specified path in the container identified by the given name.
func (daemon *Daemon) ContainerCopy(name string, res string) (io.ReadCloser, error) {
	container, err := daemon.GetContainer(name)
	if err != nil {
		return nil, err
	}

	if res[0] == '/' || res[0] == '\\' {
		res = res[1:]
	}

	// Make sure an online file-system operation is permitted.
	if err := daemon.isOnlineFSOperationPermitted(container); err != nil {
		return nil, err
	}

	return daemon.containerCopy(container, res)
}

// ContainerStatPath stats the filesystem resource at the specified path in the
// container identified by the given name.
func (daemon *Daemon) ContainerStatPath(name string, path string) (stat *types.ContainerPathStat, err error) {
	container, err := daemon.GetContainer(name)
	if err != nil {
		return nil, err
	}

	// Make sure an online file-system operation is permitted.
	if err := daemon.isOnlineFSOperationPermitted(container); err != nil {
		return nil, err
	}

	return daemon.containerStatPath(container, path)
}

// ContainerArchivePath creates an archive of the filesystem resource at the
// specified path in the container identified by the given name. Returns a
// tar archive of the resource and whether it was a directory or a single file.
func (daemon *Daemon) ContainerArchivePath(name string, path string) (content io.ReadCloser, stat *types.ContainerPathStat, err error) {
	container, err := daemon.GetContainer(name)
	if err != nil {
		return nil, nil, err
	}

	// Make sure an online file-system operation is permitted.
	if err := daemon.isOnlineFSOperationPermitted(container); err != nil {
		return nil, nil, err
	}

	return daemon.containerArchivePath(container, path)
}

// ContainerExtractToDir extracts the given archive to the specified location
// in the filesystem of the container identified by the given name. The given
// path must be of a directory in the container. If it is not, the error will
// be ErrExtractPointNotDirectory. If noOverwriteDirNonDir is true then it will
// be an error if unpacking the given content would cause an existing directory
// to be replaced with a non-directory and vice versa.
func (daemon *Daemon) ContainerExtractToDir(name, path string, copyUIDGID, noOverwriteDirNonDir bool, content io.Reader) error {
	container, err := daemon.GetContainer(name)
	if err != nil {
		return err
	}

	// Make sure an online file-system operation is permitted.
	if err := daemon.isOnlineFSOperationPermitted(container); err != nil {
		return err
	}

	return daemon.containerExtractToDir(container, path, copyUIDGID, noOverwriteDirNonDir, content)
}

// containerStatPath stats the filesystem resource at the specified path in this
// container. Returns stat info about the resource.
func (daemon *Daemon) containerStatPath(container *container.Container, path string) (stat *types.ContainerPathStat, err error) {
	container.Lock()
	defer container.Unlock()

	if err = daemon.Mount(container); err != nil {
		return nil, err
	}
	defer daemon.Unmount(container)

	err = daemon.mountVolumes(container)
	defer container.DetachAndUnmount(daemon.LogVolumeEvent)
	if err != nil {
		return nil, err
	}

	resolvedPath, absPath, err := container.ResolvePath(path)
	if err != nil {
		return nil, err
	}

	return container.StatPath(resolvedPath, absPath)
}

// containerArchivePath creates an archive of the filesystem resource at the specified
// path in this container. Returns a tar archive of the resource and stat info
// about the resource.
func (daemon *Daemon) containerArchivePath(container *container.Container, path string) (content io.ReadCloser, stat *types.ContainerPathStat, err error) {
	container.Lock()

	defer func() {
		if err != nil {
			// Wait to unlock the container until the archive is fully read
			// (see the ReadCloseWrapper func below) or if there is an error
			// before that occurs.
			container.Unlock()
		}
	}()

	if err = daemon.Mount(container); err != nil {
		return nil, nil, err
	}

	defer func() {
		if err != nil {
			// unmount any volumes
			container.DetachAndUnmount(daemon.LogVolumeEvent)
			// unmount the container's rootfs
			daemon.Unmount(container)
		}
	}()

	if err = daemon.mountVolumes(container); err != nil {
		return nil, nil, err
	}

	resolvedPath, absPath, err := container.ResolvePath(path)
	if err != nil {
		return nil, nil, err
	}

	stat, err = container.StatPath(resolvedPath, absPath)
	if err != nil {
		return nil, nil, err
	}

	// We need to rebase the archive entries if the last element of the
	// resolved path was a symlink that was evaluated and is now different
	// than the requested path. For example, if the given path was "/foo/bar/",
	// but it resolved to "/var/lib/docker/containers/{id}/foo/baz/", we want
	// to ensure that the archive entries start with "bar" and not "baz". This
	// also catches the case when the root directory of the container is
	// requested: we want the archive entries to start with "/" and not the
	// container ID.
	data, err := archive.TarResourceRebase(resolvedPath, filepath.Base(absPath))
	if err != nil {
		return nil, nil, err
	}

	content = ioutils.NewReadCloserWrapper(data, func() error {
		err := data.Close()
		container.DetachAndUnmount(daemon.LogVolumeEvent)
		daemon.Unmount(container)
		container.Unlock()
		return err
	})

	daemon.LogContainerEvent(container, "archive-path")

	return content, stat, nil
}

// containerExtractToDir extracts the given tar archive to the specified location in the
// filesystem of this container. The given path must be of a directory in the
// container. If it is not, the error will be ErrExtractPointNotDirectory. If
// noOverwriteDirNonDir is true then it will be an error if unpacking the
// given content would cause an existing directory to be replaced with a non-
// directory and vice versa.
func (daemon *Daemon) containerExtractToDir(container *container.Container, path string, copyUIDGID, noOverwriteDirNonDir bool, content io.Reader) (err error) {
	container.Lock()
	defer container.Unlock()

	if err = daemon.Mount(container); err != nil {
		return err
	}
	defer daemon.Unmount(container)

	err = daemon.mountVolumes(container)
	defer container.DetachAndUnmount(daemon.LogVolumeEvent)
	if err != nil {
		return err
	}

	// Check if a drive letter supplied, it must be the system drive. No-op except on Windows
	path, err = system.CheckSystemDriveAndRemoveDriveLetter(path)
	if err != nil {
		return err
	}

	// The destination path needs to be resolved to a host path, with all
	// symbolic links followed in the scope of the container's rootfs. Note
	// that we do not use `container.ResolvePath(path)` here because we need
	// to also evaluate the last path element if it is a symlink. This is so
	// that you can extract an archive to a symlink that points to a directory.

	// Consider the given path as an absolute path in the container.
	absPath := archive.PreserveTrailingDotOrSeparator(filepath.Join(string(filepath.Separator), path), path)

	// This will evaluate the last path element if it is a symlink.
	resolvedPath, err := container.GetResourcePath(absPath)
	if err != nil {
		return err
	}

	stat, err := os.Lstat(resolvedPath)
	if err != nil {
		return err
	}

	if !stat.IsDir() {
		return ErrExtractPointNotDirectory
	}

	// Need to check if the path is in a volume. If it is, it cannot be in a
	// read-only volume. If it is not in a volume, the container cannot be
	// configured with a read-only rootfs.

	// Use the resolved path relative to the container rootfs as the new
	// absPath. This way we fully follow any symlinks in a volume that may
	// lead back outside the volume.
	//
	// The Windows implementation of filepath.Rel in golang 1.4 does not
	// support volume style file path semantics. On Windows when using the
	// filter driver, we are guaranteed that the path will always be
	// a volume file path.
	var baseRel string
	if strings.HasPrefix(resolvedPath, `\\?\Volume{`) {
		if strings.HasPrefix(resolvedPath, container.BaseFS) {
			baseRel = resolvedPath[len(container.BaseFS):]
			if baseRel[:1] == `\` {
				baseRel = baseRel[1:]
			}
		}
	} else {
		baseRel, err = filepath.Rel(container.BaseFS, resolvedPath)
	}
	if err != nil {
		return err
	}
	// Make it an absolute path.
	absPath = filepath.Join(string(filepath.Separator), baseRel)

	toVolume, err := checkIfPathIsInAVolume(container, absPath)
	if err != nil {
		return err
	}

	if !toVolume && container.HostConfig.ReadonlyRootfs {
		return ErrRootFSReadOnly
	}

	options := daemon.defaultTarCopyOptions(noOverwriteDirNonDir)

	if copyUIDGID {
		var err error
		// tarCopyOptions will appropriately pull in the right uid/gid for the
		// user/group and will set the options.
		options, err = daemon.tarCopyOptions(container, noOverwriteDirNonDir)
		if err != nil {
			return err
		}
	}

	if err := chrootarchive.Untar(content, resolvedPath, options); err != nil {
		return err
	}

	daemon.LogContainerEvent(container, "extract-to-dir")

	return nil
}

func (daemon *Daemon) containerCopy(container *container.Container, resource string) (rc io.ReadCloser, err error) {
	container.Lock()

	defer func() {
		if err != nil {
			// Wait to unlock the container until the archive is fully read
			// (see the ReadCloseWrapper func below) or if there is an error
			// before that occurs.
			container.Unlock()
		}
	}()

	if err := daemon.Mount(container); err != nil {
		return nil, err
	}

	defer func() {
		if err != nil {
			// unmount any volumes
			container.DetachAndUnmount(daemon.LogVolumeEvent)
			// unmount the container's rootfs
			daemon.Unmount(container)
		}
	}()

	if err := daemon.mountVolumes(container); err != nil {
		return nil, err
	}

	basePath, err := container.GetResourcePath(resource)
	if err != nil {
		return nil, err
	}
	stat, err := os.Stat(basePath)
	if err != nil {
		return nil, err
	}
	var filter []string
	if !stat.IsDir() {
		d, f := filepath.Split(basePath)
		basePath = d
		filter = []string{f}
	} else {
		filter = []string{filepath.Base(basePath)}
		basePath = filepath.Dir(basePath)
	}
	archive, err := archive.TarWithOptions(basePath, &archive.TarOptions{
		Compression:  archive.Uncompressed,
		IncludeFiles: filter,
	})
	if err != nil {
		return nil, err
	}

	reader := ioutils.NewReadCloserWrapper(archive, func() error {
		err := archive.Close()
		container.DetachAndUnmount(daemon.LogVolumeEvent)
		daemon.Unmount(container)
		container.Unlock()
		return err
	})
	daemon.LogContainerEvent(container, "copy")
	return reader, nil
}
