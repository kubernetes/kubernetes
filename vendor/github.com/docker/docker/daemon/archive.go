package daemon

import (
	"errors"
	"io"
	"os"
	"path/filepath"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/ioutils"
)

// ErrExtractPointNotDirectory is used to convey that the operation to extract
// a tar archive to a directory in a container has failed because the specified
// path does not refer to a directory.
var ErrExtractPointNotDirectory = errors.New("extraction point is not a directory")

// ContainerCopy performs a depracated operation of archiving the resource at
// the specified path in the conatiner identified by the given name.
func (daemon *Daemon) ContainerCopy(name string, res string) (io.ReadCloser, error) {
	container, err := daemon.Get(name)
	if err != nil {
		return nil, err
	}

	if res[0] == '/' {
		res = res[1:]
	}

	return container.Copy(res)
}

// ContainerStatPath stats the filesystem resource at the specified path in the
// container identified by the given name.
func (daemon *Daemon) ContainerStatPath(name string, path string) (stat *types.ContainerPathStat, err error) {
	container, err := daemon.Get(name)
	if err != nil {
		return nil, err
	}

	return container.StatPath(path)
}

// ContainerArchivePath creates an archive of the filesystem resource at the
// specified path in the container identified by the given name. Returns a
// tar archive of the resource and whether it was a directory or a single file.
func (daemon *Daemon) ContainerArchivePath(name string, path string) (content io.ReadCloser, stat *types.ContainerPathStat, err error) {
	container, err := daemon.Get(name)
	if err != nil {
		return nil, nil, err
	}

	return container.ArchivePath(path)
}

// ContainerExtractToDir extracts the given archive to the specified location
// in the filesystem of the container identified by the given name. The given
// path must be of a directory in the container. If it is not, the error will
// be ErrExtractPointNotDirectory. If noOverwriteDirNonDir is true then it will
// be an error if unpacking the given content would cause an existing directory
// to be replaced with a non-directory and vice versa.
func (daemon *Daemon) ContainerExtractToDir(name, path string, noOverwriteDirNonDir bool, content io.Reader) error {
	container, err := daemon.Get(name)
	if err != nil {
		return err
	}

	return container.ExtractToDir(path, noOverwriteDirNonDir, content)
}

// StatPath stats the filesystem resource at the specified path in this
// container. Returns stat info about the resource.
func (container *Container) StatPath(path string) (stat *types.ContainerPathStat, err error) {
	container.Lock()
	defer container.Unlock()

	if err = container.Mount(); err != nil {
		return nil, err
	}
	defer container.Unmount()

	err = container.mountVolumes()
	defer container.UnmountVolumes(true)
	if err != nil {
		return nil, err
	}

	// Consider the given path as an absolute path in the container.
	absPath := path
	if !filepath.IsAbs(absPath) {
		absPath = archive.PreserveTrailingDotOrSeparator(filepath.Join("/", path), path)
	}

	resolvedPath, err := container.GetResourcePath(absPath)
	if err != nil {
		return nil, err
	}

	// A trailing "." or separator has important meaning. For example, if
	// `"foo"` is a symlink to some directory `"dir"`, then `os.Lstat("foo")`
	// will stat the link itself, while `os.Lstat("foo/")` will stat the link
	// target. If the basename of the path is ".", it means to archive the
	// contents of the directory with "." as the first path component rather
	// than the name of the directory. This would cause extraction of the
	// archive to *not* make another directory, but instead use the current
	// directory.
	resolvedPath = archive.PreserveTrailingDotOrSeparator(resolvedPath, absPath)

	lstat, err := os.Lstat(resolvedPath)
	if err != nil {
		return nil, err
	}

	return &types.ContainerPathStat{
		Name:  lstat.Name(),
		Path:  absPath,
		Size:  lstat.Size(),
		Mode:  lstat.Mode(),
		Mtime: lstat.ModTime(),
	}, nil
}

// ArchivePath creates an archive of the filesystem resource at the specified
// path in this container. Returns a tar archive of the resource and stat info
// about the resource.
func (container *Container) ArchivePath(path string) (content io.ReadCloser, stat *types.ContainerPathStat, err error) {
	container.Lock()

	defer func() {
		if err != nil {
			// Wait to unlock the container until the archive is fully read
			// (see the ReadCloseWrapper func below) or if there is an error
			// before that occurs.
			container.Unlock()
		}
	}()

	if err = container.Mount(); err != nil {
		return nil, nil, err
	}

	defer func() {
		if err != nil {
			// unmount any volumes
			container.UnmountVolumes(true)
			// unmount the container's rootfs
			container.Unmount()
		}
	}()

	if err = container.mountVolumes(); err != nil {
		return nil, nil, err
	}

	// Consider the given path as an absolute path in the container.
	absPath := path
	if !filepath.IsAbs(absPath) {
		absPath = archive.PreserveTrailingDotOrSeparator(filepath.Join("/", path), path)
	}

	resolvedPath, err := container.GetResourcePath(absPath)
	if err != nil {
		return nil, nil, err
	}

	// A trailing "." or separator has important meaning. For example, if
	// `"foo"` is a symlink to some directory `"dir"`, then `os.Lstat("foo")`
	// will stat the link itself, while `os.Lstat("foo/")` will stat the link
	// target. If the basename of the path is ".", it means to archive the
	// contents of the directory with "." as the first path component rather
	// than the name of the directory. This would cause extraction of the
	// archive to *not* make another directory, but instead use the current
	// directory.
	resolvedPath = archive.PreserveTrailingDotOrSeparator(resolvedPath, absPath)

	lstat, err := os.Lstat(resolvedPath)
	if err != nil {
		return nil, nil, err
	}

	stat = &types.ContainerPathStat{
		Name:  lstat.Name(),
		Path:  absPath,
		Size:  lstat.Size(),
		Mode:  lstat.Mode(),
		Mtime: lstat.ModTime(),
	}

	data, err := archive.TarResource(resolvedPath)
	if err != nil {
		return nil, nil, err
	}

	content = ioutils.NewReadCloserWrapper(data, func() error {
		err := data.Close()
		container.UnmountVolumes(true)
		container.Unmount()
		container.Unlock()
		return err
	})

	container.LogEvent("archive-path")

	return content, stat, nil
}

// ExtractToDir extracts the given tar archive to the specified location in the
// filesystem of this container. The given path must be of a directory in the
// container. If it is not, the error will be ErrExtractPointNotDirectory. If
// noOverwriteDirNonDir is true then it will be an error if unpacking the
// given content would cause an existing directory to be replaced with a non-
// directory and vice versa.
func (container *Container) ExtractToDir(path string, noOverwriteDirNonDir bool, content io.Reader) (err error) {
	container.Lock()
	defer container.Unlock()

	if err = container.Mount(); err != nil {
		return err
	}
	defer container.Unmount()

	err = container.mountVolumes()
	defer container.UnmountVolumes(true)
	if err != nil {
		return err
	}

	// Consider the given path as an absolute path in the container.
	absPath := path
	if !filepath.IsAbs(absPath) {
		absPath = archive.PreserveTrailingDotOrSeparator(filepath.Join("/", path), path)
	}

	resolvedPath, err := container.GetResourcePath(absPath)
	if err != nil {
		return err
	}

	// A trailing "." or separator has important meaning. For example, if
	// `"foo"` is a symlink to some directory `"dir"`, then `os.Lstat("foo")`
	// will stat the link itself, while `os.Lstat("foo/")` will stat the link
	// target. If the basename of the path is ".", it means to archive the
	// contents of the directory with "." as the first path component rather
	// than the name of the directory. This would cause extraction of the
	// archive to *not* make another directory, but instead use the current
	// directory.
	resolvedPath = archive.PreserveTrailingDotOrSeparator(resolvedPath, absPath)

	stat, err := os.Lstat(resolvedPath)
	if err != nil {
		return err
	}

	if !stat.IsDir() {
		return ErrExtractPointNotDirectory
	}

	baseRel, err := filepath.Rel(container.basefs, resolvedPath)
	if err != nil {
		return err
	}
	absPath = filepath.Join("/", baseRel)

	// Need to check if the path is in a volume. If it is, it cannot be in a
	// read-only volume. If it is not in a volume, the container cannot be
	// configured with a read-only rootfs.
	var toVolume bool
	for _, mnt := range container.MountPoints {
		if toVolume = mnt.hasResource(absPath); toVolume {
			if mnt.RW {
				break
			}
			return ErrVolumeReadonly
		}
	}

	if !toVolume && container.hostConfig.ReadonlyRootfs {
		return ErrContainerRootfsReadonly
	}

	options := &archive.TarOptions{
		ChownOpts: &archive.TarChownOptions{
			UID: 0, GID: 0, // TODO: use config.User? Remap to userns root?
		},
		NoOverwriteDirNonDir: noOverwriteDirNonDir,
	}

	if err := chrootarchive.Untar(content, resolvedPath, options); err != nil {
		return err
	}

	container.LogEvent("extract-to-dir")

	return nil
}
