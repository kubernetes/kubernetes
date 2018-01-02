package remotecontext

import (
	"io"
	"os"
	"path/filepath"

	"github.com/docker/docker/builder"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/symlink"
	"github.com/docker/docker/pkg/tarsum"
	"github.com/pkg/errors"
)

type archiveContext struct {
	root string
	sums tarsum.FileInfoSums
}

func (c *archiveContext) Close() error {
	return os.RemoveAll(c.root)
}

func convertPathError(err error, cleanpath string) error {
	if err, ok := err.(*os.PathError); ok {
		err.Path = cleanpath
		return err
	}
	return err
}

type modifiableContext interface {
	builder.Source
	// Remove deletes the entry specified by `path`.
	// It is usual for directory entries to delete all its subentries.
	Remove(path string) error
}

// FromArchive returns a build source from a tar stream.
//
// It extracts the tar stream to a temporary folder that is deleted as soon as
// the Context is closed.
// As the extraction happens, a tarsum is calculated for every file, and the set of
// all those sums then becomes the source of truth for all operations on this Context.
//
// Closing tarStream has to be done by the caller.
func FromArchive(tarStream io.Reader) (builder.Source, error) {
	root, err := ioutils.TempDir("", "docker-builder")
	if err != nil {
		return nil, err
	}

	tsc := &archiveContext{root: root}

	// Make sure we clean-up upon error.  In the happy case the caller
	// is expected to manage the clean-up
	defer func() {
		if err != nil {
			tsc.Close()
		}
	}()

	decompressedStream, err := archive.DecompressStream(tarStream)
	if err != nil {
		return nil, err
	}

	sum, err := tarsum.NewTarSum(decompressedStream, true, tarsum.Version1)
	if err != nil {
		return nil, err
	}

	err = chrootarchive.Untar(sum, root, nil)
	if err != nil {
		return nil, err
	}

	tsc.sums = sum.GetSums()

	return tsc, nil
}

func (c *archiveContext) Root() string {
	return c.root
}

func (c *archiveContext) Remove(path string) error {
	_, fullpath, err := normalize(path, c.root)
	if err != nil {
		return err
	}
	return os.RemoveAll(fullpath)
}

func (c *archiveContext) Hash(path string) (string, error) {
	cleanpath, fullpath, err := normalize(path, c.root)
	if err != nil {
		return "", err
	}

	rel, err := filepath.Rel(c.root, fullpath)
	if err != nil {
		return "", convertPathError(err, cleanpath)
	}

	// Use the checksum of the followed path(not the possible symlink) because
	// this is the file that is actually copied.
	if tsInfo := c.sums.GetFile(filepath.ToSlash(rel)); tsInfo != nil {
		return tsInfo.Sum(), nil
	}
	// We set sum to path by default for the case where GetFile returns nil.
	// The usual case is if relative path is empty.
	return path, nil // backwards compat TODO: see if really needed
}

func normalize(path, root string) (cleanPath, fullPath string, err error) {
	cleanPath = filepath.Clean(string(os.PathSeparator) + path)[1:]
	fullPath, err = symlink.FollowSymlinkInScope(filepath.Join(root, path), root)
	if err != nil {
		return "", "", errors.Wrapf(err, "forbidden path outside the build context: %s (%s)", path, cleanPath)
	}
	if _, err := os.Lstat(fullPath); err != nil {
		return "", "", errors.WithStack(convertPathError(err, path))
	}
	return
}
