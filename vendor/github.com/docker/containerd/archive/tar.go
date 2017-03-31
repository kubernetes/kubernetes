package archive

import (
	"archive/tar"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/docker/containerd/fs"
	"github.com/docker/containerd/log"
	"github.com/pkg/errors"
)

var (
	bufferPool = &sync.Pool{
		New: func() interface{} {
			return make([]byte, 32*1024)
		},
	}

	breakoutError = errors.New("file name outside of root")
)

// Diff returns a tar stream of the computed filesystem
// difference between the provided directories.
//
// Produces a tar using OCI style file markers for deletions. Deleted
// files will be prepended with the prefix ".wh.". This style is
// based off AUFS whiteouts.
// See https://github.com/opencontainers/image-spec/blob/master/layer.md
func Diff(ctx context.Context, a, b string) io.ReadCloser {
	r, w := io.Pipe()

	go func() {
		var err error
		cw := newChangeWriter(w, b)
		if err = fs.Changes(ctx, a, b, cw.HandleChange); err != nil {
			err = errors.Wrap(err, "failed to create diff tar stream")
		} else {
			err = cw.Close()
		}
		if err = w.CloseWithError(err); err != nil {
			log.G(ctx).WithError(err).Debugf("closing tar pipe failed")
		}
	}()

	return r
}

const (
	// whiteoutPrefix prefix means file is a whiteout. If this is followed by a
	// filename this means that file has been removed from the base layer.
	// See https://github.com/opencontainers/image-spec/blob/master/layer.md#whiteouts
	whiteoutPrefix = ".wh."

	// whiteoutMetaPrefix prefix means whiteout has a special meaning and is not
	// for removing an actual file. Normally these files are excluded from exported
	// archives.
	whiteoutMetaPrefix = whiteoutPrefix + whiteoutPrefix

	// whiteoutLinkDir is a directory AUFS uses for storing hardlink links to other
	// layers. Normally these should not go into exported archives and all changed
	// hardlinks should be copied to the top layer.
	whiteoutLinkDir = whiteoutMetaPrefix + "plnk"

	// whiteoutOpaqueDir file means directory has been made opaque - meaning
	// readdir calls to this directory do not follow to lower layers.
	whiteoutOpaqueDir = whiteoutMetaPrefix + ".opq"
)

// Apply applies a tar stream of an OCI style diff tar.
// See https://github.com/opencontainers/image-spec/blob/master/layer.md#applying-changesets
func Apply(ctx context.Context, root string, r io.Reader) (int64, error) {
	root = filepath.Clean(root)
	fn := prepareApply()
	defer fn()

	var (
		tr   = tar.NewReader(r)
		size int64
		dirs []*tar.Header

		// Used for handling opaque directory markers which
		// may occur out of order
		unpackedPaths = make(map[string]struct{})

		// Used for aufs plink directory
		aufsTempdir   = ""
		aufsHardlinks = make(map[string]*tar.Header)
	)

	// Iterate through the files in the archive.
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			// end of tar archive
			break
		}
		if err != nil {
			return 0, err
		}

		size += hdr.Size

		// Normalize name, for safety and for a simple is-root check
		hdr.Name = filepath.Clean(hdr.Name)

		if skipFile(hdr) {
			log.G(ctx).Warnf("file %q ignored: archive may not be supported on system", hdr.Name)
			continue
		}

		// Note as these operations are platform specific, so must the slash be.
		if !strings.HasSuffix(hdr.Name, string(os.PathSeparator)) {
			// Not the root directory, ensure that the parent directory exists.
			// This happened in some tests where an image had a tarfile without any
			// parent directories.
			parent := filepath.Dir(hdr.Name)
			parentPath := filepath.Join(root, parent)

			if _, err := os.Lstat(parentPath); err != nil && os.IsNotExist(err) {
				err = mkdirAll(parentPath, 0600)
				if err != nil {
					return 0, err
				}
			}
		}

		// Skip AUFS metadata dirs
		if strings.HasPrefix(hdr.Name, whiteoutMetaPrefix) {
			// Regular files inside /.wh..wh.plnk can be used as hardlink targets
			// We don't want this directory, but we need the files in them so that
			// such hardlinks can be resolved.
			if strings.HasPrefix(hdr.Name, whiteoutLinkDir) && hdr.Typeflag == tar.TypeReg {
				basename := filepath.Base(hdr.Name)
				aufsHardlinks[basename] = hdr
				if aufsTempdir == "" {
					if aufsTempdir, err = ioutil.TempDir("", "dockerplnk"); err != nil {
						return 0, err
					}
					defer os.RemoveAll(aufsTempdir)
				}
				if err := createTarFile(ctx, filepath.Join(aufsTempdir, basename), root, hdr, tr); err != nil {
					return 0, err
				}
			}

			if hdr.Name != whiteoutOpaqueDir {
				continue
			}
		}

		path := filepath.Join(root, hdr.Name)
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return 0, err
		}

		// Note as these operations are platform specific, so must the slash be.
		if strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
			return 0, errors.Wrapf(breakoutError, "%q is outside of %q", hdr.Name, root)
		}
		base := filepath.Base(path)

		if strings.HasPrefix(base, whiteoutPrefix) {
			dir := filepath.Dir(path)
			if base == whiteoutOpaqueDir {
				_, err := os.Lstat(dir)
				if err != nil {
					return 0, err
				}
				err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
					if err != nil {
						if os.IsNotExist(err) {
							err = nil // parent was deleted
						}
						return err
					}
					if path == dir {
						return nil
					}
					if _, exists := unpackedPaths[path]; !exists {
						err := os.RemoveAll(path)
						return err
					}
					return nil
				})
				if err != nil {
					return 0, err
				}
				continue
			}

			originalBase := base[len(whiteoutPrefix):]
			originalPath := filepath.Join(dir, originalBase)
			if err := os.RemoveAll(originalPath); err != nil {
				return 0, err
			}
			continue
		}
		// If path exits we almost always just want to remove and replace it.
		// The only exception is when it is a directory *and* the file from
		// the layer is also a directory. Then we want to merge them (i.e.
		// just apply the metadata from the layer).
		if fi, err := os.Lstat(path); err == nil {
			if !(fi.IsDir() && hdr.Typeflag == tar.TypeDir) {
				if err := os.RemoveAll(path); err != nil {
					return 0, err
				}
			}
		}

		srcData := io.Reader(tr)
		srcHdr := hdr

		// Hard links into /.wh..wh.plnk don't work, as we don't extract that directory, so
		// we manually retarget these into the temporary files we extracted them into
		if hdr.Typeflag == tar.TypeLink && strings.HasPrefix(filepath.Clean(hdr.Linkname), whiteoutLinkDir) {
			linkBasename := filepath.Base(hdr.Linkname)
			srcHdr = aufsHardlinks[linkBasename]
			if srcHdr == nil {
				return 0, fmt.Errorf("Invalid aufs hardlink")
			}
			tmpFile, err := os.Open(filepath.Join(aufsTempdir, linkBasename))
			if err != nil {
				return 0, err
			}
			defer tmpFile.Close()
			srcData = tmpFile
		}

		if err := createTarFile(ctx, path, root, srcHdr, srcData); err != nil {
			return 0, err
		}

		// Directory mtimes must be handled at the end to avoid further
		// file creation in them to modify the directory mtime
		if hdr.Typeflag == tar.TypeDir {
			dirs = append(dirs, hdr)
		}
		unpackedPaths[path] = struct{}{}
	}

	for _, hdr := range dirs {
		path := filepath.Join(root, hdr.Name)
		if err := chtimes(path, boundTime(latestTime(hdr.AccessTime, hdr.ModTime)), boundTime(hdr.ModTime)); err != nil {
			return 0, err
		}
	}

	return size, nil
}

type changeWriter struct {
	tw         *tar.Writer
	source     string
	whiteoutT  time.Time
	inodeCache map[uint64]string
}

func newChangeWriter(w io.Writer, source string) *changeWriter {
	return &changeWriter{
		tw:         tar.NewWriter(w),
		source:     source,
		whiteoutT:  time.Now(),
		inodeCache: map[uint64]string{},
	}
}

func (cw *changeWriter) HandleChange(k fs.ChangeKind, p string, f os.FileInfo, err error) error {
	if err != nil {
		return err
	}
	if k == fs.ChangeKindDelete {
		whiteOutDir := filepath.Dir(p)
		whiteOutBase := filepath.Base(p)
		whiteOut := filepath.Join(whiteOutDir, whiteoutPrefix+whiteOutBase)
		hdr := &tar.Header{
			Name:       whiteOut[1:],
			Size:       0,
			ModTime:    cw.whiteoutT,
			AccessTime: cw.whiteoutT,
			ChangeTime: cw.whiteoutT,
		}
		if err := cw.tw.WriteHeader(hdr); err != nil {
			errors.Wrap(err, "failed to write whiteout header")
		}
	} else {
		var (
			link   string
			err    error
			source = filepath.Join(cw.source, p)
		)

		if f.Mode()&os.ModeSymlink != 0 {
			if link, err = os.Readlink(source); err != nil {
				return err
			}
		}

		hdr, err := tar.FileInfoHeader(f, link)
		if err != nil {
			return err
		}

		hdr.Mode = int64(chmodTarEntry(os.FileMode(hdr.Mode)))

		name := p
		if strings.HasPrefix(name, string(filepath.Separator)) {
			name, err = filepath.Rel(string(filepath.Separator), name)
			if err != nil {
				return errors.Wrap(err, "failed to make path relative")
			}
		}
		name, err = tarName(name)
		if err != nil {
			return errors.Wrap(err, "cannot canonicalize path")
		}
		// suffix with '/' for directories
		if f.IsDir() && !strings.HasSuffix(name, "/") {
			name += "/"
		}
		hdr.Name = name

		if err := setHeaderForSpecialDevice(hdr, name, f); err != nil {
			return errors.Wrap(err, "failed to set device headers")
		}

		linkname, err := fs.GetLinkSource(name, f, cw.inodeCache)
		if err != nil {
			return errors.Wrap(err, "failed to get hardlink")
		}

		if linkname != "" {
			hdr.Typeflag = tar.TypeLink
			hdr.Linkname = linkname
			hdr.Size = 0
		}

		if capability, err := getxattr(source, "security.capability"); err != nil {
			return errors.Wrap(err, "failed to get capabilities xattr")
		} else if capability != nil {
			hdr.Xattrs = map[string]string{
				"security.capability": string(capability),
			}
		}

		if err := cw.tw.WriteHeader(hdr); err != nil {
			return errors.Wrap(err, "failed to write file header")
		}

		if hdr.Typeflag == tar.TypeReg && hdr.Size > 0 {
			file, err := open(source)
			if err != nil {
				return errors.Wrapf(err, "failed to open path: %v", source)
			}
			defer file.Close()

			buf := bufferPool.Get().([]byte)
			n, err := io.CopyBuffer(cw.tw, file, buf)
			bufferPool.Put(buf)
			if err != nil {
				return errors.Wrap(err, "failed to copy")
			}
			if n != hdr.Size {
				return errors.New("short write copying file")
			}
		}
	}
	return nil
}

func (cw *changeWriter) Close() error {
	if err := cw.tw.Close(); err != nil {
		return errors.Wrap(err, "failed to close tar writer")
	}
	return nil
}

func createTarFile(ctx context.Context, path, extractDir string, hdr *tar.Header, reader io.Reader) error {
	// hdr.Mode is in linux format, which we can use for syscalls,
	// but for os.Foo() calls we need the mode converted to os.FileMode,
	// so use hdrInfo.Mode() (they differ for e.g. setuid bits)
	hdrInfo := hdr.FileInfo()

	switch hdr.Typeflag {
	case tar.TypeDir:
		// Create directory unless it exists as a directory already.
		// In that case we just want to merge the two
		if fi, err := os.Lstat(path); !(err == nil && fi.IsDir()) {
			if err := os.Mkdir(path, hdrInfo.Mode()); err != nil {
				return err
			}
		}

	case tar.TypeReg, tar.TypeRegA:
		file, err := openFile(path, os.O_CREATE|os.O_WRONLY, hdrInfo.Mode())
		if err != nil {
			return err
		}
		buf := bufferPool.Get().([]byte)
		_, err = io.CopyBuffer(file, reader, buf)
		if err1 := file.Close(); err == nil {
			err = err1
		}
		if err != nil {
			return err
		}

	case tar.TypeBlock, tar.TypeChar:
		// Handle this is an OS-specific way
		if err := handleTarTypeBlockCharFifo(hdr, path); err != nil {
			return err
		}

	case tar.TypeFifo:
		// Handle this is an OS-specific way
		if err := handleTarTypeBlockCharFifo(hdr, path); err != nil {
			return err
		}

	case tar.TypeLink:
		targetPath := filepath.Join(extractDir, hdr.Linkname)
		// check for hardlink breakout
		if !strings.HasPrefix(targetPath, extractDir) {
			return errors.Wrapf(breakoutError, "invalid hardlink %q -> %q", targetPath, hdr.Linkname)
		}
		if err := os.Link(targetPath, path); err != nil {
			return err
		}

	case tar.TypeSymlink:
		// 	path 				-> hdr.Linkname = targetPath
		// e.g. /extractDir/path/to/symlink 	-> ../2/file	= /extractDir/path/2/file
		targetPath := filepath.Join(filepath.Dir(path), hdr.Linkname)

		// the reason we don't need to check symlinks in the path (with FollowSymlinkInScope) is because
		// that symlink would first have to be created, which would be caught earlier, at this very check:
		if !strings.HasPrefix(targetPath, extractDir) {
			return errors.Wrapf(breakoutError, "invalid symlink %q -> %q", path, hdr.Linkname)
		}
		if err := os.Symlink(hdr.Linkname, path); err != nil {
			return err
		}

	case tar.TypeXGlobalHeader:
		log.G(ctx).Debug("PAX Global Extended Headers found and ignored")
		return nil

	default:
		return errors.Errorf("unhandled tar header type %d\n", hdr.Typeflag)
	}

	// Lchown is not supported on Windows.
	if runtime.GOOS != "windows" {
		if err := os.Lchown(path, hdr.Uid, hdr.Gid); err != nil {
			return err
		}
	}

	for key, value := range hdr.Xattrs {
		if err := setxattr(path, key, value); err != nil {
			if errors.Cause(err) == syscall.ENOTSUP {
				log.G(ctx).WithError(err).Warnf("ignored xattr %s in archive", key)
				continue
			}
			return err
		}
	}

	// There is no LChmod, so ignore mode for symlink. Also, this
	// must happen after chown, as that can modify the file mode
	if err := handleLChmod(hdr, path, hdrInfo); err != nil {
		return err
	}

	if err := chtimes(path, boundTime(latestTime(hdr.AccessTime, hdr.ModTime)), boundTime(hdr.ModTime)); err != nil {
		return err
	}

	return nil
}
