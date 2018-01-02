package archive

import (
	"archive/tar"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/pools"
	"github.com/docker/docker/pkg/system"
	"github.com/sirupsen/logrus"
)

// UnpackLayer unpack `layer` to a `dest`. The stream `layer` can be
// compressed or uncompressed.
// Returns the size in bytes of the contents of the layer.
func UnpackLayer(dest string, layer io.Reader, options *TarOptions) (size int64, err error) {
	tr := tar.NewReader(layer)
	trBuf := pools.BufioReader32KPool.Get(tr)
	defer pools.BufioReader32KPool.Put(trBuf)

	var dirs []*tar.Header
	unpackedPaths := make(map[string]struct{})

	if options == nil {
		options = &TarOptions{}
	}
	if options.ExcludePatterns == nil {
		options.ExcludePatterns = []string{}
	}
	idMappings := idtools.NewIDMappingsFromMaps(options.UIDMaps, options.GIDMaps)

	aufsTempdir := ""
	aufsHardlinks := make(map[string]*tar.Header)

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

		// Windows does not support filenames with colons in them. Ignore
		// these files. This is not a problem though (although it might
		// appear that it is). Let's suppose a client is running docker pull.
		// The daemon it points to is Windows. Would it make sense for the
		// client to be doing a docker pull Ubuntu for example (which has files
		// with colons in the name under /usr/share/man/man3)? No, absolutely
		// not as it would really only make sense that they were pulling a
		// Windows image. However, for development, it is necessary to be able
		// to pull Linux images which are in the repository.
		//
		// TODO Windows. Once the registry is aware of what images are Windows-
		// specific or Linux-specific, this warning should be changed to an error
		// to cater for the situation where someone does manage to upload a Linux
		// image but have it tagged as Windows inadvertently.
		if runtime.GOOS == "windows" {
			if strings.Contains(hdr.Name, ":") {
				logrus.Warnf("Windows: Ignoring %s (is this a Linux image?)", hdr.Name)
				continue
			}
		}

		// Note as these operations are platform specific, so must the slash be.
		if !strings.HasSuffix(hdr.Name, string(os.PathSeparator)) {
			// Not the root directory, ensure that the parent directory exists.
			// This happened in some tests where an image had a tarfile without any
			// parent directories.
			parent := filepath.Dir(hdr.Name)
			parentPath := filepath.Join(dest, parent)

			if _, err := os.Lstat(parentPath); err != nil && os.IsNotExist(err) {
				err = system.MkdirAll(parentPath, 0600, "")
				if err != nil {
					return 0, err
				}
			}
		}

		// Skip AUFS metadata dirs
		if strings.HasPrefix(hdr.Name, WhiteoutMetaPrefix) {
			// Regular files inside /.wh..wh.plnk can be used as hardlink targets
			// We don't want this directory, but we need the files in them so that
			// such hardlinks can be resolved.
			if strings.HasPrefix(hdr.Name, WhiteoutLinkDir) && hdr.Typeflag == tar.TypeReg {
				basename := filepath.Base(hdr.Name)
				aufsHardlinks[basename] = hdr
				if aufsTempdir == "" {
					if aufsTempdir, err = ioutil.TempDir("", "dockerplnk"); err != nil {
						return 0, err
					}
					defer os.RemoveAll(aufsTempdir)
				}
				if err := createTarFile(filepath.Join(aufsTempdir, basename), dest, hdr, tr, true, nil, options.InUserNS); err != nil {
					return 0, err
				}
			}

			if hdr.Name != WhiteoutOpaqueDir {
				continue
			}
		}
		path := filepath.Join(dest, hdr.Name)
		rel, err := filepath.Rel(dest, path)
		if err != nil {
			return 0, err
		}

		// Note as these operations are platform specific, so must the slash be.
		if strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
			return 0, breakoutError(fmt.Errorf("%q is outside of %q", hdr.Name, dest))
		}
		base := filepath.Base(path)

		if strings.HasPrefix(base, WhiteoutPrefix) {
			dir := filepath.Dir(path)
			if base == WhiteoutOpaqueDir {
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
			} else {
				originalBase := base[len(WhiteoutPrefix):]
				originalPath := filepath.Join(dir, originalBase)
				if err := os.RemoveAll(originalPath); err != nil {
					return 0, err
				}
			}
		} else {
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

			trBuf.Reset(tr)
			srcData := io.Reader(trBuf)
			srcHdr := hdr

			// Hard links into /.wh..wh.plnk don't work, as we don't extract that directory, so
			// we manually retarget these into the temporary files we extracted them into
			if hdr.Typeflag == tar.TypeLink && strings.HasPrefix(filepath.Clean(hdr.Linkname), WhiteoutLinkDir) {
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

			if err := remapIDs(idMappings, srcHdr); err != nil {
				return 0, err
			}

			if err := createTarFile(path, dest, srcHdr, srcData, true, nil, options.InUserNS); err != nil {
				return 0, err
			}

			// Directory mtimes must be handled at the end to avoid further
			// file creation in them to modify the directory mtime
			if hdr.Typeflag == tar.TypeDir {
				dirs = append(dirs, hdr)
			}
			unpackedPaths[path] = struct{}{}
		}
	}

	for _, hdr := range dirs {
		path := filepath.Join(dest, hdr.Name)
		if err := system.Chtimes(path, hdr.AccessTime, hdr.ModTime); err != nil {
			return 0, err
		}
	}

	return size, nil
}

// ApplyLayer parses a diff in the standard layer format from `layer`,
// and applies it to the directory `dest`. The stream `layer` can be
// compressed or uncompressed.
// Returns the size in bytes of the contents of the layer.
func ApplyLayer(dest string, layer io.Reader) (int64, error) {
	return applyLayerHandler(dest, layer, &TarOptions{}, true)
}

// ApplyUncompressedLayer parses a diff in the standard layer format from
// `layer`, and applies it to the directory `dest`. The stream `layer`
// can only be uncompressed.
// Returns the size in bytes of the contents of the layer.
func ApplyUncompressedLayer(dest string, layer io.Reader, options *TarOptions) (int64, error) {
	return applyLayerHandler(dest, layer, options, false)
}

// do the bulk load of ApplyLayer, but allow for not calling DecompressStream
func applyLayerHandler(dest string, layer io.Reader, options *TarOptions, decompress bool) (int64, error) {
	dest = filepath.Clean(dest)

	// We need to be able to set any perms
	oldmask, err := system.Umask(0)
	if err != nil {
		return 0, err
	}
	defer system.Umask(oldmask) // ignore err, ErrNotSupportedPlatform

	if decompress {
		layer, err = DecompressStream(layer)
		if err != nil {
			return 0, err
		}
	}
	return UnpackLayer(dest, layer, options)
}
