package archive

import (
	"archive/tar"
	"bufio"
	"bytes"
	"compress/bzip2"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/fsouza/go-dockerclient/external/github.com/Sirupsen/logrus"
	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/fileutils"
	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/idtools"
	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/ioutils"
	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/pools"
	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/promise"
	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/system"
)

type (
	// Archive is a type of io.ReadCloser which has two interfaces Read and Closer.
	Archive io.ReadCloser
	// Reader is a type of io.Reader.
	Reader io.Reader
	// Compression is the state represents if compressed or not.
	Compression int
	// TarChownOptions wraps the chown options UID and GID.
	TarChownOptions struct {
		UID, GID int
	}
	// TarOptions wraps the tar options.
	TarOptions struct {
		IncludeFiles     []string
		ExcludePatterns  []string
		Compression      Compression
		NoLchown         bool
		UIDMaps          []idtools.IDMap
		GIDMaps          []idtools.IDMap
		ChownOpts        *TarChownOptions
		IncludeSourceDir bool
		// When unpacking, specifies whether overwriting a directory with a
		// non-directory is allowed and vice versa.
		NoOverwriteDirNonDir bool
		// For each include when creating an archive, the included name will be
		// replaced with the matching name from this map.
		RebaseNames map[string]string
	}

	// Archiver allows the reuse of most utility functions of this package
	// with a pluggable Untar function. Also, to facilitate the passing of
	// specific id mappings for untar, an archiver can be created with maps
	// which will then be passed to Untar operations
	Archiver struct {
		Untar   func(io.Reader, string, *TarOptions) error
		UIDMaps []idtools.IDMap
		GIDMaps []idtools.IDMap
	}

	// breakoutError is used to differentiate errors related to breaking out
	// When testing archive breakout in the unit tests, this error is expected
	// in order for the test to pass.
	breakoutError error
)

var (
	// ErrNotImplemented is the error message of function not implemented.
	ErrNotImplemented = errors.New("Function not implemented")
	defaultArchiver   = &Archiver{Untar: Untar, UIDMaps: nil, GIDMaps: nil}
)

const (
	// HeaderSize is the size in bytes of a tar header
	HeaderSize = 512
)

const (
	// Uncompressed represents the uncompressed.
	Uncompressed Compression = iota
	// Bzip2 is bzip2 compression algorithm.
	Bzip2
	// Gzip is gzip compression algorithm.
	Gzip
	// Xz is xz compression algorithm.
	Xz
)

// IsArchive checks for the magic bytes of a tar or any supported compression
// algorithm.
func IsArchive(header []byte) bool {
	compression := DetectCompression(header)
	if compression != Uncompressed {
		return true
	}
	r := tar.NewReader(bytes.NewBuffer(header))
	_, err := r.Next()
	return err == nil
}

// IsArchivePath checks if the (possibly compressed) file at the given path
// starts with a tar file header.
func IsArchivePath(path string) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()
	rdr, err := DecompressStream(file)
	if err != nil {
		return false
	}
	r := tar.NewReader(rdr)
	_, err = r.Next()
	return err == nil
}

// DetectCompression detects the compression algorithm of the source.
func DetectCompression(source []byte) Compression {
	for compression, m := range map[Compression][]byte{
		Bzip2: {0x42, 0x5A, 0x68},
		Gzip:  {0x1F, 0x8B, 0x08},
		Xz:    {0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00},
	} {
		if len(source) < len(m) {
			logrus.Debugf("Len too short")
			continue
		}
		if bytes.Compare(m, source[:len(m)]) == 0 {
			return compression
		}
	}
	return Uncompressed
}

func xzDecompress(archive io.Reader) (io.ReadCloser, <-chan struct{}, error) {
	args := []string{"xz", "-d", "-c", "-q"}

	return cmdStream(exec.Command(args[0], args[1:]...), archive)
}

// DecompressStream decompress the archive and returns a ReaderCloser with the decompressed archive.
func DecompressStream(archive io.Reader) (io.ReadCloser, error) {
	p := pools.BufioReader32KPool
	buf := p.Get(archive)
	bs, err := buf.Peek(10)
	if err != nil && err != io.EOF {
		// Note: we'll ignore any io.EOF error because there are some odd
		// cases where the layer.tar file will be empty (zero bytes) and
		// that results in an io.EOF from the Peek() call. So, in those
		// cases we'll just treat it as a non-compressed stream and
		// that means just create an empty layer.
		// See Issue 18170
		return nil, err
	}

	compression := DetectCompression(bs)
	switch compression {
	case Uncompressed:
		readBufWrapper := p.NewReadCloserWrapper(buf, buf)
		return readBufWrapper, nil
	case Gzip:
		gzReader, err := gzip.NewReader(buf)
		if err != nil {
			return nil, err
		}
		readBufWrapper := p.NewReadCloserWrapper(buf, gzReader)
		return readBufWrapper, nil
	case Bzip2:
		bz2Reader := bzip2.NewReader(buf)
		readBufWrapper := p.NewReadCloserWrapper(buf, bz2Reader)
		return readBufWrapper, nil
	case Xz:
		xzReader, chdone, err := xzDecompress(buf)
		if err != nil {
			return nil, err
		}
		readBufWrapper := p.NewReadCloserWrapper(buf, xzReader)
		return ioutils.NewReadCloserWrapper(readBufWrapper, func() error {
			<-chdone
			return readBufWrapper.Close()
		}), nil
	default:
		return nil, fmt.Errorf("Unsupported compression format %s", (&compression).Extension())
	}
}

// CompressStream compresses the dest with specified compression algorithm.
func CompressStream(dest io.WriteCloser, compression Compression) (io.WriteCloser, error) {
	p := pools.BufioWriter32KPool
	buf := p.Get(dest)
	switch compression {
	case Uncompressed:
		writeBufWrapper := p.NewWriteCloserWrapper(buf, buf)
		return writeBufWrapper, nil
	case Gzip:
		gzWriter := gzip.NewWriter(dest)
		writeBufWrapper := p.NewWriteCloserWrapper(buf, gzWriter)
		return writeBufWrapper, nil
	case Bzip2, Xz:
		// archive/bzip2 does not support writing, and there is no xz support at all
		// However, this is not a problem as docker only currently generates gzipped tars
		return nil, fmt.Errorf("Unsupported compression format %s", (&compression).Extension())
	default:
		return nil, fmt.Errorf("Unsupported compression format %s", (&compression).Extension())
	}
}

// Extension returns the extension of a file that uses the specified compression algorithm.
func (compression *Compression) Extension() string {
	switch *compression {
	case Uncompressed:
		return "tar"
	case Bzip2:
		return "tar.bz2"
	case Gzip:
		return "tar.gz"
	case Xz:
		return "tar.xz"
	}
	return ""
}

type tarAppender struct {
	TarWriter *tar.Writer
	Buffer    *bufio.Writer

	// for hardlink mapping
	SeenFiles map[uint64]string
	UIDMaps   []idtools.IDMap
	GIDMaps   []idtools.IDMap
}

// canonicalTarName provides a platform-independent and consistent posix-style
//path for files and directories to be archived regardless of the platform.
func canonicalTarName(name string, isDir bool) (string, error) {
	name, err := CanonicalTarNameForPath(name)
	if err != nil {
		return "", err
	}

	// suffix with '/' for directories
	if isDir && !strings.HasSuffix(name, "/") {
		name += "/"
	}
	return name, nil
}

func (ta *tarAppender) addTarFile(path, name string) error {
	fi, err := os.Lstat(path)
	if err != nil {
		return err
	}

	link := ""
	if fi.Mode()&os.ModeSymlink != 0 {
		if link, err = os.Readlink(path); err != nil {
			return err
		}
	}

	hdr, err := tar.FileInfoHeader(fi, link)
	if err != nil {
		return err
	}
	hdr.Mode = int64(chmodTarEntry(os.FileMode(hdr.Mode)))

	name, err = canonicalTarName(name, fi.IsDir())
	if err != nil {
		return fmt.Errorf("tar: cannot canonicalize path: %v", err)
	}
	hdr.Name = name

	inode, err := setHeaderForSpecialDevice(hdr, ta, name, fi.Sys())
	if err != nil {
		return err
	}

	// if it's not a directory and has more than 1 link,
	// it's hardlinked, so set the type flag accordingly
	if !fi.IsDir() && hasHardlinks(fi) {
		// a link should have a name that it links too
		// and that linked name should be first in the tar archive
		if oldpath, ok := ta.SeenFiles[inode]; ok {
			hdr.Typeflag = tar.TypeLink
			hdr.Linkname = oldpath
			hdr.Size = 0 // This Must be here for the writer math to add up!
		} else {
			ta.SeenFiles[inode] = name
		}
	}

	capability, _ := system.Lgetxattr(path, "security.capability")
	if capability != nil {
		hdr.Xattrs = make(map[string]string)
		hdr.Xattrs["security.capability"] = string(capability)
	}

	//handle re-mapping container ID mappings back to host ID mappings before
	//writing tar headers/files. We skip whiteout files because they were written
	//by the kernel and already have proper ownership relative to the host
	if !strings.HasPrefix(filepath.Base(hdr.Name), WhiteoutPrefix) && (ta.UIDMaps != nil || ta.GIDMaps != nil) {
		uid, gid, err := getFileUIDGID(fi.Sys())
		if err != nil {
			return err
		}
		xUID, err := idtools.ToContainer(uid, ta.UIDMaps)
		if err != nil {
			return err
		}
		xGID, err := idtools.ToContainer(gid, ta.GIDMaps)
		if err != nil {
			return err
		}
		hdr.Uid = xUID
		hdr.Gid = xGID
	}

	if err := ta.TarWriter.WriteHeader(hdr); err != nil {
		return err
	}

	if hdr.Typeflag == tar.TypeReg {
		file, err := os.Open(path)
		if err != nil {
			return err
		}

		ta.Buffer.Reset(ta.TarWriter)
		defer ta.Buffer.Reset(nil)
		_, err = io.Copy(ta.Buffer, file)
		file.Close()
		if err != nil {
			return err
		}
		err = ta.Buffer.Flush()
		if err != nil {
			return err
		}
	}

	return nil
}

func createTarFile(path, extractDir string, hdr *tar.Header, reader io.Reader, Lchown bool, chownOpts *TarChownOptions) error {
	// hdr.Mode is in linux format, which we can use for sycalls,
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
		// Source is regular file
		file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, hdrInfo.Mode())
		if err != nil {
			return err
		}
		if _, err := io.Copy(file, reader); err != nil {
			file.Close()
			return err
		}
		file.Close()

	case tar.TypeBlock, tar.TypeChar, tar.TypeFifo:
		// Handle this is an OS-specific way
		if err := handleTarTypeBlockCharFifo(hdr, path); err != nil {
			return err
		}

	case tar.TypeLink:
		targetPath := filepath.Join(extractDir, hdr.Linkname)
		// check for hardlink breakout
		if !strings.HasPrefix(targetPath, extractDir) {
			return breakoutError(fmt.Errorf("invalid hardlink %q -> %q", targetPath, hdr.Linkname))
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
			return breakoutError(fmt.Errorf("invalid symlink %q -> %q", path, hdr.Linkname))
		}
		if err := os.Symlink(hdr.Linkname, path); err != nil {
			return err
		}

	case tar.TypeXGlobalHeader:
		logrus.Debugf("PAX Global Extended Headers found and ignored")
		return nil

	default:
		return fmt.Errorf("Unhandled tar header type %d\n", hdr.Typeflag)
	}

	// Lchown is not supported on Windows.
	if Lchown && runtime.GOOS != "windows" {
		if chownOpts == nil {
			chownOpts = &TarChownOptions{UID: hdr.Uid, GID: hdr.Gid}
		}
		if err := os.Lchown(path, chownOpts.UID, chownOpts.GID); err != nil {
			return err
		}
	}

	for key, value := range hdr.Xattrs {
		if err := system.Lsetxattr(path, key, []byte(value), 0); err != nil {
			return err
		}
	}

	// There is no LChmod, so ignore mode for symlink. Also, this
	// must happen after chown, as that can modify the file mode
	if err := handleLChmod(hdr, path, hdrInfo); err != nil {
		return err
	}

	aTime := hdr.AccessTime
	if aTime.Before(hdr.ModTime) {
		// Last access time should never be before last modified time.
		aTime = hdr.ModTime
	}

	// system.Chtimes doesn't support a NOFOLLOW flag atm
	if hdr.Typeflag == tar.TypeLink {
		if fi, err := os.Lstat(hdr.Linkname); err == nil && (fi.Mode()&os.ModeSymlink == 0) {
			if err := system.Chtimes(path, aTime, hdr.ModTime); err != nil {
				return err
			}
		}
	} else if hdr.Typeflag != tar.TypeSymlink {
		if err := system.Chtimes(path, aTime, hdr.ModTime); err != nil {
			return err
		}
	} else {
		ts := []syscall.Timespec{timeToTimespec(aTime), timeToTimespec(hdr.ModTime)}
		if err := system.LUtimesNano(path, ts); err != nil && err != system.ErrNotSupportedPlatform {
			return err
		}
	}
	return nil
}

// Tar creates an archive from the directory at `path`, and returns it as a
// stream of bytes.
func Tar(path string, compression Compression) (io.ReadCloser, error) {
	return TarWithOptions(path, &TarOptions{Compression: compression})
}

// TarWithOptions creates an archive from the directory at `path`, only including files whose relative
// paths are included in `options.IncludeFiles` (if non-nil) or not in `options.ExcludePatterns`.
func TarWithOptions(srcPath string, options *TarOptions) (io.ReadCloser, error) {

	// Fix the source path to work with long path names. This is a no-op
	// on platforms other than Windows.
	srcPath = fixVolumePathPrefix(srcPath)

	patterns, patDirs, exceptions, err := fileutils.CleanPatterns(options.ExcludePatterns)

	if err != nil {
		return nil, err
	}

	pipeReader, pipeWriter := io.Pipe()

	compressWriter, err := CompressStream(pipeWriter, options.Compression)
	if err != nil {
		return nil, err
	}

	go func() {
		ta := &tarAppender{
			TarWriter: tar.NewWriter(compressWriter),
			Buffer:    pools.BufioWriter32KPool.Get(nil),
			SeenFiles: make(map[uint64]string),
			UIDMaps:   options.UIDMaps,
			GIDMaps:   options.GIDMaps,
		}

		defer func() {
			// Make sure to check the error on Close.
			if err := ta.TarWriter.Close(); err != nil {
				logrus.Debugf("Can't close tar writer: %s", err)
			}
			if err := compressWriter.Close(); err != nil {
				logrus.Debugf("Can't close compress writer: %s", err)
			}
			if err := pipeWriter.Close(); err != nil {
				logrus.Debugf("Can't close pipe writer: %s", err)
			}
		}()

		// this buffer is needed for the duration of this piped stream
		defer pools.BufioWriter32KPool.Put(ta.Buffer)

		// In general we log errors here but ignore them because
		// during e.g. a diff operation the container can continue
		// mutating the filesystem and we can see transient errors
		// from this

		stat, err := os.Lstat(srcPath)
		if err != nil {
			return
		}

		if !stat.IsDir() {
			// We can't later join a non-dir with any includes because the
			// 'walk' will error if "file/." is stat-ed and "file" is not a
			// directory. So, we must split the source path and use the
			// basename as the include.
			if len(options.IncludeFiles) > 0 {
				logrus.Warn("Tar: Can't archive a file with includes")
			}

			dir, base := SplitPathDirEntry(srcPath)
			srcPath = dir
			options.IncludeFiles = []string{base}
		}

		if len(options.IncludeFiles) == 0 {
			options.IncludeFiles = []string{"."}
		}

		seen := make(map[string]bool)

		for _, include := range options.IncludeFiles {
			rebaseName := options.RebaseNames[include]

			walkRoot := getWalkRoot(srcPath, include)
			filepath.Walk(walkRoot, func(filePath string, f os.FileInfo, err error) error {
				if err != nil {
					logrus.Debugf("Tar: Can't stat file %s to tar: %s", srcPath, err)
					return nil
				}

				relFilePath, err := filepath.Rel(srcPath, filePath)
				if err != nil || (!options.IncludeSourceDir && relFilePath == "." && f.IsDir()) {
					// Error getting relative path OR we are looking
					// at the source directory path. Skip in both situations.
					return nil
				}

				if options.IncludeSourceDir && include == "." && relFilePath != "." {
					relFilePath = strings.Join([]string{".", relFilePath}, string(filepath.Separator))
				}

				skip := false

				// If "include" is an exact match for the current file
				// then even if there's an "excludePatterns" pattern that
				// matches it, don't skip it. IOW, assume an explicit 'include'
				// is asking for that file no matter what - which is true
				// for some files, like .dockerignore and Dockerfile (sometimes)
				if include != relFilePath {
					skip, err = fileutils.OptimizedMatches(relFilePath, patterns, patDirs)
					if err != nil {
						logrus.Debugf("Error matching %s: %v", relFilePath, err)
						return err
					}
				}

				if skip {
					if !exceptions && f.IsDir() {
						return filepath.SkipDir
					}
					return nil
				}

				if seen[relFilePath] {
					return nil
				}
				seen[relFilePath] = true

				// Rename the base resource.
				if rebaseName != "" {
					var replacement string
					if rebaseName != string(filepath.Separator) {
						// Special case the root directory to replace with an
						// empty string instead so that we don't end up with
						// double slashes in the paths.
						replacement = rebaseName
					}

					relFilePath = strings.Replace(relFilePath, include, replacement, 1)
				}

				if err := ta.addTarFile(filePath, relFilePath); err != nil {
					logrus.Debugf("Can't add file %s to tar: %s", filePath, err)
				}
				return nil
			})
		}
	}()

	return pipeReader, nil
}

// Unpack unpacks the decompressedArchive to dest with options.
func Unpack(decompressedArchive io.Reader, dest string, options *TarOptions) error {
	tr := tar.NewReader(decompressedArchive)
	trBuf := pools.BufioReader32KPool.Get(nil)
	defer pools.BufioReader32KPool.Put(trBuf)

	var dirs []*tar.Header
	remappedRootUID, remappedRootGID, err := idtools.GetRootUIDGID(options.UIDMaps, options.GIDMaps)
	if err != nil {
		return err
	}

	// Iterate through the files in the archive.
loop:
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			// end of tar archive
			break
		}
		if err != nil {
			return err
		}

		// Normalize name, for safety and for a simple is-root check
		// This keeps "../" as-is, but normalizes "/../" to "/". Or Windows:
		// This keeps "..\" as-is, but normalizes "\..\" to "\".
		hdr.Name = filepath.Clean(hdr.Name)

		for _, exclude := range options.ExcludePatterns {
			if strings.HasPrefix(hdr.Name, exclude) {
				continue loop
			}
		}

		// After calling filepath.Clean(hdr.Name) above, hdr.Name will now be in
		// the filepath format for the OS on which the daemon is running. Hence
		// the check for a slash-suffix MUST be done in an OS-agnostic way.
		if !strings.HasSuffix(hdr.Name, string(os.PathSeparator)) {
			// Not the root directory, ensure that the parent directory exists
			parent := filepath.Dir(hdr.Name)
			parentPath := filepath.Join(dest, parent)
			if _, err := os.Lstat(parentPath); err != nil && os.IsNotExist(err) {
				err = system.MkdirAll(parentPath, 0777)
				if err != nil {
					return err
				}
			}
		}

		path := filepath.Join(dest, hdr.Name)
		rel, err := filepath.Rel(dest, path)
		if err != nil {
			return err
		}
		if strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
			return breakoutError(fmt.Errorf("%q is outside of %q", hdr.Name, dest))
		}

		// If path exits we almost always just want to remove and replace it
		// The only exception is when it is a directory *and* the file from
		// the layer is also a directory. Then we want to merge them (i.e.
		// just apply the metadata from the layer).
		if fi, err := os.Lstat(path); err == nil {
			if options.NoOverwriteDirNonDir && fi.IsDir() && hdr.Typeflag != tar.TypeDir {
				// If NoOverwriteDirNonDir is true then we cannot replace
				// an existing directory with a non-directory from the archive.
				return fmt.Errorf("cannot overwrite directory %q with non-directory %q", path, dest)
			}

			if options.NoOverwriteDirNonDir && !fi.IsDir() && hdr.Typeflag == tar.TypeDir {
				// If NoOverwriteDirNonDir is true then we cannot replace
				// an existing non-directory with a directory from the archive.
				return fmt.Errorf("cannot overwrite non-directory %q with directory %q", path, dest)
			}

			if fi.IsDir() && hdr.Name == "." {
				continue
			}

			if !(fi.IsDir() && hdr.Typeflag == tar.TypeDir) {
				if err := os.RemoveAll(path); err != nil {
					return err
				}
			}
		}
		trBuf.Reset(tr)

		// if the options contain a uid & gid maps, convert header uid/gid
		// entries using the maps such that lchown sets the proper mapped
		// uid/gid after writing the file. We only perform this mapping if
		// the file isn't already owned by the remapped root UID or GID, as
		// that specific uid/gid has no mapping from container -> host, and
		// those files already have the proper ownership for inside the
		// container.
		if hdr.Uid != remappedRootUID {
			xUID, err := idtools.ToHost(hdr.Uid, options.UIDMaps)
			if err != nil {
				return err
			}
			hdr.Uid = xUID
		}
		if hdr.Gid != remappedRootGID {
			xGID, err := idtools.ToHost(hdr.Gid, options.GIDMaps)
			if err != nil {
				return err
			}
			hdr.Gid = xGID
		}

		if err := createTarFile(path, dest, hdr, trBuf, !options.NoLchown, options.ChownOpts); err != nil {
			return err
		}

		// Directory mtimes must be handled at the end to avoid further
		// file creation in them to modify the directory mtime
		if hdr.Typeflag == tar.TypeDir {
			dirs = append(dirs, hdr)
		}
	}

	for _, hdr := range dirs {
		path := filepath.Join(dest, hdr.Name)

		if err := system.Chtimes(path, hdr.AccessTime, hdr.ModTime); err != nil {
			return err
		}
	}
	return nil
}

// Untar reads a stream of bytes from `archive`, parses it as a tar archive,
// and unpacks it into the directory at `dest`.
// The archive may be compressed with one of the following algorithms:
//  identity (uncompressed), gzip, bzip2, xz.
// FIXME: specify behavior when target path exists vs. doesn't exist.
func Untar(tarArchive io.Reader, dest string, options *TarOptions) error {
	return untarHandler(tarArchive, dest, options, true)
}

// UntarUncompressed reads a stream of bytes from `archive`, parses it as a tar archive,
// and unpacks it into the directory at `dest`.
// The archive must be an uncompressed stream.
func UntarUncompressed(tarArchive io.Reader, dest string, options *TarOptions) error {
	return untarHandler(tarArchive, dest, options, false)
}

// Handler for teasing out the automatic decompression
func untarHandler(tarArchive io.Reader, dest string, options *TarOptions, decompress bool) error {
	if tarArchive == nil {
		return fmt.Errorf("Empty archive")
	}
	dest = filepath.Clean(dest)
	if options == nil {
		options = &TarOptions{}
	}
	if options.ExcludePatterns == nil {
		options.ExcludePatterns = []string{}
	}

	r := tarArchive
	if decompress {
		decompressedArchive, err := DecompressStream(tarArchive)
		if err != nil {
			return err
		}
		defer decompressedArchive.Close()
		r = decompressedArchive
	}

	return Unpack(r, dest, options)
}

// TarUntar is a convenience function which calls Tar and Untar, with the output of one piped into the other.
// If either Tar or Untar fails, TarUntar aborts and returns the error.
func (archiver *Archiver) TarUntar(src, dst string) error {
	logrus.Debugf("TarUntar(%s %s)", src, dst)
	archive, err := TarWithOptions(src, &TarOptions{Compression: Uncompressed})
	if err != nil {
		return err
	}
	defer archive.Close()

	var options *TarOptions
	if archiver.UIDMaps != nil || archiver.GIDMaps != nil {
		options = &TarOptions{
			UIDMaps: archiver.UIDMaps,
			GIDMaps: archiver.GIDMaps,
		}
	}
	return archiver.Untar(archive, dst, options)
}

// TarUntar is a convenience function which calls Tar and Untar, with the output of one piped into the other.
// If either Tar or Untar fails, TarUntar aborts and returns the error.
func TarUntar(src, dst string) error {
	return defaultArchiver.TarUntar(src, dst)
}

// UntarPath untar a file from path to a destination, src is the source tar file path.
func (archiver *Archiver) UntarPath(src, dst string) error {
	archive, err := os.Open(src)
	if err != nil {
		return err
	}
	defer archive.Close()
	var options *TarOptions
	if archiver.UIDMaps != nil || archiver.GIDMaps != nil {
		options = &TarOptions{
			UIDMaps: archiver.UIDMaps,
			GIDMaps: archiver.GIDMaps,
		}
	}
	return archiver.Untar(archive, dst, options)
}

// UntarPath is a convenience function which looks for an archive
// at filesystem path `src`, and unpacks it at `dst`.
func UntarPath(src, dst string) error {
	return defaultArchiver.UntarPath(src, dst)
}

// CopyWithTar creates a tar archive of filesystem path `src`, and
// unpacks it at filesystem path `dst`.
// The archive is streamed directly with fixed buffering and no
// intermediary disk IO.
func (archiver *Archiver) CopyWithTar(src, dst string) error {
	srcSt, err := os.Stat(src)
	if err != nil {
		return err
	}
	if !srcSt.IsDir() {
		return archiver.CopyFileWithTar(src, dst)
	}
	// Create dst, copy src's content into it
	logrus.Debugf("Creating dest directory: %s", dst)
	if err := system.MkdirAll(dst, 0755); err != nil {
		return err
	}
	logrus.Debugf("Calling TarUntar(%s, %s)", src, dst)
	return archiver.TarUntar(src, dst)
}

// CopyWithTar creates a tar archive of filesystem path `src`, and
// unpacks it at filesystem path `dst`.
// The archive is streamed directly with fixed buffering and no
// intermediary disk IO.
func CopyWithTar(src, dst string) error {
	return defaultArchiver.CopyWithTar(src, dst)
}

// CopyFileWithTar emulates the behavior of the 'cp' command-line
// for a single file. It copies a regular file from path `src` to
// path `dst`, and preserves all its metadata.
func (archiver *Archiver) CopyFileWithTar(src, dst string) (err error) {
	logrus.Debugf("CopyFileWithTar(%s, %s)", src, dst)
	srcSt, err := os.Stat(src)
	if err != nil {
		return err
	}

	if srcSt.IsDir() {
		return fmt.Errorf("Can't copy a directory")
	}

	// Clean up the trailing slash. This must be done in an operating
	// system specific manner.
	if dst[len(dst)-1] == os.PathSeparator {
		dst = filepath.Join(dst, filepath.Base(src))
	}
	// Create the holding directory if necessary
	if err := system.MkdirAll(filepath.Dir(dst), 0700); err != nil {
		return err
	}

	r, w := io.Pipe()
	errC := promise.Go(func() error {
		defer w.Close()

		srcF, err := os.Open(src)
		if err != nil {
			return err
		}
		defer srcF.Close()

		hdr, err := tar.FileInfoHeader(srcSt, "")
		if err != nil {
			return err
		}
		hdr.Name = filepath.Base(dst)
		hdr.Mode = int64(chmodTarEntry(os.FileMode(hdr.Mode)))

		remappedRootUID, remappedRootGID, err := idtools.GetRootUIDGID(archiver.UIDMaps, archiver.GIDMaps)
		if err != nil {
			return err
		}

		// only perform mapping if the file being copied isn't already owned by the
		// uid or gid of the remapped root in the container
		if remappedRootUID != hdr.Uid {
			xUID, err := idtools.ToHost(hdr.Uid, archiver.UIDMaps)
			if err != nil {
				return err
			}
			hdr.Uid = xUID
		}
		if remappedRootGID != hdr.Gid {
			xGID, err := idtools.ToHost(hdr.Gid, archiver.GIDMaps)
			if err != nil {
				return err
			}
			hdr.Gid = xGID
		}

		tw := tar.NewWriter(w)
		defer tw.Close()
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
		if _, err := io.Copy(tw, srcF); err != nil {
			return err
		}
		return nil
	})
	defer func() {
		if er := <-errC; err != nil {
			err = er
		}
	}()

	err = archiver.Untar(r, filepath.Dir(dst), nil)
	if err != nil {
		r.CloseWithError(err)
	}
	return err
}

// CopyFileWithTar emulates the behavior of the 'cp' command-line
// for a single file. It copies a regular file from path `src` to
// path `dst`, and preserves all its metadata.
//
// Destination handling is in an operating specific manner depending
// where the daemon is running. If `dst` ends with a trailing slash
// the final destination path will be `dst/base(src)`  (Linux) or
// `dst\base(src)` (Windows).
func CopyFileWithTar(src, dst string) (err error) {
	return defaultArchiver.CopyFileWithTar(src, dst)
}

// cmdStream executes a command, and returns its stdout as a stream.
// If the command fails to run or doesn't complete successfully, an error
// will be returned, including anything written on stderr.
func cmdStream(cmd *exec.Cmd, input io.Reader) (io.ReadCloser, <-chan struct{}, error) {
	chdone := make(chan struct{})
	cmd.Stdin = input
	pipeR, pipeW := io.Pipe()
	cmd.Stdout = pipeW
	var errBuf bytes.Buffer
	cmd.Stderr = &errBuf

	// Run the command and return the pipe
	if err := cmd.Start(); err != nil {
		return nil, nil, err
	}

	// Copy stdout to the returned pipe
	go func() {
		if err := cmd.Wait(); err != nil {
			pipeW.CloseWithError(fmt.Errorf("%s: %s", err, errBuf.String()))
		} else {
			pipeW.Close()
		}
		close(chdone)
	}()

	return pipeR, chdone, nil
}

// NewTempArchive reads the content of src into a temporary file, and returns the contents
// of that file as an archive. The archive can only be read once - as soon as reading completes,
// the file will be deleted.
func NewTempArchive(src Archive, dir string) (*TempArchive, error) {
	f, err := ioutil.TempFile(dir, "")
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(f, src); err != nil {
		return nil, err
	}
	if _, err := f.Seek(0, 0); err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	return &TempArchive{File: f, Size: size}, nil
}

// TempArchive is a temporary archive. The archive can only be read once - as soon as reading completes,
// the file will be deleted.
type TempArchive struct {
	*os.File
	Size   int64 // Pre-computed from Stat().Size() as a convenience
	read   int64
	closed bool
}

// Close closes the underlying file if it's still open, or does a no-op
// to allow callers to try to close the TempArchive multiple times safely.
func (archive *TempArchive) Close() error {
	if archive.closed {
		return nil
	}

	archive.closed = true

	return archive.File.Close()
}

func (archive *TempArchive) Read(data []byte) (int, error) {
	n, err := archive.File.Read(data)
	archive.read += int64(n)
	if err != nil || archive.read == archive.Size {
		archive.Close()
		os.Remove(archive.File.Name())
	}
	return n, err
}
