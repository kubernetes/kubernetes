package archive

import (
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
	"path"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/docker/docker/vendor/src/code.google.com/p/go/src/pkg/archive/tar"

	log "github.com/Sirupsen/logrus"
	"github.com/docker/docker/pkg/fileutils"
	"github.com/docker/docker/pkg/pools"
	"github.com/docker/docker/pkg/promise"
	"github.com/docker/docker/pkg/system"
)

type (
	Archive       io.ReadCloser
	ArchiveReader io.Reader
	Compression   int
	TarOptions    struct {
		IncludeFiles    []string
		ExcludePatterns []string
		Compression     Compression
		NoLchown        bool
		Name            string
	}

	// Archiver allows the reuse of most utility functions of this package
	// with a pluggable Untar function.
	Archiver struct {
		Untar func(io.Reader, string, *TarOptions) error
	}

	// breakoutError is used to differentiate errors related to breaking out
	// When testing archive breakout in the unit tests, this error is expected
	// in order for the test to pass.
	breakoutError error
)

var (
	ErrNotImplemented = errors.New("Function not implemented")
	defaultArchiver   = &Archiver{Untar}
)

const (
	Uncompressed Compression = iota
	Bzip2
	Gzip
	Xz
)

func IsArchive(header []byte) bool {
	compression := DetectCompression(header)
	if compression != Uncompressed {
		return true
	}
	r := tar.NewReader(bytes.NewBuffer(header))
	_, err := r.Next()
	return err == nil
}

func DetectCompression(source []byte) Compression {
	for compression, m := range map[Compression][]byte{
		Bzip2: {0x42, 0x5A, 0x68},
		Gzip:  {0x1F, 0x8B, 0x08},
		Xz:    {0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00},
	} {
		if len(source) < len(m) {
			log.Debugf("Len too short")
			continue
		}
		if bytes.Compare(m, source[:len(m)]) == 0 {
			return compression
		}
	}
	return Uncompressed
}

func xzDecompress(archive io.Reader) (io.ReadCloser, error) {
	args := []string{"xz", "-d", "-c", "-q"}

	return CmdStream(exec.Command(args[0], args[1:]...), archive)
}

func DecompressStream(archive io.Reader) (io.ReadCloser, error) {
	p := pools.BufioReader32KPool
	buf := p.Get(archive)
	bs, err := buf.Peek(10)
	if err != nil {
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
		xzReader, err := xzDecompress(buf)
		if err != nil {
			return nil, err
		}
		readBufWrapper := p.NewReadCloserWrapper(buf, xzReader)
		return readBufWrapper, nil
	default:
		return nil, fmt.Errorf("Unsupported compression format %s", (&compression).Extension())
	}
}

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

	nlink, inode, err := setHeaderForSpecialDevice(hdr, ta, name, fi.Sys())
	if err != nil {
		return err
	}

	// if it's a regular file and has more than 1 link,
	// it's hardlinked, so set the type flag accordingly
	if fi.Mode().IsRegular() && nlink > 1 {
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

func createTarFile(path, extractDir string, hdr *tar.Header, reader io.Reader, Lchown bool) error {
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
		mode := uint32(hdr.Mode & 07777)
		switch hdr.Typeflag {
		case tar.TypeBlock:
			mode |= syscall.S_IFBLK
		case tar.TypeChar:
			mode |= syscall.S_IFCHR
		case tar.TypeFifo:
			mode |= syscall.S_IFIFO
		}

		if err := system.Mknod(path, mode, int(system.Mkdev(hdr.Devmajor, hdr.Devminor))); err != nil {
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
		log.Debugf("PAX Global Extended Headers found and ignored")
		return nil

	default:
		return fmt.Errorf("Unhandled tar header type %d\n", hdr.Typeflag)
	}

	if err := os.Lchown(path, hdr.Uid, hdr.Gid); err != nil && Lchown {
		return err
	}

	for key, value := range hdr.Xattrs {
		if err := system.Lsetxattr(path, key, []byte(value), 0); err != nil {
			return err
		}
	}

	// There is no LChmod, so ignore mode for symlink. Also, this
	// must happen after chown, as that can modify the file mode
	if hdr.Typeflag != tar.TypeSymlink {
		if err := os.Chmod(path, hdrInfo.Mode()); err != nil {
			return err
		}
	}

	ts := []syscall.Timespec{timeToTimespec(hdr.AccessTime), timeToTimespec(hdr.ModTime)}
	// syscall.UtimesNano doesn't support a NOFOLLOW flag atm, and
	if hdr.Typeflag != tar.TypeSymlink {
		if err := system.UtimesNano(path, ts); err != nil && err != system.ErrNotSupportedPlatform {
			return err
		}
	} else {
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

func escapeName(name string) string {
	escaped := make([]byte, 0)
	for i, c := range []byte(name) {
		if i == 0 && c == '/' {
			continue
		}
		// all printable chars except "-" which is 0x2d
		if (0x20 <= c && c <= 0x7E) && c != 0x2d {
			escaped = append(escaped, c)
		} else {
			escaped = append(escaped, fmt.Sprintf("\\%03o", c)...)
		}
	}
	return string(escaped)
}

// TarWithOptions creates an archive from the directory at `path`, only including files whose relative
// paths are included in `options.IncludeFiles` (if non-nil) or not in `options.ExcludePatterns`.
func TarWithOptions(srcPath string, options *TarOptions) (io.ReadCloser, error) {
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
		}
		// this buffer is needed for the duration of this piped stream
		defer pools.BufioWriter32KPool.Put(ta.Buffer)

		// In general we log errors here but ignore them because
		// during e.g. a diff operation the container can continue
		// mutating the filesystem and we can see transient errors
		// from this

		if options.IncludeFiles == nil {
			options.IncludeFiles = []string{"."}
		}

		seen := make(map[string]bool)

		var renamedRelFilePath string // For when tar.Options.Name is set
		for _, include := range options.IncludeFiles {
			filepath.Walk(filepath.Join(srcPath, include), func(filePath string, f os.FileInfo, err error) error {
				if err != nil {
					log.Debugf("Tar: Can't stat file %s to tar: %s", srcPath, err)
					return nil
				}

				relFilePath, err := filepath.Rel(srcPath, filePath)
				if err != nil || (relFilePath == "." && f.IsDir()) {
					// Error getting relative path OR we are looking
					// at the root path. Skip in both situations.
					return nil
				}

				skip := false

				// If "include" is an exact match for the current file
				// then even if there's an "excludePatterns" pattern that
				// matches it, don't skip it. IOW, assume an explicit 'include'
				// is asking for that file no matter what - which is true
				// for some files, like .dockerignore and Dockerfile (sometimes)
				if include != relFilePath {
					skip, err = fileutils.Matches(relFilePath, options.ExcludePatterns)
					if err != nil {
						log.Debugf("Error matching %s", relFilePath, err)
						return err
					}
				}

				if skip {
					if f.IsDir() {
						return filepath.SkipDir
					}
					return nil
				}

				if seen[relFilePath] {
					return nil
				}
				seen[relFilePath] = true

				// Rename the base resource
				if options.Name != "" && filePath == srcPath+"/"+filepath.Base(relFilePath) {
					renamedRelFilePath = relFilePath
				}
				// Set this to make sure the items underneath also get renamed
				if options.Name != "" {
					relFilePath = strings.Replace(relFilePath, renamedRelFilePath, options.Name, 1)
				}

				if err := ta.addTarFile(filePath, relFilePath); err != nil {
					log.Debugf("Can't add file %s to tar: %s", filePath, err)
				}
				return nil
			})
		}

		// Make sure to check the error on Close.
		if err := ta.TarWriter.Close(); err != nil {
			log.Debugf("Can't close tar writer: %s", err)
		}
		if err := compressWriter.Close(); err != nil {
			log.Debugf("Can't close compress writer: %s", err)
		}
		if err := pipeWriter.Close(); err != nil {
			log.Debugf("Can't close pipe writer: %s", err)
		}
	}()

	return pipeReader, nil
}

func Unpack(decompressedArchive io.Reader, dest string, options *TarOptions) error {
	tr := tar.NewReader(decompressedArchive)
	trBuf := pools.BufioReader32KPool.Get(nil)
	defer pools.BufioReader32KPool.Put(trBuf)

	var dirs []*tar.Header

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
		// This keeps "../" as-is, but normalizes "/../" to "/"
		hdr.Name = filepath.Clean(hdr.Name)

		for _, exclude := range options.ExcludePatterns {
			if strings.HasPrefix(hdr.Name, exclude) {
				continue loop
			}
		}

		if !strings.HasSuffix(hdr.Name, "/") {
			// Not the root directory, ensure that the parent directory exists
			parent := filepath.Dir(hdr.Name)
			parentPath := filepath.Join(dest, parent)
			if _, err := os.Lstat(parentPath); err != nil && os.IsNotExist(err) {
				err = os.MkdirAll(parentPath, 0777)
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
		if strings.HasPrefix(rel, "../") {
			return breakoutError(fmt.Errorf("%q is outside of %q", hdr.Name, dest))
		}

		// If path exits we almost always just want to remove and replace it
		// The only exception is when it is a directory *and* the file from
		// the layer is also a directory. Then we want to merge them (i.e.
		// just apply the metadata from the layer).
		if fi, err := os.Lstat(path); err == nil {
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
		if err := createTarFile(path, dest, hdr, trBuf, !options.NoLchown); err != nil {
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
		ts := []syscall.Timespec{timeToTimespec(hdr.AccessTime), timeToTimespec(hdr.ModTime)}
		if err := syscall.UtimesNano(path, ts); err != nil {
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
func Untar(archive io.Reader, dest string, options *TarOptions) error {
	if archive == nil {
		return fmt.Errorf("Empty archive")
	}
	dest = filepath.Clean(dest)
	if options == nil {
		options = &TarOptions{}
	}
	if options.ExcludePatterns == nil {
		options.ExcludePatterns = []string{}
	}
	decompressedArchive, err := DecompressStream(archive)
	if err != nil {
		return err
	}
	defer decompressedArchive.Close()
	return Unpack(decompressedArchive, dest, options)
}

func (archiver *Archiver) TarUntar(src, dst string) error {
	log.Debugf("TarUntar(%s %s)", src, dst)
	archive, err := TarWithOptions(src, &TarOptions{Compression: Uncompressed})
	if err != nil {
		return err
	}
	defer archive.Close()
	return archiver.Untar(archive, dst, nil)
}

// TarUntar is a convenience function which calls Tar and Untar, with the output of one piped into the other.
// If either Tar or Untar fails, TarUntar aborts and returns the error.
func TarUntar(src, dst string) error {
	return defaultArchiver.TarUntar(src, dst)
}

func (archiver *Archiver) UntarPath(src, dst string) error {
	archive, err := os.Open(src)
	if err != nil {
		return err
	}
	defer archive.Close()
	if err := archiver.Untar(archive, dst, nil); err != nil {
		return err
	}
	return nil
}

// UntarPath is a convenience function which looks for an archive
// at filesystem path `src`, and unpacks it at `dst`.
func UntarPath(src, dst string) error {
	return defaultArchiver.UntarPath(src, dst)
}

func (archiver *Archiver) CopyWithTar(src, dst string) error {
	srcSt, err := os.Stat(src)
	if err != nil {
		return err
	}
	if !srcSt.IsDir() {
		return archiver.CopyFileWithTar(src, dst)
	}
	// Create dst, copy src's content into it
	log.Debugf("Creating dest directory: %s", dst)
	if err := os.MkdirAll(dst, 0755); err != nil && !os.IsExist(err) {
		return err
	}
	log.Debugf("Calling TarUntar(%s, %s)", src, dst)
	return archiver.TarUntar(src, dst)
}

// CopyWithTar creates a tar archive of filesystem path `src`, and
// unpacks it at filesystem path `dst`.
// The archive is streamed directly with fixed buffering and no
// intermediary disk IO.
func CopyWithTar(src, dst string) error {
	return defaultArchiver.CopyWithTar(src, dst)
}

func (archiver *Archiver) CopyFileWithTar(src, dst string) (err error) {
	log.Debugf("CopyFileWithTar(%s, %s)", src, dst)
	srcSt, err := os.Stat(src)
	if err != nil {
		return err
	}
	if srcSt.IsDir() {
		return fmt.Errorf("Can't copy a directory")
	}
	// Clean up the trailing /
	if dst[len(dst)-1] == '/' {
		dst = path.Join(dst, filepath.Base(src))
	}
	// Create the holding directory if necessary
	if err := os.MkdirAll(filepath.Dir(dst), 0700); err != nil && !os.IsExist(err) {
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
	return archiver.Untar(r, filepath.Dir(dst), nil)
}

// CopyFileWithTar emulates the behavior of the 'cp' command-line
// for a single file. It copies a regular file from path `src` to
// path `dst`, and preserves all its metadata.
//
// If `dst` ends with a trailing slash '/', the final destination path
// will be `dst/base(src)`.
func CopyFileWithTar(src, dst string) (err error) {
	return defaultArchiver.CopyFileWithTar(src, dst)
}

// CmdStream executes a command, and returns its stdout as a stream.
// If the command fails to run or doesn't complete successfully, an error
// will be returned, including anything written on stderr.
func CmdStream(cmd *exec.Cmd, input io.Reader) (io.ReadCloser, error) {
	if input != nil {
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return nil, err
		}
		// Write stdin if any
		go func() {
			io.Copy(stdin, input)
			stdin.Close()
		}()
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	pipeR, pipeW := io.Pipe()
	errChan := make(chan []byte)
	// Collect stderr, we will use it in case of an error
	go func() {
		errText, e := ioutil.ReadAll(stderr)
		if e != nil {
			errText = []byte("(...couldn't fetch stderr: " + e.Error() + ")")
		}
		errChan <- errText
	}()
	// Copy stdout to the returned pipe
	go func() {
		_, err := io.Copy(pipeW, stdout)
		if err != nil {
			pipeW.CloseWithError(err)
		}
		errText := <-errChan
		if err := cmd.Wait(); err != nil {
			pipeW.CloseWithError(fmt.Errorf("%s: %s", err, errText))
		} else {
			pipeW.Close()
		}
	}()
	// Run the command and return the pipe
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return pipeR, nil
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
	if err = f.Sync(); err != nil {
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
