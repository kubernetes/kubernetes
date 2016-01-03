package archive

import (
	"archive/tar"
	"errors"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/fsouza/go-dockerclient/external/github.com/Sirupsen/logrus"
)

// Errors used or returned by this file.
var (
	ErrNotDirectory      = errors.New("not a directory")
	ErrDirNotExists      = errors.New("no such directory")
	ErrCannotCopyDir     = errors.New("cannot copy directory")
	ErrInvalidCopySource = errors.New("invalid copy source content")
)

// PreserveTrailingDotOrSeparator returns the given cleaned path (after
// processing using any utility functions from the path or filepath stdlib
// packages) and appends a trailing `/.` or `/` if its corresponding  original
// path (from before being processed by utility functions from the path or
// filepath stdlib packages) ends with a trailing `/.` or `/`. If the cleaned
// path already ends in a `.` path segment, then another is not added. If the
// clean path already ends in a path separator, then another is not added.
func PreserveTrailingDotOrSeparator(cleanedPath, originalPath string) string {
	if !SpecifiesCurrentDir(cleanedPath) && SpecifiesCurrentDir(originalPath) {
		if !HasTrailingPathSeparator(cleanedPath) {
			// Add a separator if it doesn't already end with one (a cleaned
			// path would only end in a separator if it is the root).
			cleanedPath += string(filepath.Separator)
		}
		cleanedPath += "."
	}

	if !HasTrailingPathSeparator(cleanedPath) && HasTrailingPathSeparator(originalPath) {
		cleanedPath += string(filepath.Separator)
	}

	return cleanedPath
}

// AssertsDirectory returns whether the given path is
// asserted to be a directory, i.e., the path ends with
// a trailing '/' or `/.`, assuming a path separator of `/`.
func AssertsDirectory(path string) bool {
	return HasTrailingPathSeparator(path) || SpecifiesCurrentDir(path)
}

// HasTrailingPathSeparator returns whether the given
// path ends with the system's path separator character.
func HasTrailingPathSeparator(path string) bool {
	return len(path) > 0 && os.IsPathSeparator(path[len(path)-1])
}

// SpecifiesCurrentDir returns whether the given path specifies
// a "current directory", i.e., the last path segment is `.`.
func SpecifiesCurrentDir(path string) bool {
	return filepath.Base(path) == "."
}

// SplitPathDirEntry splits the given path between its
// parent directory and its basename in that directory.
func SplitPathDirEntry(localizedPath string) (dir, base string) {
	normalizedPath := filepath.ToSlash(localizedPath)
	vol := filepath.VolumeName(normalizedPath)
	normalizedPath = normalizedPath[len(vol):]

	if normalizedPath == "/" {
		// Specifies the root path.
		return filepath.FromSlash(vol + normalizedPath), "."
	}

	trimmedPath := vol + strings.TrimRight(normalizedPath, "/")

	dir = filepath.FromSlash(path.Dir(trimmedPath))
	base = filepath.FromSlash(path.Base(trimmedPath))

	return dir, base
}

// TarResource archives the resource at the given sourcePath into a Tar
// archive. A non-nil error is returned if sourcePath does not exist or is
// asserted to be a directory but exists as another type of file.
//
// This function acts as a convenient wrapper around TarWithOptions, which
// requires a directory as the source path. TarResource accepts either a
// directory or a file path and correctly sets the Tar options.
func TarResource(sourcePath string) (content Archive, err error) {
	if _, err = os.Lstat(sourcePath); err != nil {
		// Catches the case where the source does not exist or is not a
		// directory if asserted to be a directory, as this also causes an
		// error.
		return
	}

	if len(sourcePath) > 1 && HasTrailingPathSeparator(sourcePath) {
		// In the case where the source path is a symbolic link AND it ends
		// with a path separator, we will want to evaluate the symbolic link.
		trimmedPath := sourcePath[:len(sourcePath)-1]
		stat, err := os.Lstat(trimmedPath)
		if err != nil {
			return nil, err
		}

		if stat.Mode()&os.ModeSymlink != 0 {
			if sourcePath, err = filepath.EvalSymlinks(trimmedPath); err != nil {
				return nil, err
			}
		}
	}

	// Separate the source path between it's directory and
	// the entry in that directory which we are archiving.
	sourceDir, sourceBase := SplitPathDirEntry(sourcePath)

	filter := []string{sourceBase}

	logrus.Debugf("copying %q from %q", sourceBase, sourceDir)

	return TarWithOptions(sourceDir, &TarOptions{
		Compression:      Uncompressed,
		IncludeFiles:     filter,
		IncludeSourceDir: true,
	})
}

// CopyInfo holds basic info about the source
// or destination path of a copy operation.
type CopyInfo struct {
	Path   string
	Exists bool
	IsDir  bool
}

// CopyInfoStatPath stats the given path to create a CopyInfo
// struct representing that resource. If mustExist is true, then
// it is an error if there is no file or directory at the given path.
func CopyInfoStatPath(path string, mustExist bool) (CopyInfo, error) {
	pathInfo := CopyInfo{Path: path}

	fileInfo, err := os.Lstat(path)

	if err == nil {
		pathInfo.Exists, pathInfo.IsDir = true, fileInfo.IsDir()
	} else if os.IsNotExist(err) && !mustExist {
		err = nil
	}

	return pathInfo, err
}

// PrepareArchiveCopy prepares the given srcContent archive, which should
// contain the archived resource described by srcInfo, to the destination
// described by dstInfo. Returns the possibly modified content archive along
// with the path to the destination directory which it should be extracted to.
func PrepareArchiveCopy(srcContent ArchiveReader, srcInfo, dstInfo CopyInfo) (dstDir string, content Archive, err error) {
	// Separate the destination path between its directory and base
	// components in case the source archive contents need to be rebased.
	dstDir, dstBase := SplitPathDirEntry(dstInfo.Path)
	_, srcBase := SplitPathDirEntry(srcInfo.Path)

	switch {
	case dstInfo.Exists && dstInfo.IsDir:
		// The destination exists as a directory. No alteration
		// to srcContent is needed as its contents can be
		// simply extracted to the destination directory.
		return dstInfo.Path, ioutil.NopCloser(srcContent), nil
	case dstInfo.Exists && srcInfo.IsDir:
		// The destination exists as some type of file and the source
		// content is a directory. This is an error condition since
		// you cannot copy a directory to an existing file location.
		return "", nil, ErrCannotCopyDir
	case dstInfo.Exists:
		// The destination exists as some type of file and the source content
		// is also a file. The source content entry will have to be renamed to
		// have a basename which matches the destination path's basename.
		return dstDir, rebaseArchiveEntries(srcContent, srcBase, dstBase), nil
	case srcInfo.IsDir:
		// The destination does not exist and the source content is an archive
		// of a directory. The archive should be extracted to the parent of
		// the destination path instead, and when it is, the directory that is
		// created as a result should take the name of the destination path.
		// The source content entries will have to be renamed to have a
		// basename which matches the destination path's basename.
		return dstDir, rebaseArchiveEntries(srcContent, srcBase, dstBase), nil
	case AssertsDirectory(dstInfo.Path):
		// The destination does not exist and is asserted to be created as a
		// directory, but the source content is not a directory. This is an
		// error condition since you cannot create a directory from a file
		// source.
		return "", nil, ErrDirNotExists
	default:
		// The last remaining case is when the destination does not exist, is
		// not asserted to be a directory, and the source content is not an
		// archive of a directory. It this case, the destination file will need
		// to be created when the archive is extracted and the source content
		// entry will have to be renamed to have a basename which matches the
		// destination path's basename.
		return dstDir, rebaseArchiveEntries(srcContent, srcBase, dstBase), nil
	}

}

// rebaseArchiveEntries rewrites the given srcContent archive replacing
// an occurance of oldBase with newBase at the beginning of entry names.
func rebaseArchiveEntries(srcContent ArchiveReader, oldBase, newBase string) Archive {
	rebased, w := io.Pipe()

	go func() {
		srcTar := tar.NewReader(srcContent)
		rebasedTar := tar.NewWriter(w)

		for {
			hdr, err := srcTar.Next()
			if err == io.EOF {
				// Signals end of archive.
				rebasedTar.Close()
				w.Close()
				return
			}
			if err != nil {
				w.CloseWithError(err)
				return
			}

			hdr.Name = strings.Replace(hdr.Name, oldBase, newBase, 1)

			if err = rebasedTar.WriteHeader(hdr); err != nil {
				w.CloseWithError(err)
				return
			}

			if _, err = io.Copy(rebasedTar, srcTar); err != nil {
				w.CloseWithError(err)
				return
			}
		}
	}()

	return rebased
}

// CopyResource performs an archive copy from the given source path to the
// given destination path. The source path MUST exist and the destination
// path's parent directory must exist.
func CopyResource(srcPath, dstPath string) error {
	var (
		srcInfo CopyInfo
		err     error
	)

	// Clean the source and destination paths.
	srcPath = PreserveTrailingDotOrSeparator(filepath.Clean(srcPath), srcPath)
	dstPath = PreserveTrailingDotOrSeparator(filepath.Clean(dstPath), dstPath)

	if srcInfo, err = CopyInfoStatPath(srcPath, true); err != nil {
		return err
	}

	content, err := TarResource(srcPath)
	if err != nil {
		return err
	}
	defer content.Close()

	return CopyTo(content, srcInfo, dstPath)
}

// CopyTo handles extracting the given content whose
// entries should be sourced from srcInfo to dstPath.
func CopyTo(content ArchiveReader, srcInfo CopyInfo, dstPath string) error {
	dstInfo, err := CopyInfoStatPath(dstPath, false)
	if err != nil {
		return err
	}

	if !dstInfo.Exists {
		// Ensure destination parent dir exists.
		dstParent, _ := SplitPathDirEntry(dstPath)

		dstStat, err := os.Lstat(dstParent)
		if err != nil {
			return err
		}
		if !dstStat.IsDir() {
			return ErrNotDirectory
		}
	}

	dstDir, copyArchive, err := PrepareArchiveCopy(content, srcInfo, dstInfo)
	if err != nil {
		return err
	}
	defer copyArchive.Close()

	options := &TarOptions{
		NoLchown:             true,
		NoOverwriteDirNonDir: true,
	}

	return Untar(copyArchive, dstDir, options)
}
