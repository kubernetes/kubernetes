package archive

import (
	"archive/tar"
	"errors"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/pkg/system"
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
	// Ensure paths are in platform semantics
	cleanedPath = normalizePath(cleanedPath)
	originalPath = normalizePath(originalPath)

	if !specifiesCurrentDir(cleanedPath) && specifiesCurrentDir(originalPath) {
		if !hasTrailingPathSeparator(cleanedPath) {
			// Add a separator if it doesn't already end with one (a cleaned
			// path would only end in a separator if it is the root).
			cleanedPath += string(filepath.Separator)
		}
		cleanedPath += "."
	}

	if !hasTrailingPathSeparator(cleanedPath) && hasTrailingPathSeparator(originalPath) {
		cleanedPath += string(filepath.Separator)
	}

	return cleanedPath
}

// assertsDirectory returns whether the given path is
// asserted to be a directory, i.e., the path ends with
// a trailing '/' or `/.`, assuming a path separator of `/`.
func assertsDirectory(path string) bool {
	return hasTrailingPathSeparator(path) || specifiesCurrentDir(path)
}

// hasTrailingPathSeparator returns whether the given
// path ends with the system's path separator character.
func hasTrailingPathSeparator(path string) bool {
	return len(path) > 0 && os.IsPathSeparator(path[len(path)-1])
}

// specifiesCurrentDir returns whether the given path specifies
// a "current directory", i.e., the last path segment is `.`.
func specifiesCurrentDir(path string) bool {
	return filepath.Base(path) == "."
}

// SplitPathDirEntry splits the given path between its directory name and its
// basename by first cleaning the path but preserves a trailing "." if the
// original path specified the current directory.
func SplitPathDirEntry(path string) (dir, base string) {
	cleanedPath := filepath.Clean(normalizePath(path))

	if specifiesCurrentDir(path) {
		cleanedPath += string(filepath.Separator) + "."
	}

	return filepath.Dir(cleanedPath), filepath.Base(cleanedPath)
}

// TarResource archives the resource described by the given CopyInfo to a Tar
// archive. A non-nil error is returned if sourcePath does not exist or is
// asserted to be a directory but exists as another type of file.
//
// This function acts as a convenient wrapper around TarWithOptions, which
// requires a directory as the source path. TarResource accepts either a
// directory or a file path and correctly sets the Tar options.
func TarResource(sourceInfo CopyInfo) (content Archive, err error) {
	return TarResourceRebase(sourceInfo.Path, sourceInfo.RebaseName)
}

// TarResourceRebase is like TarResource but renames the first path element of
// items in the resulting tar archive to match the given rebaseName if not "".
func TarResourceRebase(sourcePath, rebaseName string) (content Archive, err error) {
	sourcePath = normalizePath(sourcePath)
	if _, err = os.Lstat(sourcePath); err != nil {
		// Catches the case where the source does not exist or is not a
		// directory if asserted to be a directory, as this also causes an
		// error.
		return
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
		RebaseNames: map[string]string{
			sourceBase: rebaseName,
		},
	})
}

// CopyInfo holds basic info about the source
// or destination path of a copy operation.
type CopyInfo struct {
	Path       string
	Exists     bool
	IsDir      bool
	RebaseName string
}

// CopyInfoSourcePath stats the given path to create a CopyInfo
// struct representing that resource for the source of an archive copy
// operation. The given path should be an absolute local path. A source path
// has all symlinks evaluated that appear before the last path separator ("/"
// on Unix). As it is to be a copy source, the path must exist.
func CopyInfoSourcePath(path string, followLink bool) (CopyInfo, error) {
	// normalize the file path and then evaluate the symbol link
	// we will use the target file instead of the symbol link if
	// followLink is set
	path = normalizePath(path)

	resolvedPath, rebaseName, err := ResolveHostSourcePath(path, followLink)
	if err != nil {
		return CopyInfo{}, err
	}

	stat, err := os.Lstat(resolvedPath)
	if err != nil {
		return CopyInfo{}, err
	}

	return CopyInfo{
		Path:       resolvedPath,
		Exists:     true,
		IsDir:      stat.IsDir(),
		RebaseName: rebaseName,
	}, nil
}

// CopyInfoDestinationPath stats the given path to create a CopyInfo
// struct representing that resource for the destination of an archive copy
// operation. The given path should be an absolute local path.
func CopyInfoDestinationPath(path string) (info CopyInfo, err error) {
	maxSymlinkIter := 10 // filepath.EvalSymlinks uses 255, but 10 already seems like a lot.
	path = normalizePath(path)
	originalPath := path

	stat, err := os.Lstat(path)

	if err == nil && stat.Mode()&os.ModeSymlink == 0 {
		// The path exists and is not a symlink.
		return CopyInfo{
			Path:   path,
			Exists: true,
			IsDir:  stat.IsDir(),
		}, nil
	}

	// While the path is a symlink.
	for n := 0; err == nil && stat.Mode()&os.ModeSymlink != 0; n++ {
		if n > maxSymlinkIter {
			// Don't follow symlinks more than this arbitrary number of times.
			return CopyInfo{}, errors.New("too many symlinks in " + originalPath)
		}

		// The path is a symbolic link. We need to evaluate it so that the
		// destination of the copy operation is the link target and not the
		// link itself. This is notably different than CopyInfoSourcePath which
		// only evaluates symlinks before the last appearing path separator.
		// Also note that it is okay if the last path element is a broken
		// symlink as the copy operation should create the target.
		var linkTarget string

		linkTarget, err = os.Readlink(path)
		if err != nil {
			return CopyInfo{}, err
		}

		if !system.IsAbs(linkTarget) {
			// Join with the parent directory.
			dstParent, _ := SplitPathDirEntry(path)
			linkTarget = filepath.Join(dstParent, linkTarget)
		}

		path = linkTarget
		stat, err = os.Lstat(path)
	}

	if err != nil {
		// It's okay if the destination path doesn't exist. We can still
		// continue the copy operation if the parent directory exists.
		if !os.IsNotExist(err) {
			return CopyInfo{}, err
		}

		// Ensure destination parent dir exists.
		dstParent, _ := SplitPathDirEntry(path)

		parentDirStat, err := os.Lstat(dstParent)
		if err != nil {
			return CopyInfo{}, err
		}
		if !parentDirStat.IsDir() {
			return CopyInfo{}, ErrNotDirectory
		}

		return CopyInfo{Path: path}, nil
	}

	// The path exists after resolving symlinks.
	return CopyInfo{
		Path:   path,
		Exists: true,
		IsDir:  stat.IsDir(),
	}, nil
}

// PrepareArchiveCopy prepares the given srcContent archive, which should
// contain the archived resource described by srcInfo, to the destination
// described by dstInfo. Returns the possibly modified content archive along
// with the path to the destination directory which it should be extracted to.
func PrepareArchiveCopy(srcContent Reader, srcInfo, dstInfo CopyInfo) (dstDir string, content Archive, err error) {
	// Ensure in platform semantics
	srcInfo.Path = normalizePath(srcInfo.Path)
	dstInfo.Path = normalizePath(dstInfo.Path)

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
		if len(srcInfo.RebaseName) != 0 {
			srcBase = srcInfo.RebaseName
		}
		return dstDir, RebaseArchiveEntries(srcContent, srcBase, dstBase), nil
	case srcInfo.IsDir:
		// The destination does not exist and the source content is an archive
		// of a directory. The archive should be extracted to the parent of
		// the destination path instead, and when it is, the directory that is
		// created as a result should take the name of the destination path.
		// The source content entries will have to be renamed to have a
		// basename which matches the destination path's basename.
		if len(srcInfo.RebaseName) != 0 {
			srcBase = srcInfo.RebaseName
		}
		return dstDir, RebaseArchiveEntries(srcContent, srcBase, dstBase), nil
	case assertsDirectory(dstInfo.Path):
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
		if len(srcInfo.RebaseName) != 0 {
			srcBase = srcInfo.RebaseName
		}
		return dstDir, RebaseArchiveEntries(srcContent, srcBase, dstBase), nil
	}

}

// RebaseArchiveEntries rewrites the given srcContent archive replacing
// an occurrence of oldBase with newBase at the beginning of entry names.
func RebaseArchiveEntries(srcContent Reader, oldBase, newBase string) Archive {
	if oldBase == string(os.PathSeparator) {
		// If oldBase specifies the root directory, use an empty string as
		// oldBase instead so that newBase doesn't replace the path separator
		// that all paths will start with.
		oldBase = ""
	}

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
func CopyResource(srcPath, dstPath string, followLink bool) error {
	var (
		srcInfo CopyInfo
		err     error
	)

	// Ensure in platform semantics
	srcPath = normalizePath(srcPath)
	dstPath = normalizePath(dstPath)

	// Clean the source and destination paths.
	srcPath = PreserveTrailingDotOrSeparator(filepath.Clean(srcPath), srcPath)
	dstPath = PreserveTrailingDotOrSeparator(filepath.Clean(dstPath), dstPath)

	if srcInfo, err = CopyInfoSourcePath(srcPath, followLink); err != nil {
		return err
	}

	content, err := TarResource(srcInfo)
	if err != nil {
		return err
	}
	defer content.Close()

	return CopyTo(content, srcInfo, dstPath)
}

// CopyTo handles extracting the given content whose
// entries should be sourced from srcInfo to dstPath.
func CopyTo(content Reader, srcInfo CopyInfo, dstPath string) error {
	// The destination path need not exist, but CopyInfoDestinationPath will
	// ensure that at least the parent directory exists.
	dstInfo, err := CopyInfoDestinationPath(normalizePath(dstPath))
	if err != nil {
		return err
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

// ResolveHostSourcePath decides real path need to be copied with parameters such as
// whether to follow symbol link or not, if followLink is true, resolvedPath will return
// link target of any symbol link file, else it will only resolve symlink of directory
// but return symbol link file itself without resolving.
func ResolveHostSourcePath(path string, followLink bool) (resolvedPath, rebaseName string, err error) {
	if followLink {
		resolvedPath, err = filepath.EvalSymlinks(path)
		if err != nil {
			return
		}

		resolvedPath, rebaseName = GetRebaseName(path, resolvedPath)
	} else {
		dirPath, basePath := filepath.Split(path)

		// if not follow symbol link, then resolve symbol link of parent dir
		var resolvedDirPath string
		resolvedDirPath, err = filepath.EvalSymlinks(dirPath)
		if err != nil {
			return
		}
		// resolvedDirPath will have been cleaned (no trailing path separators) so
		// we can manually join it with the base path element.
		resolvedPath = resolvedDirPath + string(filepath.Separator) + basePath
		if hasTrailingPathSeparator(path) && filepath.Base(path) != filepath.Base(resolvedPath) {
			rebaseName = filepath.Base(path)
		}
	}
	return resolvedPath, rebaseName, nil
}

// GetRebaseName normalizes and compares path and resolvedPath,
// return completed resolved path and rebased file name
func GetRebaseName(path, resolvedPath string) (string, string) {
	// linkTarget will have been cleaned (no trailing path separators and dot) so
	// we can manually join it with them
	var rebaseName string
	if specifiesCurrentDir(path) && !specifiesCurrentDir(resolvedPath) {
		resolvedPath += string(filepath.Separator) + "."
	}

	if hasTrailingPathSeparator(path) && !hasTrailingPathSeparator(resolvedPath) {
		resolvedPath += string(filepath.Separator)
	}

	if filepath.Base(path) != filepath.Base(resolvedPath) {
		// In the case where the path had a trailing separator and a symlink
		// evaluation has changed the last path component, we will need to
		// rebase the name in the archive that is being copied to match the
		// originally requested name.
		rebaseName = filepath.Base(path)
	}
	return resolvedPath, rebaseName
}
