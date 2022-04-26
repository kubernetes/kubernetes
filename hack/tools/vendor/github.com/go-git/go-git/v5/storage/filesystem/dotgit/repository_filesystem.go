package dotgit

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/go-git/go-billy/v5"
)

// RepositoryFilesystem is a billy.Filesystem compatible object wrapper
// which handles dot-git filesystem operations and supports commondir according to git scm layout:
// https://github.com/git/git/blob/master/Documentation/gitrepository-layout.txt
type RepositoryFilesystem struct {
	dotGitFs       billy.Filesystem
	commonDotGitFs billy.Filesystem
}

func NewRepositoryFilesystem(dotGitFs, commonDotGitFs billy.Filesystem) *RepositoryFilesystem {
	return &RepositoryFilesystem{
		dotGitFs:       dotGitFs,
		commonDotGitFs: commonDotGitFs,
	}
}

func (fs *RepositoryFilesystem) mapToRepositoryFsByPath(path string) billy.Filesystem {
	// Nothing to decide if commondir not defined
	if fs.commonDotGitFs == nil {
		return fs.dotGitFs
	}

	cleanPath := filepath.Clean(path)

	// Check exceptions for commondir (https://git-scm.com/docs/gitrepository-layout#Documentation/gitrepository-layout.txt)
	switch cleanPath {
	case fs.dotGitFs.Join(logsPath, "HEAD"):
		return fs.dotGitFs
	case fs.dotGitFs.Join(refsPath, "bisect"), fs.dotGitFs.Join(refsPath, "rewritten"), fs.dotGitFs.Join(refsPath, "worktree"):
		return fs.dotGitFs
	}

	// Determine dot-git root by first path element.
	// There are some elements which should always use commondir when commondir defined.
	// Usual dot-git root will be used for the rest of files.
	switch strings.Split(cleanPath, string(filepath.Separator))[0] {
	case objectsPath, refsPath, packedRefsPath, configPath, branchesPath, hooksPath, infoPath, remotesPath, logsPath, shallowPath, worktreesPath:
		return fs.commonDotGitFs
	default:
		return fs.dotGitFs
	}
}

func (fs *RepositoryFilesystem) Create(filename string) (billy.File, error) {
	return fs.mapToRepositoryFsByPath(filename).Create(filename)
}

func (fs *RepositoryFilesystem) Open(filename string) (billy.File, error) {
	return fs.mapToRepositoryFsByPath(filename).Open(filename)
}

func (fs *RepositoryFilesystem) OpenFile(filename string, flag int, perm os.FileMode) (billy.File, error) {
	return fs.mapToRepositoryFsByPath(filename).OpenFile(filename, flag, perm)
}

func (fs *RepositoryFilesystem) Stat(filename string) (os.FileInfo, error) {
	return fs.mapToRepositoryFsByPath(filename).Stat(filename)
}

func (fs *RepositoryFilesystem) Rename(oldpath, newpath string) error {
	return fs.mapToRepositoryFsByPath(oldpath).Rename(oldpath, newpath)
}

func (fs *RepositoryFilesystem) Remove(filename string) error {
	return fs.mapToRepositoryFsByPath(filename).Remove(filename)
}

func (fs *RepositoryFilesystem) Join(elem ...string) string {
	return fs.dotGitFs.Join(elem...)
}

func (fs *RepositoryFilesystem) TempFile(dir, prefix string) (billy.File, error) {
	return fs.mapToRepositoryFsByPath(dir).TempFile(dir, prefix)
}

func (fs *RepositoryFilesystem) ReadDir(path string) ([]os.FileInfo, error) {
	return fs.mapToRepositoryFsByPath(path).ReadDir(path)
}

func (fs *RepositoryFilesystem) MkdirAll(filename string, perm os.FileMode) error {
	return fs.mapToRepositoryFsByPath(filename).MkdirAll(filename, perm)
}

func (fs *RepositoryFilesystem) Lstat(filename string) (os.FileInfo, error) {
	return fs.mapToRepositoryFsByPath(filename).Lstat(filename)
}

func (fs *RepositoryFilesystem) Symlink(target, link string) error {
	return fs.mapToRepositoryFsByPath(target).Symlink(target, link)
}

func (fs *RepositoryFilesystem) Readlink(link string) (string, error) {
	return fs.mapToRepositoryFsByPath(link).Readlink(link)
}

func (fs *RepositoryFilesystem) Chroot(path string) (billy.Filesystem, error) {
	return fs.mapToRepositoryFsByPath(path).Chroot(path)
}

func (fs *RepositoryFilesystem) Root() string {
	return fs.dotGitFs.Root()
}
