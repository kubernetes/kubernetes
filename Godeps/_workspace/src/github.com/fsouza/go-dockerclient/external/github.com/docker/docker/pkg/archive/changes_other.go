// +build !linux

package archive

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/system"
)

func collectFileInfoForChanges(oldDir, newDir string) (*FileInfo, *FileInfo, error) {
	var (
		oldRoot, newRoot *FileInfo
		err1, err2       error
		errs             = make(chan error, 2)
	)
	go func() {
		oldRoot, err1 = collectFileInfo(oldDir)
		errs <- err1
	}()
	go func() {
		newRoot, err2 = collectFileInfo(newDir)
		errs <- err2
	}()

	// block until both routines have returned
	for i := 0; i < 2; i++ {
		if err := <-errs; err != nil {
			return nil, nil, err
		}
	}

	return oldRoot, newRoot, nil
}

func collectFileInfo(sourceDir string) (*FileInfo, error) {
	root := newRootFileInfo()

	err := filepath.Walk(sourceDir, func(path string, f os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Rebase path
		relPath, err := filepath.Rel(sourceDir, path)
		if err != nil {
			return err
		}

		// As this runs on the daemon side, file paths are OS specific.
		relPath = filepath.Join(string(os.PathSeparator), relPath)

		// See https://github.com/golang/go/issues/9168 - bug in filepath.Join.
		// Temporary workaround. If the returned path starts with two backslashes,
		// trim it down to a single backslash. Only relevant on Windows.
		if runtime.GOOS == "windows" {
			if strings.HasPrefix(relPath, `\\`) {
				relPath = relPath[1:]
			}
		}

		if relPath == string(os.PathSeparator) {
			return nil
		}

		parent := root.LookUp(filepath.Dir(relPath))
		if parent == nil {
			return fmt.Errorf("collectFileInfo: Unexpectedly no parent for %s", relPath)
		}

		info := &FileInfo{
			name:     filepath.Base(relPath),
			children: make(map[string]*FileInfo),
			parent:   parent,
		}

		s, err := system.Lstat(path)
		if err != nil {
			return err
		}
		info.stat = s

		info.capability, _ = system.Lgetxattr(path, "security.capability")

		parent.children[info.name] = info

		return nil
	})
	if err != nil {
		return nil, err
	}
	return root, nil
}
