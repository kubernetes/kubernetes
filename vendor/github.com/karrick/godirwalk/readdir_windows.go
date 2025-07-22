// +build windows

package godirwalk

import "os"

// MinimumScratchBufferSize specifies the minimum size of the scratch buffer
// that ReadDirents, ReadDirnames, Scanner, and Walk will use when reading file
// entries from the operating system. During program startup it is initialized
// to the result from calling `os.Getpagesize()` for non Windows environments,
// and 0 for Windows.
var MinimumScratchBufferSize = 0

func newScratchBuffer() []byte { return nil }

func readDirents(osDirname string, _ []byte) ([]*Dirent, error) {
	dh, err := os.Open(osDirname)
	if err != nil {
		return nil, err
	}

	fileinfos, err := dh.Readdir(-1)
	if err != nil {
		_ = dh.Close()
		return nil, err
	}

	entries := make([]*Dirent, len(fileinfos))

	for i, fi := range fileinfos {
		entries[i] = &Dirent{
			name:     fi.Name(),
			path:     osDirname,
			modeType: fi.Mode() & os.ModeType,
		}
	}

	if err = dh.Close(); err != nil {
		return nil, err
	}
	return entries, nil
}

func readDirnames(osDirname string, _ []byte) ([]string, error) {
	dh, err := os.Open(osDirname)
	if err != nil {
		return nil, err
	}

	fileinfos, err := dh.Readdir(-1)
	if err != nil {
		_ = dh.Close()
		return nil, err
	}

	entries := make([]string, len(fileinfos))

	for i, fi := range fileinfos {
		entries[i] = fi.Name()
	}

	if err = dh.Close(); err != nil {
		return nil, err
	}
	return entries, nil
}
