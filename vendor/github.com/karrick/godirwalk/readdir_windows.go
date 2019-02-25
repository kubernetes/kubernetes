package godirwalk

import (
	"os"

	"github.com/pkg/errors"
)

// The functions in this file are mere wrappers of what is already provided by
// standard library, in order to provide the same API as this library provides.
//
// The scratch buffer argument is ignored by this architecture.
//
// Please send PR or link to article if you know of a more performant way of
// enumerating directory contents and mode types on Windows.

func readdirents(osDirname string, _ []byte) (Dirents, error) {
	dh, err := os.Open(osDirname)
	if err != nil {
		return nil, errors.Wrap(err, "cannot Open")
	}

	fileinfos, err := dh.Readdir(0)
	if er := dh.Close(); err == nil {
		err = er
	}
	if err != nil {
		return nil, errors.Wrap(err, "cannot Readdir")
	}

	entries := make(Dirents, len(fileinfos))
	for i, info := range fileinfos {
		entries[i] = &Dirent{name: info.Name(), modeType: info.Mode() & os.ModeType}
	}

	return entries, nil
}

func readdirnames(osDirname string, _ []byte) ([]string, error) {
	dh, err := os.Open(osDirname)
	if err != nil {
		return nil, errors.Wrap(err, "cannot Open")
	}

	entries, err := dh.Readdirnames(0)
	if er := dh.Close(); err == nil {
		err = er
	}
	if err != nil {
		return nil, errors.Wrap(err, "cannot Readdirnames")
	}

	return entries, nil
}
