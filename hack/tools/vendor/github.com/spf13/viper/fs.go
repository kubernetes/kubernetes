//go:build go1.16 && finder
// +build go1.16,finder

package viper

import (
	"errors"
	"io/fs"
	"path"
)

type finder struct {
	paths      []string
	fileNames  []string
	extensions []string

	withoutExtension bool
}

func (f finder) Find(fsys fs.FS) (string, error) {
	for _, searchPath := range f.paths {
		for _, fileName := range f.fileNames {
			for _, extension := range f.extensions {
				filePath := path.Join(searchPath, fileName+"."+extension)

				ok, err := fileExists(fsys, filePath)
				if err != nil {
					return "", err
				}

				if ok {
					return filePath, nil
				}
			}

			if f.withoutExtension {
				filePath := path.Join(searchPath, fileName)

				ok, err := fileExists(fsys, filePath)
				if err != nil {
					return "", err
				}

				if ok {
					return filePath, nil
				}
			}
		}
	}

	return "", nil
}

func fileExists(fsys fs.FS, filePath string) (bool, error) {
	fileInfo, err := fs.Stat(fsys, filePath)
	if err == nil {
		return !fileInfo.IsDir(), nil
	}

	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}

	return false, err
}
