package io

import "io/ioutil"

// FileObj allows mocking the access to files
type FileObj interface {
	Load() ([]byte, error)
	Path() string
}

// File represents a file that can be loaded from the file system
type File struct {
	FilePath string
}

func (f File) Path() string {
	return f.FilePath
}

func (f File) Load() ([]byte, error) {
	return ioutil.ReadFile(f.FilePath)
}

// FileGeneratorFunc returns a list of files that can be loaded and processed
type FileGeneratorFunc func() ([]FileObj, error)

func (a FileGeneratorFunc) Combine(b FileGeneratorFunc) FileGeneratorFunc {
	return func() ([]FileObj, error) {
		files, err := a()
		if err != nil {
			return nil, err
		}
		additionalFiles, err := b()
		if err != nil {
			return nil, err
		}
		files = append(files, additionalFiles...)
		return files, err
	}
}

func GoFilesInPathsGenerator(paths []string) FileGeneratorFunc {
	return FilesInPathsGenerator(paths, isGoFile)
}

func FilesInPathsGenerator(paths []string, fileCheckFun fileCheckFunction) FileGeneratorFunc {
	return func() (foundFiles []FileObj, err error) {
		for _, path := range paths {
			files, err := FindFilesForPath(path, fileCheckFun)
			if err != nil {
				return nil, err
			}
			for _, filePath := range files {
				foundFiles = append(foundFiles, File{filePath})
			}
		}
		return foundFiles, nil
	}
}
