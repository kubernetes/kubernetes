package io

import (
	"io/ioutil"
	"os"
)

type stdInFile struct {
}

func (s stdInFile) Load() ([]byte, error) {
	return ioutil.ReadAll(os.Stdin)
}

func (s stdInFile) Path() string {
	return "StdIn"
}

var StdInGenerator FileGeneratorFunc = func() ([]FileObj, error) {
	stat, err := os.Stdin.Stat()
	if err != nil {
		return nil, err
	}
	if (stat.Mode() & os.ModeCharDevice) == 0 {
		return []FileObj{stdInFile{}}, nil
	}
	return []FileObj{}, nil
}
