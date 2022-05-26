package goimports

import (
	"bytes"
	"fmt"
	"io/ioutil"

	"golang.org/x/tools/imports"
)

func Run(filename string) ([]byte, error) {
	src, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	res, err := imports.Process(filename, src, options)
	if err != nil {
		return nil, err
	}

	if bytes.Equal(src, res) {
		return nil, nil
	}

	// formatting has changed
	data, err := diff(src, res, filename)
	if err != nil {
		return nil, fmt.Errorf("error computing diff: %s", err)
	}

	return data, nil
}
