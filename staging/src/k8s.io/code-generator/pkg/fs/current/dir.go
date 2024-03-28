/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package current

import (
	"errors"
	"path/filepath"
	"runtime"
)

// Dir is the __dirname equivalent.
func Dir() (string, error) {
	filename, err := file(2)
	if err != nil {
		return "", err
	}
	return filepath.Dir(filename), nil
}

var ErrCantGetCurrentFilename = errors.New("unable to get the current filename")

func file(skip int) (string, error) {
	_, filename, _, ok := runtime.Caller(skip)
	if !ok {
		return "", ErrCantGetCurrentFilename
	}
	return filename, nil
}
