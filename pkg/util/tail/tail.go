/*
Copyright 2017 The Kubernetes Authors.

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

package tail

import (
	"io"
	"os"
)

const (
	// blockSize is the block size used in tail.
	blockSize = 1024
)

// ReadAtMost reads at most max bytes from the end of the file identified by path or
// returns an error. It returns true if the file was longer than max. It will
// allocate up to max bytes.
func ReadAtMost(path string, max int64) ([]byte, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, false, err
	}
	size := fi.Size()
	if size == 0 {
		return nil, false, nil
	}
	if size < max {
		max = size
	}
	offset, err := f.Seek(-max, io.SeekEnd)
	if err != nil {
		return nil, false, err
	}
	data, err := io.ReadAll(f)
	return data, offset > 0, err
}
