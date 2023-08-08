/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package debug

import (
	"io"
	"os"
	"path"
	"sync"
)

// FileProvider implements a debugging provider that creates a real file for
// every call to NewFile. It maintains a list of all files that it creates,
// such that it can close them when its Flush function is called.
type FileProvider struct {
	Path string

	mu    sync.Mutex
	files []*os.File
}

func (fp *FileProvider) NewFile(p string) io.WriteCloser {
	f, err := os.Create(path.Join(fp.Path, p))
	if err != nil {
		panic(err)
	}

	fp.mu.Lock()
	defer fp.mu.Unlock()
	fp.files = append(fp.files, f)

	return NewFileWriterCloser(f, p)
}

func (fp *FileProvider) Flush() {
	fp.mu.Lock()
	defer fp.mu.Unlock()
	for _, f := range fp.files {
		f.Close()
	}
}

type FileWriterCloser struct {
	f *os.File
	p string
}

func NewFileWriterCloser(f *os.File, p string) *FileWriterCloser {
	return &FileWriterCloser{
		f,
		p,
	}
}

func (fwc *FileWriterCloser) Write(p []byte) (n int, err error) {
	return fwc.f.Write(Scrub(p))
}

func (fwc *FileWriterCloser) Close() error {
	return fwc.f.Close()
}
