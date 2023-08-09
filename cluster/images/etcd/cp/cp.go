/*
Copyright 2020 The Kubernetes Authors.

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

package main

import (
	"io"
	"log"
	"os"
	"path/filepath"
)

func main() {
	if len(os.Args) != 3 {
		log.Fatal("Usage: cp SOURCE DEST")
	}

	sf, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatalf("unable to open source file [%s]: %q", os.Args[1], err)
	}
	defer sf.Close()
	fi, err := sf.Stat()
	if err != nil {
		log.Fatalf("unable to stat source file [%s]: %q", os.Args[1], err)
	}

	dir := filepath.Dir(os.Args[2])
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Fatalf("unable to create directory [%s]: %q", dir, err)
	}
	df, err := os.Create(os.Args[2])
	if err != nil {
		log.Fatalf("unable to create destination file [%s]: %q", os.Args[1], err)
	}
	defer df.Close()

	_, err = io.Copy(df, sf)
	if err != nil {
		log.Fatalf("unable to copy [%s] to [%s]: %q", os.Args[1], os.Args[2], err)
	}

	if err := df.Close(); err != nil {
		log.Fatalf("unable to close destination file: %q", err)
	}
	if err := os.Chmod(os.Args[2], fi.Mode()); err != nil {
		log.Fatalf("unable to close destination file: %q", err)
	}
}
