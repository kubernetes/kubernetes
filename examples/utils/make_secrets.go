/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// A small script to build a secret volume from a directory. This is a stopgap
// to make examples useful until #4822 is resolved.
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"

	// This installs the legacy v1 API
	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/latest"
)

var secretname = flag.String("name", "", "Name for secret volume. Defaults to first directory name.")

func main() {
	flag.Parse()
	if flag.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [ options ] DIRECTORY...\n", path.Base(os.Args[0]))
		flag.PrintDefaults()
		return
	}

	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name: *secretname,
		},
		Data: make(map[string][]byte),
	}
	if secret.ObjectMeta.Name == "" {
		secret.ObjectMeta.Name = path.Base(flag.Arg(0))
	}

	for _, directory := range flag.Args() {
		readfiles := func(path string, info os.FileInfo, err error) error {
			if err != nil {
				log.Fatalf("Error building secrets: %v", err)
			}
			if !info.IsDir() {
				b, err := ioutil.ReadFile(path)
				if err != nil {
					log.Fatalf("Cannot read file %v, %v", path, err)
				}

				relpath, err := filepath.Rel(directory, path)
				if err != nil {
					log.Fatalf("Cannot determine relpath for %v, %v", path, err)
				}

				secret.Data[relpath] = b
			}
			return nil
		}

		filepath.Walk(directory, readfiles)
	}

	fmt.Printf(runtime.EncodeOrDie(latest.GroupOrDie("").Codec, secret))
}
