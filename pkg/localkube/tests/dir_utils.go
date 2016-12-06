/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package tests

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"k8s.io/minikube/pkg/minikube/constants"
)

func MakeTempDir() string {
	tempDir, err := ioutil.TempDir("", "minipath")
	if err != nil {
		log.Fatal(err)
	}
	err = os.MkdirAll(filepath.Join(tempDir, "addons"), 0777)
	if err != nil {
		log.Fatal(err)
	}
	err = os.MkdirAll(filepath.Join(tempDir, "cache", "iso"), 0777)
	if err != nil {
		log.Fatal(err)
	}
	err = os.MkdirAll(filepath.Join(tempDir, "cache", "localkube"), 0777)
	if err != nil {
		log.Fatal(err)
	}
	constants.Minipath = tempDir
	return tempDir
}
