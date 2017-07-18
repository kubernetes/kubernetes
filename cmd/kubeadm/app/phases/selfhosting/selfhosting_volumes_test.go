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

package selfhosting

import (
	"io/ioutil"
	"log"
	"os"
	"testing"
)

func createTemporaryFile(name string) *os.File {
	content := []byte("foo")
	tmpfile, err := ioutil.TempFile("", name)
	if err != nil {
		log.Fatal(err)
	}

	if _, err := tmpfile.Write(content); err != nil {
		log.Fatal(err)
	}

	return tmpfile
}

func TestCreateTLSSecretFromFile(t *testing.T) {
	tmpCert := createTemporaryFile("foo.crt")
	defer os.Remove(tmpCert.Name())
	tmpKey := createTemporaryFile("foo.key")
	defer os.Remove(tmpKey.Name())

	_, err := createTLSSecretFromFiles("foo", tmpCert.Name(), tmpKey.Name())
	if err != nil {
		log.Fatal(err)
	}

	if err := tmpCert.Close(); err != nil {
		log.Fatal(err)
	}

	if err := tmpKey.Close(); err != nil {
		log.Fatal(err)
	}
}

func TestCreateOpaqueSecretFromFile(t *testing.T) {
	tmpFile := createTemporaryFile("foo")
	defer os.Remove(tmpFile.Name())

	_, err := createOpaqueSecretFromFile("foo", tmpFile.Name())
	if err != nil {
		log.Fatal(err)
	}

	if err := tmpFile.Close(); err != nil {
		log.Fatal(err)
	}
}
