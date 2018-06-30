/*
Copyright 2018 The Kubernetes Authors.

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

package fixtures

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

var (
	// CaCertPath is the filepath to a certificate that can be used as a CA
	// certificate.
	CaCertPath string
	// ServerCertPath is the filepath to a leaf certifiacte signed by the CA at
	// `CaCertPath`.
	ServerCertPath string
	// ServerKeyPath is the filepath to the private key for the ceritifiacte at
	// `ServerCertPath`.
	ServerKeyPath string
	// InvalidCertPath is the filepath to an invalid certificate.
	InvalidCertPath string
)

func init() {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		panic("Cannot get path to the fixtures")
	}

	fixturesDir := filepath.Dir(thisFile)

	cwd, err := os.Getwd()
	if err != nil {
		panic("Cannot get CWD: " + err.Error())
	}

	// When tests run in a bazel sandbox `runtime.Caller()`
	// returns a relative path, when run with plain `go test` the path
	// returned is absolute. To make those fixtures work in both those cases,
	// we prepend the CWD iff the CWD is not yet part of the path to the fixtures.
	if !strings.HasPrefix(fixturesDir, cwd) {
		fixturesDir = filepath.Join(cwd, fixturesDir)
	}

	CaCertPath = filepath.Join(fixturesDir, "ca.pem")
	ServerCertPath = filepath.Join(fixturesDir, "server.pem")
	ServerKeyPath = filepath.Join(fixturesDir, "server.key")
	InvalidCertPath = filepath.Join(fixturesDir, "invalid.pem")
}
