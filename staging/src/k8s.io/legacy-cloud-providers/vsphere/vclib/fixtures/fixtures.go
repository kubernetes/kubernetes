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
	"fmt"
	"os"
	"path/filepath"
	"runtime"
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
	fixturesDir, err := pkgPath()
	if err != nil {
		panic(fmt.Sprintf("Cannot get path to the fixtures: %s", err))
	}

	CaCertPath = filepath.Join(fixturesDir, "ca.pem")
	ServerCertPath = filepath.Join(fixturesDir, "server.pem")
	ServerKeyPath = filepath.Join(fixturesDir, "server.key")
	InvalidCertPath = filepath.Join(fixturesDir, "invalid.pem")
}

// pkgPath returns the absolute file path to this package's directory. With go
// test, we can just look at the runtime call stack. However, bazel compiles go
// binaries with the -trimpath option so the simple approach fails however we
// can consult environment variables to derive the path.
//
// The approach taken here works for both go test and bazel on the assumption
// that if and only if trimpath is passed, we are running under bazel.
func pkgPath() (string, error) {
	_, thisFile, _, ok := runtime.Caller(1)
	if !ok {
		return "", fmt.Errorf("failed to get current file")
	}

	pkgPath := filepath.Dir(thisFile)

	// If we find bazel env variables, then -trimpath was passed so we need to
	// construct the path from the environment.
	if testSrcdir, testWorkspace := os.Getenv("TEST_SRCDIR"), os.Getenv("TEST_WORKSPACE"); testSrcdir != "" && testWorkspace != "" {
		pkgPath = filepath.Join(testSrcdir, testWorkspace, pkgPath)
	}

	// If the path is still not absolute, something other than bazel compiled
	// with -trimpath.
	if !filepath.IsAbs(pkgPath) {
		return "", fmt.Errorf("can't construct an absolute path from %q", pkgPath)
	}

	return pkgPath, nil
}
