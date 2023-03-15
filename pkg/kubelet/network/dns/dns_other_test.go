//go:build !windows
// +build !windows

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

package dns

import (
	"fmt"
	"os"
	"testing"
)

var (
	defaultResolvConf = "/etc/resolv.conf"
	// configurer.getHostDNSConfig is faked on Windows, while it is not faked on Linux.
	fakeGetHostDNSConfigCustom = getHostDNSConfig
)

// getResolvConf returns a temporary resolv.conf file containing the testHostNameserver nameserver and
// testHostDomain search field, and a cleanup function for the temporary file.
func getResolvConf(t *testing.T) (string, func()) {
	resolvConfContent := []byte(fmt.Sprintf("nameserver %s\nsearch %s\n", testHostNameserver, testHostDomain))
	tmpfile, err := os.CreateTemp("", "tmpResolvConf")
	if err != nil {
		t.Fatal(err)
	}

	cleanup := func() {
		os.Remove(tmpfile.Name())
	}

	if _, err := tmpfile.Write(resolvConfContent); err != nil {
		cleanup()
		t.Fatal(err)
	}
	if err := tmpfile.Close(); err != nil {
		cleanup()
		t.Fatal(err)
	}

	return tmpfile.Name(), cleanup
}
