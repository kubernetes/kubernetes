// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build host coreos src kvm

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
	taas "github.com/coreos/rkt/tests/testutils/aci-server"
)

func TestAuthSanity(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()
	server, image := runAuthServer(t, taas.AuthNone)
	defer authCleanup(server, image)
	expectedRunRkt(ctx, t, server.URL, "sanity", authSuccessfulDownload)
}

const (
	// It cannot have any spaces, because of an actool limitation
	// - if we pass an `--exec=/inspect
	// --print-msg='Authentication succeeded.'` string to actool
	// it will split the --exec value by spaces, disregarding the
	// single quotes around "Authentication succeeded.". The
	// result is that /inspect receives two parameters -
	// "--print-msg='Authentication" and "succeeded.'"
	authSuccessfulDownload = "AuthenticationSucceeded."
	authFailedDownload     = "bad HTTP status code: 401"
	authACIName            = "rkt-inspect-auth-test.aci"
)

type authConfDir int

const (
	authConfDirNone authConfDir = iota
	authConfDirLocal
	authConfDirSystem
)

type genericAuthTest struct {
	name         string
	confDir      authConfDir
	expectedLine string
}

func TestAuthBasic(t *testing.T) {
	tests := []genericAuthTest{
		{"basic-no-config", authConfDirNone, authFailedDownload},
		{"basic-local-config", authConfDirLocal, authSuccessfulDownload},
		{"basic-system-config", authConfDirSystem, authSuccessfulDownload},
	}
	testAuthGeneric(t, taas.AuthBasic, tests)
}

func TestAuthOauth(t *testing.T) {
	tests := []genericAuthTest{
		{"oauth-no-config", authConfDirNone, authFailedDownload},
		{"oauth-local-config", authConfDirLocal, authSuccessfulDownload},
		{"oauth-system-config", authConfDirSystem, authSuccessfulDownload},
	}
	testAuthGeneric(t, taas.AuthOauth, tests)
}

func testAuthGeneric(t *testing.T, auth taas.AuthType, tests []genericAuthTest) {
	server, image := runAuthServer(t, auth)
	defer authCleanup(server, image)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()
	for _, tt := range tests {
		switch tt.confDir {
		case authConfDirNone:
			// no config to write
		case authConfDirLocal:
			writeConfig(t, authDir(ctx.LocalDir()), "test.json", server.Conf)
		case authConfDirSystem:
			writeConfig(t, authDir(ctx.SystemDir()), "test.json", server.Conf)
		default:
			panic("Wrong config directory")
		}
		expectedRunRkt(ctx, t, server.URL, tt.name, tt.expectedLine)
		ctx.Reset()
	}
}

func TestAuthOverride(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()
	server, image := runAuthServer(t, taas.AuthOauth)
	defer authCleanup(server, image)
	hash := "sha512-" + getHashOrPanic(image)
	tests := []struct {
		systemConfig         string
		localConfig          string
		name                 string
		resultBeforeOverride string
		resultAfterOverride  string
	}{
		{server.Conf, getInvalidOAuthConfig(server.Conf), "valid-system-invalid-local", authSuccessfulDownload, authFailedDownload},
		{getInvalidOAuthConfig(server.Conf), server.Conf, "invalid-system-valid-local", authFailedDownload, authSuccessfulDownload},
	}
	for _, tt := range tests {
		writeConfig(t, authDir(ctx.SystemDir()), "test.json", tt.systemConfig)
		expectedRunRkt(ctx, t, server.URL, tt.name+"-1", tt.resultBeforeOverride)
		if tt.resultBeforeOverride == authSuccessfulDownload {
			// Remove the image from the store since it was fetched in the
			// previous run and the test aci-server returns a
			// Cache-Control max-age header
			removeFromCas(t, ctx, hash)
		}
		writeConfig(t, authDir(ctx.LocalDir()), "test.json", tt.localConfig)
		expectedRunRkt(ctx, t, server.URL, tt.name+"-2", tt.resultAfterOverride)
		ctx.Reset()
	}
}

func TestAuthIgnore(t *testing.T) {
	server, image := runAuthServer(t, taas.AuthOauth)
	defer authCleanup(server, image)
	testAuthIgnoreBogusFiles(t, server)
	testAuthIgnoreSubdirectories(t, server)
}

func testAuthIgnoreBogusFiles(t *testing.T, server *taas.Server) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()
	writeConfig(t, authDir(ctx.SystemDir()), "README", "This is system config")
	writeConfig(t, authDir(ctx.LocalDir()), "README", "This is local config")
	writeConfig(t, authDir(ctx.SystemDir()), "test.notjson", server.Conf)
	writeConfig(t, authDir(ctx.LocalDir()), "test.notjson", server.Conf)
	expectedRunRkt(ctx, t, server.URL, "oauth-bogus-files", authFailedDownload)
}

func testAuthIgnoreSubdirectories(t *testing.T, server *taas.Server) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()
	localSubdir := filepath.Join(ctx.LocalDir(), "subdir")
	systemSubdir := filepath.Join(ctx.SystemDir(), "subdir")
	writeConfig(t, authDir(localSubdir), "test.json", server.Conf)
	writeConfig(t, authDir(systemSubdir), "test.json", server.Conf)
	expectedRunRkt(ctx, t, server.URL, "oauth-subdirectories", authFailedDownload)
}

func runAuthServer(t *testing.T, auth taas.AuthType) (*taas.Server, string) {
	setup := taas.GetDefaultServerSetup()
	setup.Auth = auth
	setup.Port = taas.PortRandom
	server := runServer(t, setup)
	image := patchTestACI(authACIName, fmt.Sprintf("--exec=/inspect --print-msg='%s'", authSuccessfulDownload))
	fileSet := make(map[string]string, 1)
	fileSet[authACIName] = image
	if err := server.UpdateFileSet(fileSet); err != nil {
		t.Fatalf("Failed to populate a file list in test aci server: %v", err)
	}
	return server, image
}

func authCleanup(server *taas.Server, image string) {
	server.Close()
	_ = os.Remove(image)
}

// expectedRunRkt tries to fetch and run an auth test ACI from host.
func expectedRunRkt(ctx *testutils.RktRunCtx, t *testing.T, host, testName, line string) {
	t.Logf("test name: %s", testName)
	// First, check that --insecure-options=image,tls is required
	// The server does not provide signatures for now.
	cmd := fmt.Sprintf(`%s --debug run --no-store --mds-register=false %s/%s`, ctx.Cmd(), host, authACIName)
	child := spawnOrFail(t, cmd)
	defer child.Wait()
	signatureErrorLine := "error downloading the signature file"
	if err := expectWithOutput(child, signatureErrorLine); err != nil {
		t.Fatalf("Didn't receive expected output %q: %v", signatureErrorLine, err)
	}

	// Then, run with --insecure-options=image,tls
	cmd = fmt.Sprintf(`%s --debug --insecure-options=image,tls run --no-store --mds-register=false %s/%s`, ctx.Cmd(), host, authACIName)
	child = spawnOrFail(t, cmd)
	defer child.Wait()
	if err := expectWithOutput(child, line); err != nil {
		t.Fatalf("Didn't receive expected output %q: %v", line, err)
	}
}

func getInvalidOAuthConfig(conf string) string {
	return strings.Replace(conf, "sometoken", "someobviouslywrongtoken", 1)
}
