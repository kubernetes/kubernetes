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

package main

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/tests/testutils"
	taas "github.com/coreos/rkt/tests/testutils/aci-server"
)

// TestFetchFromFile tests that 'rkt fetch/run/prepare' for a file will always
// fetch the file regardless of the specified behavior (default, store only,
// remote only).
func TestFetchFromFile(t *testing.T) {
	image := "rkt-inspect-implicit-fetch.aci"
	imagePath := patchTestACI(image, "--exec=/inspect")

	defer os.Remove(imagePath)

	tests := []struct {
		args  string
		image string
	}{
		{"--insecure-options=image fetch", imagePath},
		{"--insecure-options=image fetch --store-only", imagePath},
		{"--insecure-options=image fetch --no-store", imagePath},
		{"--insecure-options=image run --mds-register=false", imagePath},
		{"--insecure-options=image run --mds-register=false --store-only", imagePath},
		{"--insecure-options=image run --mds-register=false --no-store", imagePath},
		{"--insecure-options=image prepare", imagePath},
		{"--insecure-options=image prepare --store-only", imagePath},
		{"--insecure-options=image prepare --no-store", imagePath},
	}

	for _, tt := range tests {
		testFetchFromFile(t, tt.args, tt.image)
	}
}

func testFetchFromFile(t *testing.T, arg string, image string) {
	fetchFromFileMsg := fmt.Sprintf("using image from file %s", image)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s %s %s", ctx.Cmd(), arg, image)

	// 1. Run cmd, should get $fetchFromFileMsg.
	child := spawnOrFail(t, cmd)
	if err := expectWithOutput(child, fetchFromFileMsg); err != nil {
		t.Fatalf("%q should be found: %v", fetchFromFileMsg, err)
	}
	child.Wait()

	// 1. Run cmd again, should get $fetchFromFileMsg.
	runRktAndCheckOutput(t, cmd, fetchFromFileMsg, false)
}

// TestFetch tests that 'rkt fetch/run/prepare' for any type (image name string
// or URL) except file:// URL will work with the default, store only
// (--store-only) and remote only (--no-store) behaviors.
func TestFetch(t *testing.T) {
	image := "rkt-inspect-implicit-fetch.aci"
	imagePath := patchTestACI(image, "--exec=/inspect")

	defer os.Remove(imagePath)

	tests := []struct {
		args      string
		image     string
		imageArgs string
		finalURL  string
	}{
		{"--insecure-options=image fetch", "coreos.com/etcd:v2.1.2", "", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		{"--insecure-options=image fetch", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci", "", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		{"--insecure-options=image fetch", "docker://busybox", "", "docker://busybox"},
		{"--insecure-options=image fetch", "docker://busybox:latest", "", "docker://busybox:latest"},
		{"--insecure-options=image run --mds-register=false", "coreos.com/etcd:v2.1.2", "--exec /dev/null", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		{"--insecure-options=image run --mds-register=false", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci", "--exec /dev/null", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		{"--insecure-options=image run --mds-register=false", "docker://busybox", "", "docker://busybox"},
		{"--insecure-options=image run --mds-register=false", "docker://busybox:latest", "", "docker://busybox:latest"},
		{"--insecure-options=image prepare", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci", "", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		{"--insecure-options=image prepare", "coreos.com/etcd:v2.1.2", "", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		// test --insecure-options=tls to make sure
		// https://github.com/coreos/rkt/issues/1829 is not an issue anymore
		{"--insecure-options=image,tls prepare", "docker://busybox", "", "docker://busybox"},
		{"--insecure-options=image prepare", "docker://busybox:latest", "", "docker://busybox:latest"},
	}

	for _, tt := range tests {
		testFetchDefault(t, tt.args, tt.image, tt.imageArgs, tt.finalURL)
		testFetchStoreOnly(t, tt.args, tt.image, tt.imageArgs, tt.finalURL)
		testFetchNoStore(t, tt.args, tt.image, tt.imageArgs, tt.finalURL)
	}
}

func TestFetchFullHash(t *testing.T) {
	imagePath := getInspectImagePath()

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	tests := []struct {
		fetchArgs          string
		expectedHashLength int
	}{
		{"", len("sha512-") + 32},
		{"--full", len("sha512-") + 64},
	}

	for _, tt := range tests {
		hash := importImageAndFetchHash(t, ctx, tt.fetchArgs, imagePath)
		if len(hash) != tt.expectedHashLength {
			t.Fatalf("expected hash length of %d, got %d", tt.expectedHashLength, len(hash))
		}
	}
}

func testFetchDefault(t *testing.T, arg string, image string, imageArgs string, finalURL string) {
	remoteFetchMsgTpl := `remote fetching from URL %q`
	storeMsgTpl := `using image from local store for .* %s`
	if finalURL == "" {
		finalURL = image
	}
	remoteFetchMsg := fmt.Sprintf(remoteFetchMsgTpl, finalURL)
	storeMsg := fmt.Sprintf(storeMsgTpl, image)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s %s %s %s", ctx.Cmd(), arg, image, imageArgs)

	// 1. Run cmd with the image not available in the store, should get $remoteFetchMsg.
	child := spawnOrFail(t, cmd)
	if err := expectWithOutput(child, remoteFetchMsg); err != nil {
		t.Fatalf("%q should be found: %v", remoteFetchMsg, err)
	}
	child.Wait()

	// 2. Run cmd with the image available in the store, should get $storeMsg.
	runRktAndCheckRegexOutput(t, cmd, storeMsg)
}

func testFetchStoreOnly(t *testing.T, args string, image string, imageArgs string, finalURL string) {
	cannotFetchMsgTpl := `unable to fetch.* image from .* %q`
	storeMsgTpl := `using image from local store for .* %s`
	cannotFetchMsg := fmt.Sprintf(cannotFetchMsgTpl, image)
	storeMsg := fmt.Sprintf(storeMsgTpl, image)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --store-only %s %s %s", ctx.Cmd(), args, image, imageArgs)

	// 1. Run cmd with the image not available in the store should get $cannotFetchMsg.
	runRktAndCheckRegexOutput(t, cmd, cannotFetchMsg)

	importImageAndFetchHash(t, ctx, "", image)

	// 2. Run cmd with the image available in the store, should get $storeMsg.
	runRktAndCheckRegexOutput(t, cmd, storeMsg)
}

func testFetchNoStore(t *testing.T, args string, image string, imageArgs string, finalURL string) {
	remoteFetchMsgTpl := `remote fetching from URL %q`
	remoteFetchMsg := fmt.Sprintf(remoteFetchMsgTpl, finalURL)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	importImageAndFetchHash(t, ctx, "", image)

	cmd := fmt.Sprintf("%s --no-store %s %s %s", ctx.Cmd(), args, image, imageArgs)

	// 1. Run cmd with the image available in the store, should get $remoteFetchMsg.
	child := spawnOrFail(t, cmd)
	if err := expectWithOutput(child, remoteFetchMsg); err != nil {
		t.Fatalf("%q should be found: %v", remoteFetchMsg, err)
	}
	child.Wait()
}

type synchronizedBool struct {
	value bool
	lock  sync.Mutex
}

func (b *synchronizedBool) Read() bool {
	b.lock.Lock()
	value := b.value
	b.lock.Unlock()
	return value
}

func (b *synchronizedBool) Write(value bool) {
	b.lock.Lock()
	b.value = value
	b.lock.Unlock()
}

func TestResumedFetch(t *testing.T) {
	image := "rkt-inspect-implicit-fetch.aci"
	imagePath := patchTestACI(image, "--exec=/inspect")
	defer os.Remove(imagePath)

	hash := types.ShortHash("sha512-" + getHashOrPanic(imagePath))

	kill := make(chan struct{})
	reportkill := make(chan struct{})

	shouldInterrupt := &synchronizedBool{}
	shouldInterrupt.Write(true)

	server := httptest.NewServer(testServerHandler(t, shouldInterrupt, imagePath, kill, reportkill))
	defer server.Close()

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --no-store --insecure-options=image fetch %s", ctx.Cmd(), server.URL)
	child := spawnOrFail(t, cmd)
	<-kill
	err := child.Close()
	if err != nil {
		panic(err)
	}
	reportkill <- struct{}{}

	// rkt has fetched the first half of the image
	// If it fetches the first half again these channels will be written to.
	// Closing them to make the test panic if they're written to.
	close(kill)
	close(reportkill)

	child = spawnOrFail(t, cmd)
	if _, _, err := expectRegexWithOutput(child, ".*"+hash); err != nil {
		t.Fatalf("hash didn't match: %v", err)
	}
	waitOrFail(t, child, 0)
}

func TestResumedFetchInvalidCache(t *testing.T) {
	image := "rkt-inspect-implicit-fetch.aci"
	imagePath := patchTestACI(image, "--exec=/inspect")
	defer os.Remove(imagePath)

	hash := types.ShortHash("sha512-" + getHashOrPanic(imagePath))

	kill := make(chan struct{})
	reportkill := make(chan struct{})

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	shouldInterrupt := &synchronizedBool{}
	shouldInterrupt.Write(true)

	// Fetch the first half of the image, and kill rkt once it reaches halfway.
	server := httptest.NewServer(testServerHandler(t, shouldInterrupt, imagePath, kill, reportkill))
	defer server.Close()
	cmd := fmt.Sprintf("%s --no-store --insecure-options=image fetch %s", ctx.Cmd(), server.URL)
	child := spawnOrFail(t, cmd)
	<-kill
	err := child.Close()
	if err != nil {
		panic(err)
	}
	reportkill <- struct{}{}

	// Fetch the image again. The server doesn't support Etags or the
	// Last-Modified header, so the cached version should be invalidated. If
	// rkt tries to use the cache, the hash won't check out.
	shouldInterrupt.Write(false)
	child = spawnOrFail(t, cmd)
	if _, s, err := expectRegexWithOutput(child, ".*"+hash); err != nil {
		t.Fatalf("hash didn't match: %v\nin: %s", err, s)
	}
	waitOrFail(t, child, 0)
}

func testServerHandler(t *testing.T, shouldInterrupt *synchronizedBool, imagePath string, kill, waitforkill chan struct{}) http.HandlerFunc {
	interruptingHandler := testInterruptingServerHandler(t, imagePath, kill, waitforkill)
	simpleHandler := testSimpleServerHandler(t, imagePath)

	return func(w http.ResponseWriter, r *http.Request) {
		if shouldInterrupt.Read() {
			interruptingHandler(w, r)
		} else {
			simpleHandler(w, r)
		}
	}
}

func testInterruptingServerHandler(t *testing.T, imagePath string, kill, waitforkill chan struct{}) http.HandlerFunc {
	finfo, err := os.Stat(imagePath)
	if err != nil {
		panic(err)
	}
	cutoff := finfo.Size() / 2
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			headers := w.Header()
			headers["Accept-Ranges"] = []string{"bytes"}
			headers["Last-Modified"] = []string{"Mon, 02 Jan 2006 15:04:05 MST"}
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method != "GET" {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		file, err := os.Open(imagePath)
		if err != nil {
			panic(err)
		}
		defer file.Close()

		rangeHeaders, ok := r.Header["Range"]
		if ok && len(rangeHeaders) == 1 && strings.HasPrefix(rangeHeaders[0], "bytes=") {
			rangeHeader := rangeHeaders[0][6:] // The first (and only) range header, with len("bytes=") characters chopped off the front
			tokens := strings.Split(rangeHeader, "-")
			if len(tokens) != 2 {
				t.Fatalf("couldn't parse range header: %q", rangeHeader)
			}

			start, err := strconv.Atoi(tokens[0])
			if err != nil {
				if tokens[0] == "" {
					start = 0 // If start wasn't specified, start at the beginning
				} else {
					t.Fatalf("requested non-int starting location: %s", tokens[0])
				}
			}
			end, err := strconv.Atoi(tokens[1])
			if err != nil {
				if tokens[1] == "" {
					end = int(finfo.Size()) - 1 // If end wasn't specified, end at the end
				} else {
					t.Fatalf("requested non-int ending location: %s", tokens[0])
				}
			}

			_, err = file.Seek(int64(start), os.SEEK_SET)
			if err != nil {
				panic(err)
			}

			_, err = io.CopyN(w, file, int64(end-start+1))
			if err != nil {
				panic(err)
			}

			return
		}

		_, err = io.CopyN(w, file, cutoff)
		if err != nil {
			panic(err)
		}

		kill <- struct{}{}
		<-waitforkill
	}
}

func testSimpleServerHandler(t *testing.T, imagePath string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "HEAD" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method != "GET" {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		file, err := os.Open(imagePath)
		if err != nil {
			panic(err)
		}
		defer file.Close()

		_, err = io.Copy(w, file)
		if err != nil {
			panic(err)
		}
	}
}

func TestDeferredSignatureDownload(t *testing.T) {
	imageName := "localhost/rkt-inspect-deferred-signature-download"
	imageFileName := fmt.Sprintf("%s.aci", filepath.Base(imageName))
	// no spaces between words, because of an actool limitation
	successMsg := "deferredSignatureDownloadWasSuccessful"

	args := []string{
		fmt.Sprintf("--exec=/inspect --print-msg='%s'", successMsg),
		fmt.Sprintf("--name=%s", imageName),
	}
	image := patchTestACI(imageFileName, args...)
	defer os.Remove(image)

	asc := runSignImage(t, image, 1)
	defer os.Remove(asc)
	ascBase := filepath.Base(asc)

	setup := taas.GetDefaultServerSetup()
	setup.Server = taas.ServerQuay
	server := runServer(t, setup)
	defer server.Close()
	fileSet := make(map[string]string, 2)
	fileSet[imageFileName] = image
	fileSet[ascBase] = asc
	if err := server.UpdateFileSet(fileSet); err != nil {
		t.Fatalf("Failed to populate a file list in test aci server: %v", err)
	}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	runRktTrust(t, ctx, "", 1)

	runCmd := fmt.Sprintf("%s --debug --insecure-options=tls run %s", ctx.Cmd(), imageName)
	child := spawnOrFail(t, runCmd)
	defer waitOrFail(t, child, 0)

	expectedMessages := []string{
		"server requested deferring the signature download",
		successMsg,
	}
	for _, msg := range expectedMessages {
		if err := expectWithOutput(child, msg); err != nil {
			t.Fatalf("Could not find expected msg %q, output follows:\n%v", msg, err)
		}
	}
}
