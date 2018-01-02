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
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/aci/acitest"
	"github.com/coreos/rkt/tests/testutils"
	taas "github.com/coreos/rkt/tests/testutils/aci-server"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
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
		{"--insecure-options=image --debug fetch", imagePath},
		{"--insecure-options=image --debug fetch --store-only", imagePath},
		{"--insecure-options=image --debug fetch --no-store", imagePath},
		{"--insecure-options=image --debug run --mds-register=false", imagePath},
		{"--insecure-options=image --debug run --mds-register=false --store-only", imagePath},
		{"--insecure-options=image --debug run --mds-register=false --no-store", imagePath},
		{"--insecure-options=image --debug prepare", imagePath},
		{"--insecure-options=image --debug prepare --store-only", imagePath},
		{"--insecure-options=image --debug prepare --no-store", imagePath},
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
	waitOrFail(t, child, 0)

	// 1. Run cmd again, should get $fetchFromFileMsg.
	runRktAndCheckOutput(t, cmd, fetchFromFileMsg, false)
}

// TestFetchAny tests that 'rkt fetch/run/prepare' for any type (image name string
// or URL) except file:// URL will work with the default, store only
// (--store-only) and remote only (--no-store) behaviors.
func TestFetchAny(t *testing.T) {
	image := "rkt-inspect-implicit-fetch.aci"
	imagePath := patchTestACI(image, "--exec=/inspect")

	defer os.Remove(imagePath)

	tests := []struct {
		args      string
		image     string
		imageArgs string
		finalURL  string
	}{
		{"--insecure-options=image --debug fetch", "coreos.com/etcd:v2.1.2", "", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		{"--insecure-options=image --debug fetch", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci", "", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		{"--insecure-options=image --debug fetch", "docker://busybox", "", "docker://busybox"},
		{"--insecure-options=image --debug fetch", "docker://busybox:latest", "", "docker://busybox:latest"},
		{"--insecure-options=image --debug run --mds-register=false", "coreos.com/etcd:v2.1.2", "--exec /etcdctl", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		{"--insecure-options=image --debug run --mds-register=false", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci", "--exec /etcdctl", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		{"--insecure-options=image --debug run --mds-register=false", "docker://busybox", "", "docker://busybox"},
		{"--insecure-options=image --debug run --mds-register=false", "docker://busybox:latest", "", "docker://busybox:latest"},
		{"--insecure-options=image --debug prepare", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci", "", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		{"--insecure-options=image --debug prepare", "coreos.com/etcd:v2.1.2", "", "https://github.com/coreos/etcd/releases/download/v2.1.2/etcd-v2.1.2-linux-amd64.aci"},
		// test --insecure-options=tls to make sure
		// https://github.com/coreos/rkt/issues/1829 is not an issue anymore
		{"--insecure-options=image,tls --debug prepare", "docker://busybox", "", "docker://busybox"},
		{"--insecure-options=image --debug prepare", "docker://busybox:latest", "", "docker://busybox:latest"},
	}

	for _, tt := range tests {
		testFetchNew(t, tt.args, tt.image, tt.imageArgs, tt.finalURL)
		testFetchNever(t, tt.args, tt.image, tt.imageArgs, tt.finalURL)
		testFetchUpdate(t, tt.args, tt.image, tt.imageArgs, tt.finalURL)
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
		hash, err := importImageAndFetchHash(t, ctx, tt.fetchArgs, imagePath)
		if err != nil {
			t.Fatalf("%v", err)
		}
		if len(hash) != tt.expectedHashLength {
			t.Fatalf("expected hash length of %d, got %d", tt.expectedHashLength, len(hash))
		}
	}
}

func testFetchNew(t *testing.T, arg string, image string, imageArgs string, finalURL string) {
	remoteFetchMsgTpl := `remote fetching from URL %q`
	storeMsgTpl := `using image from local store for .* %s`
	if finalURL == "" {
		finalURL = image
	}
	remoteFetchMsg := fmt.Sprintf(remoteFetchMsgTpl, finalURL)
	storeMsg := fmt.Sprintf(storeMsgTpl, image)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --pull-policy=new %s %s %s", ctx.Cmd(), arg, image, imageArgs)

	// 1. Run cmd with the image not available in the store, should get $remoteFetchMsg.
	err := runRktAndCheckRegexOutput(t, cmd, remoteFetchMsg)
	status, _ := common.GetExitStatus(err)
	if status != 0 {
		t.Logf("%v", err)
		t.Skip("remote fetching failed, probably a network failure. Skipping...")
	}

	// 2. Run cmd with the image available in the store, should get $storeMsg.
	runRktAndCheckRegexOutput(t, cmd, storeMsg)
}

func testFetchNever(t *testing.T, args string, image string, imageArgs string, finalURL string) {
	cannotFetchMsgTpl := `unable to fetch.* image from .* %q`
	storeMsgTpl := `using image from local store for .* %s`
	cannotFetchMsg := fmt.Sprintf(cannotFetchMsgTpl, image)
	storeMsg := fmt.Sprintf(storeMsgTpl, image)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --pull-policy=never %s %s %s", ctx.Cmd(), args, image, imageArgs)

	// 1. Run cmd with the image not available in the store should get $cannotFetchMsg.
	runRktAndCheckRegexOutput(t, cmd, cannotFetchMsg)

	if _, err := importImageAndFetchHash(t, ctx, "", image); err != nil {
		t.Skip(fmt.Sprintf("%v, probably a network failure. Skipping...", err))
	}

	// 2. Run cmd with the image available in the store, should get $storeMsg.
	runRktAndCheckRegexOutput(t, cmd, storeMsg)
}

func testFetchUpdate(t *testing.T, args string, image string, imageArgs string, finalURL string) {
	remoteFetchMsgTpl := `remote fetching from URL %q`
	remoteFetchMsg := fmt.Sprintf(remoteFetchMsgTpl, finalURL)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	if _, err := importImageAndFetchHash(t, ctx, "", image); err != nil {
		t.Skip(fmt.Sprintf("%v, probably a network failure. Skipping...", err))
	}

	cmd := fmt.Sprintf("%s --pull-policy=update %s %s %s", ctx.Cmd(), args, image, imageArgs)

	// 1. Run cmd with the image available in the store, should get $remoteFetchMsg.
	err := runRktAndCheckRegexOutput(t, cmd, remoteFetchMsg)
	status, _ := common.GetExitStatus(err)
	if status != 0 {
		t.Logf("%v", err)
		t.Skip("remote fetching failed, probably a network failure. Skipping...")
	}

	if err != nil {
		t.Fatalf("%q should be found: %v", remoteFetchMsg, err)
	}
}

func TestFetchNoStoreCacheControl(t *testing.T) {
	imageName := "rkt-inspect-fetch-nostore-cachecontrol"
	imageFileName := fmt.Sprintf("%s.aci", imageName)
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

	tests := []struct {
		imageArg string
		imageURL string
	}{
		{"https://127.0.0.1/" + imageFileName, "https://127.0.0.1/" + imageFileName},
		{"localhost/" + imageName, "https://127.0.0.1:443/localhost/" + imageFileName},
	}

	for _, tt := range tests {
		cmd := fmt.Sprintf("%s --no-store --debug --insecure-options=tls,image fetch %s", ctx.Cmd(), tt.imageArg)
		expectedMessage := fmt.Sprintf("fetching image from %s", tt.imageURL)
		runRktAndCheckRegexOutput(t, cmd, expectedMessage)

		cmd = fmt.Sprintf("%s --no-store --debug --insecure-options=tls,image fetch %s", ctx.Cmd(), tt.imageArg)
		expectedMessage = fmt.Sprintf("image for %s isn't expired, not fetching.", tt.imageURL)
		runRktAndCheckRegexOutput(t, cmd, expectedMessage)

		ctx.Reset()
	}
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

	cmd := fmt.Sprintf("%s --no-store --insecure-options=image fetch %s", ctx.Cmd(), server.URL+"/image.aci")
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
	cmd := fmt.Sprintf("%s --no-store --insecure-options=image fetch %s", ctx.Cmd(), server.URL+"/image.aci")
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

		// sleep a bit before signaling that rkt should be killed since it
		// might not have had time to write everything to disk
		time.Sleep(time.Second)
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

func TestDifferentDiscoveryLabels(t *testing.T) {
	const imageName = "localhost/rkt-test-different-discovery-labels-image"

	manifest, err := acitest.ImageManifestString(&schema.ImageManifest{
		Name: imageName, Labels: types.Labels{
			{"version", "1.2.0"},
			{"arch", "amd64"},
			{"os", "linux"},
		},
	})

	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	emptyImage := getEmptyImagePath()
	tmpDir := mustTempDir("rkt-TestDifferentDiscoveryLabels-")
	defer os.RemoveAll(tmpDir)

	tmpManifest, err := ioutil.TempFile(tmpDir, "manifest")
	if err != nil {
		panic(fmt.Sprintf("Cannot create temp manifest: %v", err))
	}
	if err := ioutil.WriteFile(tmpManifest.Name(), []byte(manifest), 0600); err != nil {
		panic(fmt.Sprintf("Cannot write to temp manifest: %v", err))
	}
	defer os.Remove(tmpManifest.Name())

	imageFileName := fmt.Sprintf("%s.aci", filepath.Base(imageName))
	image := patchACI(emptyImage, imageFileName, "--manifest", tmpManifest.Name())
	defer os.Remove(image)

	asc := runSignImage(t, image, 1)
	defer os.Remove(asc)
	ascBase := filepath.Base(asc)

	setup := taas.GetDefaultServerSetup()
	server := runServer(t, setup)
	defer server.Close()
	fileSet := make(map[string]string, 2)
	fileSet[imageFileName] = image
	fileSet[ascBase] = asc
	if err := server.UpdateFileSet(fileSet); err != nil {
		t.Fatalf("Failed to populate a file list in test aci server: %v", err)
	}

	tests := []struct {
		imageName       string
		expectedMessage string
	}{
		{imageName + ":2.0", fmt.Sprintf("requested value for label %q: %q differs from fetched aci label value: %q", "version", "2.0", "1.2.0")},
		{imageName + ":latest", fmt.Sprintf("requested value for label %q: %q differs from fetched aci label value: %q", "version", "latest", "1.2.0")},
		{imageName + ",arch=armv7b", fmt.Sprintf("requested value for label %q: %q differs from fetched aci label value: %q", "arch", "armv7b", "amd64")},
		{imageName + ",unexistinglabel=bla", fmt.Sprintf("requested label %q not provided by the image manifest", "unexistinglabel")},
	}

	for _, tt := range tests {
		testDifferentDiscoveryNameLabels(t, tt.imageName, tt.expectedMessage)
	}
}

func testDifferentDiscoveryNameLabels(t *testing.T, imageName string, expectedMessage string) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	runRktTrust(t, ctx, "", 1)

	// Since aci-server provided meta tag template doesn't contains
	// {version} {os} or {arch}, we can just ask for any version/os/arch
	// and always get the same ACI
	runCmd := fmt.Sprintf("%s --debug --insecure-options=tls fetch %s", ctx.Cmd(), imageName)
	child := spawnOrFail(t, runCmd)
	defer waitOrFail(t, child, 254)

	if err := expectWithOutput(child, expectedMessage); err != nil {
		t.Fatalf("Could not find expected msg %q, output follows:\n%v", expectedMessage, err)
	}
}
