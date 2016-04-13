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
	"bytes"
	"crypto/sha512"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/api/v1alpha"
	"github.com/coreos/rkt/tests/testutils"
	taas "github.com/coreos/rkt/tests/testutils/aci-server"
	"google.golang.org/grpc"
)

const (
	defaultTimeLayout = "2006-01-02 15:04:05.999 -0700 MST"
	nobodyUid         = 65534
	nobodyGid         = 65534
)

func expectCommon(p *gexpect.ExpectSubprocess, searchString string, timeout time.Duration) error {
	var err error

	p.Capture()
	if timeout == 0 {
		err = p.Expect(searchString)
	} else {
		err = p.ExpectTimeout(searchString, timeout)
	}
	if err != nil {
		return fmt.Errorf(string(p.Collect()))
	}

	return nil
}

func expectWithOutput(p *gexpect.ExpectSubprocess, searchString string) error {
	return expectCommon(p, searchString, 0)
}

func expectRegexWithOutput(p *gexpect.ExpectSubprocess, searchPattern string) ([]string, string, error) {
	return p.ExpectRegexFindWithOutput(searchPattern)
}

func expectRegexTimeoutWithOutput(p *gexpect.ExpectSubprocess, searchPattern string, timeout time.Duration) ([]string, string, error) {
	return p.ExpectTimeoutRegexFindWithOutput(searchPattern, timeout)
}

func expectTimeoutWithOutput(p *gexpect.ExpectSubprocess, searchString string, timeout time.Duration) error {
	return expectCommon(p, searchString, timeout)
}

func patchACI(inputFileName, newFileName string, args ...string) string {
	var allArgs []string

	actool := testutils.GetValueFromEnvOrPanic("ACTOOL")
	tmpDir := testutils.GetValueFromEnvOrPanic("FUNCTIONAL_TMP")

	imagePath, err := filepath.Abs(filepath.Join(tmpDir, newFileName))
	if err != nil {
		panic(fmt.Sprintf("Cannot create ACI: %v\n", err))
	}
	allArgs = append(allArgs, "patch-manifest")
	allArgs = append(allArgs, "--no-compression")
	allArgs = append(allArgs, "--overwrite")
	allArgs = append(allArgs, args...)
	allArgs = append(allArgs, inputFileName)
	allArgs = append(allArgs, imagePath)

	output, err := exec.Command(actool, allArgs...).CombinedOutput()
	if err != nil {
		panic(fmt.Sprintf("Cannot create ACI: %v: %s\n", err, output))
	}
	return imagePath
}

func patchTestACI(newFileName string, args ...string) string {
	image := getInspectImagePath()
	return patchACI(image, newFileName, args...)
}

func spawnOrFail(t *testing.T, cmd string) *gexpect.ExpectSubprocess {
	t.Logf("Running command: %v", cmd)
	child, err := gexpect.Spawn(cmd)
	if err != nil {
		t.Fatalf("Cannot exec rkt: %v", err)
	}
	return child
}

func getExitStatus(err error) int {
	if err == nil {
		return 0
	}
	if exiterr, ok := err.(*exec.ExitError); ok {
		// the program has exited with an exit code != 0
		if status, ok := exiterr.Sys().(syscall.WaitStatus); ok {
			return status.ExitStatus()
		}
	}
	return -1
}

func waitOrFail(t *testing.T, child *gexpect.ExpectSubprocess, expectedStatus int) {
	err := child.Wait()
	status := getExitStatus(err)
	if status != expectedStatus {
		t.Fatalf("rkt terminated with unexpected status %d, expected %d\nOutput:\n%s", status, expectedStatus, child.Collect())
	}
}

func spawnAndWaitOrFail(t *testing.T, cmd string, expectedStatus int) {
	child := spawnOrFail(t, cmd)
	waitOrFail(t, child, expectedStatus)
}

func getEmptyImagePath() string {
	return testutils.GetValueFromEnvOrPanic("RKT_EMPTY_IMAGE")
}

func getInspectImagePath() string {
	return testutils.GetValueFromEnvOrPanic("RKT_INSPECT_IMAGE")
}

func getHashOrPanic(path string) string {
	hash, err := getHash(path)
	if err != nil {
		panic(fmt.Sprintf("Cannot get hash from file located at %v", path))
	}
	return hash
}

func getHash(filePath string) (string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("error opening file: %v", err)
	}

	hash := sha512.New()
	r := io.TeeReader(f, hash)

	if _, err := io.Copy(ioutil.Discard, r); err != nil {
		return "", fmt.Errorf("error reading file: %v", err)
	}

	return hex.EncodeToString(hash.Sum(nil)), nil
}

func createTempDirOrPanic(dirName string) string {
	tmpDir, err := ioutil.TempDir("", dirName)
	if err != nil {
		panic(fmt.Sprintf("Cannot create temp dir: %v", err))
	}
	return tmpDir
}

func importImageAndFetchHashAsGid(t *testing.T, ctx *testutils.RktRunCtx, img string, fetchArgs string, gid int) string {
	// Import the test image into store manually.
	cmd := fmt.Sprintf("%s --insecure-options=image,tls fetch %s %s", ctx.Cmd(), fetchArgs, img)

	// TODO(jonboulle): non-root user breaks trying to read root-written
	// config directories. Should be a better way to approach this. Should
	// config directories be readable by the rkt group too?
	if gid != 0 {
		cmd = fmt.Sprintf("%s --insecure-options=image,tls fetch %s %s", ctx.CmdNoConfig(), fetchArgs, img)
	}
	child, err := gexpect.Command(cmd)
	if err != nil {
		t.Fatalf("cannot create rkt command: %v", err)
	}
	if gid != 0 {
		child.Cmd.SysProcAttr = &syscall.SysProcAttr{}
		child.Cmd.SysProcAttr.Credential = &syscall.Credential{Uid: uint32(nobodyUid), Gid: uint32(gid)}
	}

	err = child.Start()
	if err != nil {
		t.Fatalf("cannot exec rkt: %v", err)
	}

	// Read out the image hash.
	result, out, err := expectRegexWithOutput(child, "sha512-[0-9a-f]{32,64}")
	if err != nil || len(result) != 1 {
		t.Fatalf("Error: %v\nOutput: %v", err, out)
	}

	waitOrFail(t, child, 0)

	return result[0]
}

func importImageAndFetchHash(t *testing.T, ctx *testutils.RktRunCtx, fetchArgs string, img string) string {
	return importImageAndFetchHashAsGid(t, ctx, fetchArgs, img, 0)
}

func patchImportAndFetchHash(image string, patches []string, t *testing.T, ctx *testutils.RktRunCtx) string {
	imagePath := patchTestACI(image, patches...)
	defer os.Remove(imagePath)

	return importImageAndFetchHash(t, ctx, "", imagePath)
}

func patchImportAndRun(image string, patches []string, t *testing.T, ctx *testutils.RktRunCtx) {
	imagePath := patchTestACI(image, patches...)
	defer os.Remove(imagePath)

	cmd := fmt.Sprintf("%s --insecure-options=image run %s", ctx.Cmd(), imagePath)
	spawnAndWaitOrFail(t, cmd, 0)
}

func runGC(t *testing.T, ctx *testutils.RktRunCtx) {
	cmd := fmt.Sprintf("%s gc --grace-period=0s", ctx.Cmd())
	spawnAndWaitOrFail(t, cmd, 0)
}

func runImageGC(t *testing.T, ctx *testutils.RktRunCtx) {
	cmd := fmt.Sprintf("%s image gc", ctx.Cmd())
	spawnAndWaitOrFail(t, cmd, 0)
}

func removeFromCas(t *testing.T, ctx *testutils.RktRunCtx, hash string) {
	cmd := fmt.Sprintf("%s image rm %s", ctx.Cmd(), hash)
	spawnAndWaitOrFail(t, cmd, 0)
}

func runRktAndGetUUID(t *testing.T, rktCmd string) string {
	child := spawnOrFail(t, rktCmd)
	defer waitOrFail(t, child, 0)

	result, out, err := expectRegexWithOutput(child, "\n[0-9a-f-]{36}")
	if err != nil || len(result) != 1 {
		t.Fatalf("Error: %v\nOutput: %v", err, out)
	}

	podIDStr := strings.TrimSpace(result[0])
	podID, err := types.NewUUID(podIDStr)
	if err != nil {
		t.Fatalf("%q is not a valid UUID: %v", podIDStr, err)
	}

	return podID.String()
}

func runRktAsGidAndCheckOutput(t *testing.T, rktCmd, expectedLine string, expectError bool, gid int) {
	runRktAsUidGidAndCheckOutput(t, rktCmd, expectedLine, expectError, nobodyUid, gid)
}

func runRktAsUidGidAndCheckOutput(t *testing.T, rktCmd, expectedLine string, expectError bool, uid, gid int) {
	child, err := gexpect.Command(rktCmd)
	if err != nil {
		t.Fatalf("cannot exec rkt: %v", err)
	}
	if gid != 0 {
		child.Cmd.SysProcAttr = &syscall.SysProcAttr{}
		child.Cmd.SysProcAttr.Credential = &syscall.Credential{Uid: uint32(uid), Gid: uint32(gid)}
	}

	err = child.Start()
	if err != nil {
		t.Fatalf("cannot start rkt: %v", err)
	}
	expectedStatus := 0
	if expectError {
		expectedStatus = 1
	}
	defer waitOrFail(t, child, expectedStatus)

	if expectedLine != "" {
		if err := expectWithOutput(child, expectedLine); err != nil {
			t.Fatalf("didn't receive expected output %q: %v", expectedLine, err)
		}
	}
}

func startRktAsGidAndCheckOutput(t *testing.T, rktCmd, expectedLine string, gid int) *gexpect.ExpectSubprocess {
	child, err := gexpect.Command(rktCmd)
	if err != nil {
		t.Fatalf("cannot exec rkt: %v", err)
	}
	if gid != 0 {
		child.Cmd.SysProcAttr = &syscall.SysProcAttr{}
		child.Cmd.SysProcAttr.Credential = &syscall.Credential{Uid: uint32(nobodyUid), Gid: uint32(gid)}
	}

	if err := child.Start(); err != nil {
		t.Fatalf("cannot exec rkt: %v", err)
	}

	if expectedLine != "" {
		if err := expectWithOutput(child, expectedLine); err != nil {
			t.Fatalf("didn't receive expected output %q: %v", expectedLine, err)
		}
	}
	return child
}

func runRktAndCheckRegexOutput(t *testing.T, rktCmd, match string) {
	child := spawnOrFail(t, rktCmd)
	defer child.Wait()

	result, out, err := expectRegexWithOutput(child, match)
	if err != nil || len(result) != 1 {
		t.Fatalf("%q regex must be found one time, Error: %v\nOutput: %v", match, err, out)
	}
}

func runRktAndCheckOutput(t *testing.T, rktCmd, expectedLine string, expectError bool) {
	runRktAsGidAndCheckOutput(t, rktCmd, expectedLine, expectError, 0)
}

func startRktAndCheckOutput(t *testing.T, rktCmd, expectedLine string) *gexpect.ExpectSubprocess {
	return startRktAsGidAndCheckOutput(t, rktCmd, expectedLine, 0)
}

func checkAppStatus(t *testing.T, ctx *testutils.RktRunCtx, multiApps bool, appName, expected string) {
	cmd := fmt.Sprintf(`/bin/sh -c "`+
		`UUID=$(%s list --full|grep '%s'|awk '{print $1}') ;`+
		`echo -n 'status=' ;`+
		`%s status $UUID|grep '^app-%s.*=[0-9]*$'|cut -d= -f2"`,
		ctx.Cmd(), appName, ctx.Cmd(), appName)

	if multiApps {
		cmd = fmt.Sprintf(`/bin/sh -c "`+
			`UUID=$(%s list --full|grep '^[a-f0-9]'|awk '{print $1}') ;`+
			`echo -n 'status=' ;`+
			`%s status $UUID|grep '^app-%s.*=[0-9]*$'|cut -d= -f2"`,
			ctx.Cmd(), ctx.Cmd(), appName)
	}

	t.Logf("Get status for app %s\n", appName)
	child := spawnOrFail(t, cmd)
	defer waitOrFail(t, child, 0)

	if err := expectWithOutput(child, expected); err != nil {
		// For debugging purposes, print the full output of
		// "rkt list" and "rkt status"
		cmd := fmt.Sprintf(`%s list --full ;`+
			`UUID=$(%s list --full|grep  '^[a-f0-9]'|awk '{print $1}') ;`+
			`%s status $UUID`,
			ctx.Cmd(), ctx.Cmd(), ctx.Cmd())
		out, err2 := exec.Command("/bin/sh", "-c", cmd).CombinedOutput()
		if err2 != nil {
			t.Logf("Could not run rkt status: %v. %s", err2, out)
		} else {
			t.Logf("%s\n", out)
		}

		t.Fatalf("Failed to get the status for app %s: expected: %s. %v",
			appName, expected, err)
	}
}

type imageInfo struct {
	id         string
	name       string
	version    string
	importTime int64
	size       int64
	manifest   []byte
}

type appInfo struct {
	name     string
	exitCode int
	image    *imageInfo
	// TODO(yifan): Add app state.
}

type networkInfo struct {
	name string
	ipv4 string
}

type podInfo struct {
	id       string
	pid      int
	state    string
	apps     map[string]*appInfo
	networks map[string]*networkInfo
	manifest []byte
}

// parsePodInfo parses the 'rkt status $UUID' result into podInfo struct.
// For example, the 'result' can be:
// state=running
// networks=default:ip4=172.16.28.103
// pid=14352
// exited=false
func parsePodInfoOutput(t *testing.T, result string, p *podInfo) {
	lines := strings.Split(strings.TrimSuffix(result, "\n"), "\n")
	for _, line := range lines {
		tuples := strings.SplitN(line, "=", 2)
		if len(tuples) != 2 {
			t.Fatalf("Unexpected line: %v", line)
		}

		switch tuples[0] {
		case "state":
			p.state = tuples[1]
		case "networks":
			networks := strings.Split(tuples[1], ",")
			for _, n := range networks {
				fields := strings.Split(n, ":")
				if len(fields) != 2 {
					t.Fatalf("Unexpected network info format: %v", n)
				}

				ip4 := strings.Split(fields[1], "=")
				if len(ip4) != 2 {
					t.Fatalf("Unexpected network info format: %v", n)
				}

				networkName := fields[0]
				p.networks[networkName] = &networkInfo{
					name: networkName,
					ipv4: ip4[1],
				}
			}
		case "pid":
			pid, err := strconv.Atoi(tuples[1])
			if err != nil {
				t.Fatalf("Cannot parse the pod's pid %q: %v", tuples[1], err)
			}
			p.pid = pid
		}
		if strings.HasPrefix(tuples[0], "app-") {
			exitCode, err := strconv.Atoi(tuples[1])
			if err != nil {
				t.Fatalf("cannot parse exit code from %q : %v", tuples[1], err)
			}
			appName := strings.TrimPrefix(tuples[0], "app-")

			for _, app := range p.apps {
				if app.name == appName {
					app.exitCode = exitCode
					break
				}
			}
		}
	}
}

func getPodDir(t *testing.T, ctx *testutils.RktRunCtx, podID string) string {
	podsDir := path.Join(ctx.DataDir(), "pods")

	dirs, err := ioutil.ReadDir(podsDir)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	for _, dir := range dirs {
		podDir := path.Join(podsDir, dir.Name(), podID)
		if _, err := os.Stat(podDir); err == nil {
			return podDir
		}
	}
	t.Fatalf("Failed to find pod directory for pod %q", podID)
	return ""
}

// getPodInfo returns the pod info for the given pod ID.
func getPodInfo(t *testing.T, ctx *testutils.RktRunCtx, podID string) *podInfo {
	p := &podInfo{
		id:       podID,
		apps:     make(map[string]*appInfo),
		networks: make(map[string]*networkInfo),
	}

	// Read pod manifest.
	output, err := exec.Command("/bin/bash", "-c", fmt.Sprintf("%s cat-manifest %s", ctx.Cmd(), podID)).CombinedOutput()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Trim the last '\n' character.
	p.manifest = bytes.TrimSpace(output)

	// Fill app infos.
	var manifest schema.PodManifest
	if err := json.Unmarshal(p.manifest, &manifest); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	for _, app := range manifest.Apps {
		appName := app.Name.String()
		p.apps[appName] = &appInfo{
			name: appName,
			// TODO(yifan): Get the image's name.
			image: &imageInfo{id: app.Image.ID.String()},
		}
	}

	// Fill other infos.
	output, err = exec.Command("/bin/bash", "-c", fmt.Sprintf("%s status %s", ctx.Cmd(), podID)).CombinedOutput()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	parsePodInfoOutput(t, string(output), p)

	return p
}

// parseImageInfoOutput parses the 'rkt image list' result into imageInfo struct.
// For example, the 'result' can be:
// 'sha512-e9b77714dbbfda12cb9e136318b103a6f0ce082004d09d0224a620d2bbf38133 nginx:latest 2015-10-16 17:42:57.741 -0700 PDT true'
func parseImageInfoOutput(t *testing.T, result string) *imageInfo {
	fields := regexp.MustCompile("\t+").Split(result, 6)
	nameVersion := strings.Split(fields[1], ":")
	if len(nameVersion) != 2 {
		t.Fatalf("Failed to parse name version string: %q", fields[1])
	}
	importTime, err := time.Parse(defaultTimeLayout, fields[2])
	if err != nil {
		t.Fatalf("Failed to parse time string: %q", fields[2])
	}
	size, err := strconv.Atoi(fields[4])
	if err != nil {
		t.Fatalf("Failed to parse image size string: %q", fields[4])
	}

	return &imageInfo{
		id:         fields[0],
		name:       nameVersion[0],
		version:    nameVersion[1],
		importTime: importTime.Unix(),
		size:       int64(size),
	}
}

// getImageInfo returns the image info for the given image ID.
func getImageInfo(t *testing.T, ctx *testutils.RktRunCtx, imageID string) *imageInfo {
	output, err := exec.Command("/bin/bash", "-c", fmt.Sprintf("%s image list --full | grep %s", ctx.Cmd(), imageID)).CombinedOutput()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	imgInfo := parseImageInfoOutput(t, string(output))

	// Get manifest
	output, err = exec.Command("/bin/bash", "-c", fmt.Sprintf("%s image cat-manifest %s", ctx.Cmd(), imageID)).CombinedOutput()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	imgInfo.manifest = bytes.TrimSuffix(output, []byte{'\n'})
	return imgInfo
}

func newAPIClientOrFail(t *testing.T, address string) (v1alpha.PublicAPIClient, *grpc.ClientConn) {
	conn, err := grpc.Dial(address, grpc.WithInsecure())
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	c := v1alpha.NewPublicAPIClient(conn)
	return c, conn
}

func runServer(t *testing.T, setup *taas.ServerSetup) *taas.Server {
	server := taas.NewServer(setup)
	go serverHandler(t, server)
	return server
}

func serverHandler(t *testing.T, server *taas.Server) {
	for {
		select {
		case msg, ok := <-server.Msg:
			if ok {
				t.Logf("server: %v", msg)
			} else {
				return
			}
		}
	}
}

func runSignImage(t *testing.T, imageFile string, keyIndex int) string {
	ascFile := fmt.Sprintf("%s.asc", imageFile)

	// keys stored in tests/secring.gpg, tests/pubring.gpg, tests/key1.gpg, tests/key2.gpg
	keyFingerprint := ""
	switch keyIndex {
	case 1:
		keyFingerprint = "D9DCEF41"
	case 2:
		keyFingerprint = "585091E3"
	default:
		panic("unknown key")
	}

	cmd := fmt.Sprintf("gpg --no-default-keyring --secret-keyring ./secring.gpg --keyring ./pubring.gpg --default-key %s --output %s --detach-sig %s",
		keyFingerprint, ascFile, imageFile)
	spawnAndWaitOrFail(t, cmd, 0)
	return ascFile
}

func runRktTrust(t *testing.T, ctx *testutils.RktRunCtx, prefix string, keyIndex int) {
	var cmd string
	keyFile := fmt.Sprintf("key%d.gpg", keyIndex)
	if prefix == "" {
		cmd = fmt.Sprintf(`%s trust --root %s`, ctx.Cmd(), keyFile)
	} else {
		cmd = fmt.Sprintf(`%s trust --prefix %s %s`, ctx.Cmd(), prefix, keyFile)
	}

	child := spawnOrFail(t, cmd)
	defer waitOrFail(t, child, 0)

	expected := "Are you sure you want to trust this key"
	if err := expectWithOutput(child, expected); err != nil {
		t.Fatalf("Expected but didn't find %q in %v", expected, err)
	}

	if err := child.SendLine("yes"); err != nil {
		t.Fatalf("Cannot confirm rkt trust: %s", err)
	}

	if prefix == "" {
		expected = "Added root key at"
	} else {
		expected = fmt.Sprintf(`Added key for prefix "%s" at`, prefix)
	}
	if err := expectWithOutput(child, expected); err != nil {
		t.Fatalf("Expected but didn't find %q in %v", expected, err)
	}
}

func checkUserNS() error {
	// CentOS 7 pretends to support user namespaces, but does not.
	// See https://bugzilla.redhat.com/show_bug.cgi?id=1168776#c5
	// Check if it really works
	return exec.Command("/bin/bash", "-c", "unshare -U true").Run()
}
