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

	"golang.org/x/crypto/openpgp"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/api/v1alpha"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/tests/testutils"
	taas "github.com/coreos/rkt/tests/testutils/aci-server"
	shellquote "github.com/kballard/go-shellquote"
	"google.golang.org/grpc"
)

const (
	defaultTimeLayout = "2006-01-02 15:04:05.999 -0700 MST"

	baseAppName = "rkt-inspect"
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
	t.Logf("Spawning command: %v\n", cmd)
	child, err := gexpect.Spawn(cmd)
	if err != nil {
		t.Fatalf("Cannot exec rkt: %v", err)
	}
	return child
}

// waitOrFail waits for the child to exit, draining all its output.
// If a non-negative return value is provided, child exit status must match.
func waitOrFail(t *testing.T, child *gexpect.ExpectSubprocess, expectedStatus int) {
	bufOut := []string{}
	// TODO(lucab): gexpect should accept those channels from the caller
	ttyIn, ttyOut := child.AsyncInteractChannels()
	close(ttyIn)
	// drain output till gexpect closes the channel (on EOF or error)
	for line := range ttyOut {
		bufOut = append(bufOut, line)
	}
	err := child.Wait()
	status, _ := common.GetExitStatus(err)
	if expectedStatus >= 0 && status != expectedStatus {
		t.Fatalf("rkt terminated with unexpected status %d, expected %d\nOutput:\n%s", status, expectedStatus, bufOut)
	}
}

// waitPodReady waits for the pod supervisor to get ready, busy-looping until `timeout`
// while waiting for it. It returns the pod UUID or an error on failure.
func waitPodReady(ctx *testutils.RktRunCtx, t *testing.T, uuidFile string, timeout time.Duration) (string, error) {
	var podUUID []byte
	var err error
	interval := 500 * time.Millisecond
	elapsed := time.Duration(0)

	for elapsed < timeout {
		time.Sleep(interval)
		elapsed += interval
		podUUID, err = ioutil.ReadFile(uuidFile)
		if err == nil {
			break
		}
	}
	if err != nil {
		return "", fmt.Errorf("Can't read pod UUID: %v", err)
	}

	// wait up to one minute for the pod supervisor to be ready
	cmd := strings.Fields(fmt.Sprintf("%s status --wait-ready=%s %s", ctx.Cmd(), timeout, podUUID))
	statusCmd := exec.Command(cmd[0], cmd[1:]...)
	t.Logf("Running command: %v\n", cmd)
	output, err := statusCmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("Failed to wait for pod readiness, error %v output %v", err, string(output))
	}

	return string(podUUID), nil
}

// waitAppAttachable waits for an attachable application to get ready, busy-looping until `timeout`
// while waiting for it. It returns an error on failure.
func waitAppAttachable(ctx *testutils.RktRunCtx, t *testing.T, podUUID, appName string, timeout time.Duration) error {
	var (
		err         error
		output      []byte
		appNameFlag string
	)

	if appName != "" {
		appNameFlag = "--app=" + appName
	}
	cmd := strings.Fields(fmt.Sprintf("%s attach --mode=list %s %s", ctx.Cmd(), appNameFlag, podUUID))

	interval := 500 * time.Millisecond
	elapsed := time.Duration(0)
	for elapsed < timeout {
		time.Sleep(interval)
		elapsed += interval
		statusCmd := exec.Command(cmd[0], cmd[1:]...)
		output, err = statusCmd.CombinedOutput()
		if err == nil {
			break
		}
	}
	if err != nil {
		return fmt.Errorf("%s", output)
	}
	return nil
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

func mustTempDir(dirName string) string {
	tmpDir, err := ioutil.TempDir("", dirName)
	if err != nil {
		panic(fmt.Sprintf("Cannot create temp dir: %v", err))
	}
	return tmpDir
}

// createFileOrPanic creates an empty file within the given directory
// with a specified name. Panics if file creation fails for any reason.
func createFileOrPanic(dirName, fileName string) string {
	name := filepath.Join(dirName, fileName)
	file, err := os.Create(name)
	if err != nil {
		panic(err)
	}

	defer file.Close()
	return name
}

func importImageAndFetchHashAsUidGid(t *testing.T, ctx *testutils.RktRunCtx, img string, fetchArgs string, uid int, gid int) (string, error) {
	// Import the test image into store manually.
	cmd := fmt.Sprintf("%s --insecure-options=image,tls fetch --pull-policy=new %s %s", ctx.Cmd(), fetchArgs, img)

	// TODO(jonboulle): non-root user breaks trying to read root-written
	// config directories. Should be a better way to approach this. Should
	// config directories be readable by the rkt group too?
	if gid != 0 {
		cmd = fmt.Sprintf("%s --insecure-options=image,tls fetch --pull-policy=new %s %s", ctx.CmdNoConfig(), fetchArgs, img)
	}
	child, err := gexpect.Command(cmd)
	if err != nil {
		t.Fatalf("cannot create rkt command: %v", err)
	}
	if gid != 0 {
		child.Cmd.SysProcAttr = &syscall.SysProcAttr{}
		child.Cmd.SysProcAttr.Credential = &syscall.Credential{Uid: uint32(uid), Gid: uint32(gid)}
	}

	err = child.Start()
	if err != nil {
		t.Fatalf("cannot exec rkt: %v", err)
	}

	// Read out the image hash.
	result, out, err := expectRegexWithOutput(child, "sha512-[0-9a-f]{32,64}")
	if exitErr := checkExitStatus(child); exitErr != nil {
		t.Logf("%v", exitErr)
		return "", fmt.Errorf("fetching of %q failed", img)
	}
	if err != nil || len(result) != 1 {
		t.Fatalf("Error: %v\nOutput: %v", err, out)
	}

	return result[0], nil
}

func importImageAndFetchHash(t *testing.T, ctx *testutils.RktRunCtx, fetchArgs string, img string) (string, error) {
	return importImageAndFetchHashAsUidGid(t, ctx, fetchArgs, img, 0, 0)
}

func importImageAndRun(imagePath string, t *testing.T, ctx *testutils.RktRunCtx) {
	cmd := fmt.Sprintf("%s --insecure-options=image run %s", ctx.Cmd(), imagePath)
	spawnAndWaitOrFail(t, cmd, 0)
}

func importImageAndPrepare(imagePath string, t *testing.T, ctx *testutils.RktRunCtx) {
	cmd := fmt.Sprintf("%s --insecure-options=image prepare %s", ctx.Cmd(), imagePath)
	spawnAndWaitOrFail(t, cmd, 0)
}

func patchImportAndFetchHash(image string, patches []string, t *testing.T, ctx *testutils.RktRunCtx) (string, error) {
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

func patchImportAndPrepare(image string, patches []string, t *testing.T, ctx *testutils.RktRunCtx) {
	imagePath := patchTestACI(image, patches...)
	defer os.Remove(imagePath)

	cmd := fmt.Sprintf("%s --insecure-options=image prepare %s", ctx.Cmd(), imagePath)
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

	result, out, err := expectRegexWithOutput(child, "[0-9a-f-]{36}")
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
	nobodyUid, _ := testutils.GetUnprivilegedUidGid()
	runRktAsUidGidAndCheckOutput(t, rktCmd, expectedLine, false, expectError, nobodyUid, gid)
}

func runRktAsGidAndCheckREOutput(t *testing.T, rktCmd, expectedLine string, expectError bool, gid int) {
	nobodyUid, _ := testutils.GetUnprivilegedUidGid()
	runRktAsUidGidAndCheckOutput(t, rktCmd, expectedLine, true, expectError, nobodyUid, gid)
}

func runRktAsUidGidAndCheckOutput(t *testing.T, rktCmd, expectedLine string, lineIsRegex, expectError bool, uid, gid int) {
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
		expectedStatus = 254
	}
	defer waitOrFail(t, child, expectedStatus)

	if expectedLine != "" {
		if lineIsRegex == true {
			_, _, err := expectRegexWithOutput(child, expectedLine)
			if err != nil {
				t.Fatalf("didn't receive expected regex %q in output: %v", expectedLine, err)
			}
		} else {
			err = expectWithOutput(child, expectedLine)
			if err != nil {
				t.Fatalf("didn't receive expected output %q: %v", expectedLine, err)
			}
		}

	}
}

func runRkt(t *testing.T, rktCmd string, uid, gid int) (string, int) {
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

	_, linesChan := child.AsyncInteractChannels()

	var buf bytes.Buffer
	for line := range linesChan {
		buf.WriteString(line + "\n") // reappend newline
	}

	status, _ := common.GetExitStatus(child.Wait())
	return buf.String(), status
}

func startRktAsGidAndCheckOutput(t *testing.T, rktCmd, expectedLine string, gid int) *gexpect.ExpectSubprocess {
	child, err := gexpect.Command(rktCmd)
	if err != nil {
		t.Fatalf("cannot exec rkt: %v", err)
	}
	if gid != 0 {
		child.Cmd.SysProcAttr = &syscall.SysProcAttr{}
		nobodyUid, _ := testutils.GetUnprivilegedUidGid()
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

func startRktAsUidGidAndCheckOutput(t *testing.T, rktCmd, expectedLine string, expectError bool, uid, gid int) *gexpect.ExpectSubprocess {
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

	if expectedLine != "" {
		if err := expectWithOutput(child, expectedLine); err != nil {
			t.Fatalf("didn't receive expected output %q: %v", expectedLine, err)
		}
	}
	return child
}

func runRktAndCheckRegexOutput(t *testing.T, rktCmd, match string) error {
	re, err := regexp.Compile(match)
	if err != nil {
		t.Fatalf("error compiling regex %q: %v", match, err)
	}

	args, err := shellquote.Split(rktCmd)
	if err != nil {
		t.Fatalf("error splitting cmd %q: %v", rktCmd, err)
	}

	path, err := exec.LookPath(args[0])
	cmd := exec.Command(path, args[1:]...)

	out, err := cmd.CombinedOutput()

	result := re.MatchString(string(out))
	if !result {
		t.Fatalf("%q regex must be found\nOutput: %q", match, string(out))
	}

	return err
}

func runRktAndCheckOutput(t *testing.T, rktCmd, expectedLine string, expectError bool) {
	runRktAsGidAndCheckOutput(t, rktCmd, expectedLine, expectError, 0)
}

func runRktAndCheckREOutput(t *testing.T, rktCmd, expectedLine string, expectError bool) {
	runRktAsGidAndCheckREOutput(t, rktCmd, expectedLine, expectError, 0)
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
	id        string
	pid       int
	state     string
	apps      map[string]*appInfo
	networks  map[string]*networkInfo
	manifest  *schema.PodManifest
	createdAt int64
	startedAt int64
}

type imagePatch struct {
	name    string
	patches []string
}

// parsePodInfo parses the 'rkt status $UUID' result into podInfo struct.
// For example, the 'result' can be:
// state=running
// networks=default:ip4=172.16.28.103
// pid=14352
// exited=false
// created=2016-04-01 19:12:03.447 -0700 PDT
// started=2016-04-01 19:12:04.279 -0700 PDT
func parsePodInfoOutput(t *testing.T, result string, p *podInfo) {
	lines := strings.Split(strings.TrimSuffix(result, "\n"), "\n")
	for _, line := range lines {
		tuples := strings.SplitN(line, "=", 2)
		if len(tuples) != 2 {
			t.Logf("Unexpected line: %v", line)
			continue
		}

		switch tuples[0] {
		case "state":
			p.state = tuples[1]
		case "networks":
			if tuples[1] == "" {
				break
			}
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
		case "created":
			createdAt, err := time.Parse(defaultTimeLayout, tuples[1])
			if err != nil {
				t.Fatalf("Cannot parse the pod's creation time %q: %v", tuples[1], err)
			}
			p.createdAt = createdAt.UnixNano()
		case "started":
			startedAt, err := time.Parse(defaultTimeLayout, tuples[1])
			if err != nil {
				t.Fatalf("Cannot parse the pod's start time %q: %v", tuples[1], err)
			}
			p.startedAt = startedAt.UnixNano()
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
		pid:      -1,
		apps:     make(map[string]*appInfo),
		networks: make(map[string]*networkInfo),
	}

	// Read pod manifest.
	output, err := exec.Command("/bin/bash", "-c", fmt.Sprintf("%s cat-manifest %s", ctx.Cmd(), podID)).CombinedOutput()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Trim the last '\n' character.
	mfst := bytes.TrimSpace(output)

	// Fill app infos.
	if err := json.Unmarshal(mfst, &p.manifest); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	for _, app := range p.manifest.Apps {
		appName := app.Name.String()
		p.apps[appName] = &appInfo{
			name: appName,
			// TODO(yifan): Get the image's name.
			image: &imageInfo{id: app.Image.ID.String()},
		}
	}

	// Fill other infos.
	output, _ = exec.Command("/bin/bash", "-c", fmt.Sprintf("%s status %s", ctx.Cmd(), podID)).CombinedOutput()
	parsePodInfoOutput(t, string(output), p)

	return p
}

// parseImageInfoOutput parses the 'rkt image list' result into imageInfo struct.
// For example, the 'result' can be:
// 'sha512-e9b77714dbbfda12cb9e136318b103a6f0ce082004d09d0224a620d2bbf38133 nginx:latest 2015-10-16 17:42:57.741 -0700 PDT true'
func parseImageInfoOutput(t *testing.T, result string) *imageInfo {
	fields := regexp.MustCompile("\t+").Split(result, -1)
	nameVersion := strings.Split(fields[1], ":")
	if len(nameVersion) != 2 {
		t.Fatalf("Failed to parse name version string: %q", fields[1])
	}
	importTime, err := time.Parse(defaultTimeLayout, fields[3])
	if err != nil {
		t.Fatalf("Failed to parse time string: %q", fields[3])
	}
	size, err := strconv.Atoi(fields[2])
	if err != nil {
		t.Fatalf("Failed to parse image size string: %q", fields[2])
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
	output, err = exec.Command("/bin/bash", "-c",
		fmt.Sprintf("%s image cat-manifest --pretty-print=false %s", ctx.Cmd(), imageID)).CombinedOutput()
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

func runSignImage(t *testing.T, imagePath string, keyIndex int) string {
	// keys stored in tests/secring.gpg.
	keyFingerprint := ""
	switch keyIndex {
	case 1:
		keyFingerprint = "D9DCEF41"
	case 2:
		keyFingerprint = "585091E3"
	default:
		panic("unknown key")
	}

	secringFile, err := os.Open("./secring.gpg")
	if err != nil {
		t.Fatalf("Cannot open secring.gpg file: %v", err)
	}
	defer secringFile.Close()

	entityList, err := openpgp.ReadKeyRing(secringFile)
	if err != nil {
		t.Fatalf("Failed to read secring.gpg file: %v", err)
	}

	var signingEntity *openpgp.Entity
	for _, entity := range entityList {
		if entity.PrivateKey.KeyIdShortString() == keyFingerprint {
			signingEntity = entity
		}
	}

	imageFile, err := os.Open(imagePath)
	if err != nil {
		t.Fatalf("Cannot open image file %s: %v", imagePath, err)
	}
	defer imageFile.Close()

	ascPath := fmt.Sprintf("%s.asc", imagePath)
	ascFile, err := os.Create(ascPath)
	if err != nil {
		t.Fatalf("Cannot create asc file %s: %v", ascPath, err)
	}
	defer ascFile.Close()

	err = openpgp.ArmoredDetachSign(ascFile, signingEntity, imageFile, nil)
	if err != nil {
		t.Fatalf("Cannot create armored detached signature: %v", err)
	}

	return ascPath
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

func generatePodManifestFile(t *testing.T, manifest *schema.PodManifest) string {
	tmpDir := testutils.GetValueFromEnvOrPanic("FUNCTIONAL_TMP")
	f, err := ioutil.TempFile(tmpDir, "rkt-test-manifest-")
	if err != nil {
		t.Fatalf("Cannot create tmp pod manifest: %v", err)
	}

	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("Cannot marshal pod manifest: %v", err)
	}
	if err := ioutil.WriteFile(f.Name(), data, 0600); err != nil {
		t.Fatalf("Cannot write pod manifest file: %v", err)
	}
	return f.Name()
}

func checkUserNS() error {
	// CentOS 7 pretends to support user namespaces, but does not.
	// See https://bugzilla.redhat.com/show_bug.cgi?id=1168776#c5
	// Check if it really works
	return exec.Command("/bin/bash", "-c", "unshare -U true").Run()
}

func authDir(confDir string) string {
	return filepath.Join(confDir, "auth.d")
}

func pathsDir(confDir string) string {
	return filepath.Join(confDir, "paths.d")
}

func stage1Dir(confDir string) string {
	return filepath.Join(confDir, "stage1.d")
}

func writeConfig(t *testing.T, dir, filename, contents string) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatalf("Failed to create config directory %q: %v", dir, err)
	}
	path := filepath.Join(dir, filename)
	os.Remove(path)
	if err := ioutil.WriteFile(path, []byte(contents), 0644); err != nil {
		t.Fatalf("Failed to write file %q: %v", path, err)
	}
}

func verifyHostFile(t *testing.T, tmpdir, filename string, i int, expectedResult string) {
	filePath := path.Join(tmpdir, filename)
	defer os.Remove(filePath)

	// Verify the file is written to host.
	if strings.Contains(expectedResult, "host:") {
		data, err := ioutil.ReadFile(filePath)
		if err != nil {
			t.Fatalf("%d: Cannot read the host file: %v", i, err)
		}
		if string(data) != expectedResult {
			t.Fatalf("%d: Expecting %q in the host file, but saw %q", i, expectedResult, data)
		}
	}
}

func executeFuncsReverse(funcs []func()) {
	n := len(funcs)
	for i := n - 1; i >= 0; i-- {
		funcs[i]()
	}
}

func unmountPod(t *testing.T, ctx *testutils.RktRunCtx, uuid string, rmNetns bool) {
	podDir := filepath.Join(ctx.DataDir(), "pods", "run", uuid)
	stage1MntPath := filepath.Join(podDir, "stage1", "rootfs")
	stage2MntPath := filepath.Join(stage1MntPath, "opt", "stage2", "rkt-inspect", "rootfs")

	netnsPath := filepath.Join(podDir, "netns")
	podNetNSPathBytes, err := ioutil.ReadFile(netnsPath)
	// There may be no netns, e.g. kvm or --net=host
	if err != nil {
		if !os.IsNotExist(err) {
			t.Fatalf(`cannot read "netns" stage1: %v`, err)
		} else {
			rmNetns = false
		}
	}

	if err := syscall.Unmount(stage2MntPath, 0); err != nil {
		t.Fatalf("cannot umount stage2: %v", err)
	}

	if err := syscall.Unmount(stage1MntPath, 0); err != nil {
		t.Fatalf("cannot umount stage1: %v", err)
	}

	if rmNetns {
		podNetNSPath := string(podNetNSPathBytes)

		if err := syscall.Unmount(podNetNSPath, 0); err != nil {
			t.Fatalf("cannot umount pod netns: %v", err)
		}

		_ = os.RemoveAll(podNetNSPath)
	}
}

func checkExitStatus(child *gexpect.ExpectSubprocess) error {
	err := child.Wait()
	status, _ := common.GetExitStatus(err)
	if status != 0 {
		return fmt.Errorf("rkt terminated with unexpected status %d, expected %d\nOutput:\n%s", status, 0, child.Collect())
	}

	return nil
}

// combinedOutput executes the given command c for the given test context t
// and fails test t if command execution failed.
// It returns the command output.
func combinedOutput(t *testing.T, c *exec.Cmd) string {
	t.Log("Running", c.Args)
	out, err := c.CombinedOutput()

	if err != nil {
		t.Fatal(err, "output", string(out))
	}

	return string(out)
}

// retry is the struct that represents retrying function calls.
type retry struct {
	n int
	t time.Duration
}

// Retry retries the given function f n times with a delay t between invocations
// until no error is returned from f or n is exceeded.
// The last occured error is returned.
func (r retry) Retry(f func() error) error {
	var err error

	for i := 0; i < r.n; i++ {
		err = f()
		if err == nil {
			return nil
		}
		time.Sleep(r.t)
	}

	return err
}
