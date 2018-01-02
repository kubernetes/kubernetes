// Copyright 2015 The appc Authors
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

/*

This validator tool is intended to be run within an App Container Executor
(ACE), and verifies that the ACE has been set up correctly.

This verifies the _apps perspective_ of the execution environment.

Changes to the validator need to be reflected in app_manifest.json, and vice-versa

The App Container Execution spec defines the following expectations within the execution environment:
 - Working Directory defaults to the root of the application image, overridden with "workingDirectory"
 - PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
 - AC_APP_NAME the entrypoint that this process was defined from

In addition, we validate:
 - The expected mount points are mounted
 - metadata service reachable at http://169.254.169.255

TODO(jonboulle):
 - should we validate Isolators? (e.g. MemoryLimit + malloc, or capabilities)
 - should we validate ports? (e.g. that they are available to bind to within the network namespace of the pod)

*/

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

const (
	standardPath     = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
	appNameEnv       = "AC_APP_NAME"
	metadataPathBase = "/acMetadata/v1"

	// marker files to validate
	prestartFile = "/prestart"
	mainFile     = "/main"
	poststopFile = "/poststop"

	mainVolFile     = "/db/main"
	sidekickVolFile = "/db/sidekick"

	timeout = 5 * time.Second
)

var (
	// Expected values must be kept in sync with app_manifest.json
	workingDirectory = "/opt/acvalidator"
	// "Environment"
	env = map[string]string{
		"IN_ACE_VALIDATOR": "correct",
	}
	// "MountPoints"
	mps = map[string]types.MountPoint{
		"database": types.MountPoint{
			Path:     "/db",
			ReadOnly: false,
		},
	}
	// "Name"
	an = "ace-validator-main"
)

type results []error

// main outputs diagnostic information to stderr and exits 1 if validation fails
func main() {
	if len(os.Args) != 2 {
		stderr("usage: %s [main|sidekick|preStart|postStop]", os.Args[0])
		os.Exit(64)
	}
	mode := os.Args[1]
	var res results
	switch strings.ToLower(mode) {
	case "main":
		res = validateMain()
	case "sidekick":
		res = validateSidekick()
	case "prestart":
		res = validatePrestart()
	case "poststop":
		res = validatePoststop()
	default:
		stderr("unrecognized mode: %s", mode)
		os.Exit(64)
	}
	if len(res) == 0 {
		fmt.Printf("%s OK\n", mode)
		os.Exit(0)
	}
	fmt.Printf("%s FAIL\n", mode)
	for _, err := range res {
		fmt.Fprintln(os.Stderr, "==>", err)
	}
	os.Exit(1)
}

func validateMain() (errs results) {
	errs = append(errs, assertExists(prestartFile)...)
	errs = append(errs, assertNotExistsAndCreate(mainFile)...)
	errs = append(errs, assertNotExists(poststopFile)...)
	errs = append(errs, ValidatePath(standardPath)...)
	errs = append(errs, ValidateWorkingDirectory(workingDirectory)...)
	errs = append(errs, ValidateEnvironment(env)...)
	errs = append(errs, ValidateMountpoints(mps)...)
	errs = append(errs, ValidateAppNameEnv(an)...)
	errs = append(errs, ValidateMetadataSvc()...)
	errs = append(errs, waitForFile(sidekickVolFile, timeout)...)
	errs = append(errs, assertNotExistsAndCreate(mainVolFile)...)
	return
}

func validateSidekick() (errs results) {
	errs = append(errs, assertNotExistsAndCreate(sidekickVolFile)...)
	errs = append(errs, waitForFile(mainVolFile, timeout)...)
	return
}

func validatePrestart() (errs results) {
	errs = append(errs, assertNotExistsAndCreate(prestartFile)...)
	errs = append(errs, assertNotExists(mainFile)...)
	errs = append(errs, assertNotExists(poststopFile)...)
	return
}

func validatePoststop() (errs results) {
	errs = append(errs, assertExists(prestartFile)...)
	errs = append(errs, assertExists(mainFile)...)
	errs = append(errs, assertNotExistsAndCreate(poststopFile)...)
	return
}

// ValidatePath ensures that the PATH has been set up correctly within the
// environment in which this process is being run
func ValidatePath(wp string) results {
	r := results{}
	gp := os.Getenv("PATH")
	if wp != gp {
		r = append(r, fmt.Errorf("PATH not set appropriately (need %q, got %q)", wp, gp))
	}
	return r
}

// ValidateWorkingDirectory ensures that the process working directory is set
// to the desired path.
func ValidateWorkingDirectory(wwd string) (r results) {
	gwd, err := os.Getwd()
	if err != nil {
		r = append(r, fmt.Errorf("error getting working directory: %v", err))
		return
	}
	if gwd != wwd {
		r = append(r, fmt.Errorf("working directory not set appropriately (need %q, got %v)", wwd, gwd))
	}
	return
}

// ValidateEnvironment ensures that the given environment contains the
// necessary/expected environment variables.
func ValidateEnvironment(wenv map[string]string) (r results) {
	for wkey, wval := range wenv {
		gval := os.Getenv(wkey)
		if gval != wval {
			err := fmt.Errorf("environment variable %q not set appropriately (need %q, got %q)", wkey, wval, gval)
			r = append(r, err)
		}
	}
	return
}

// ValidateAppNameEnv ensures that the environment variable specifying the
// entrypoint of this process is set correctly.
func ValidateAppNameEnv(want string) (r results) {
	if got := os.Getenv(appNameEnv); got != want {
		r = append(r, fmt.Errorf("%s not set appropriately (need %q, got %q)", appNameEnv, want, got))
	}
	return
}

// ValidateMountpoints ensures that the given mount points are present in the
// environment in which this process is running
func ValidateMountpoints(wmp map[string]types.MountPoint) results {
	r := results{}
	// TODO(jonboulle): verify actual source
	for _, mp := range wmp {
		if err := checkMount(mp.Path, mp.ReadOnly); err != nil {
			r = append(r, err)
		}
	}
	return r
}

func metadataRequest(req *http.Request, expectedContentType string) ([]byte, error) {
	cli := http.Client{
		Timeout: 100 * time.Millisecond,
	}

	req.Header["Metadata-Flavor"] = []string{"AppContainer"}

	resp, err := cli.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Get %s failed with %v", req.URL, resp.StatusCode)
	}

	if respContentType := resp.Header.Get("Content-Type"); respContentType != expectedContentType {
		return nil, fmt.Errorf("`%v` did not respond with expected Content-Type header.  Expected %s, received %s",
			req.URL, expectedContentType, respContentType)
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("Get %s failed on body read: %v", req.URL, err)
	}

	return body, nil
}

func metadataGet(metadataURL, path string, expectedContentType string) ([]byte, error) {
	uri := metadataURL + metadataPathBase + path
	req, err := http.NewRequest("GET", uri, nil)
	if err != nil {
		panic(err)
	}

	return metadataRequest(req, expectedContentType)
}

func metadataPost(metadataURL, path string, body []byte, expectedContentType string) ([]byte, error) {
	uri := metadataURL + metadataPathBase + path
	req, err := http.NewRequest("POST", uri, bytes.NewBuffer(body))
	if err != nil {
		panic(err)
	}
	req.Header.Set("Content-Type", "text/plain")

	return metadataRequest(req, expectedContentType)
}

func metadataPostForm(metadataURL, path string, data url.Values, expectedContentType string) ([]byte, error) {
	uri := metadataURL + metadataPathBase + path
	req, err := http.NewRequest("POST", uri, strings.NewReader(data.Encode()))
	if err != nil {
		panic(err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	return metadataRequest(req, expectedContentType)
}

func validatePodAnnotations(metadataURL string, pm *schema.PodManifest) results {
	r := results{}

	var actualAnnots types.Annotations

	annotJson, err := metadataGet(metadataURL, "/pod/annotations", "application/json")
	if err != nil {
		return append(r, err)
	}

	err = json.Unmarshal(annotJson, &actualAnnots)
	if err != nil {
		return append(r, err)
	}

	if !reflect.DeepEqual(actualAnnots, pm.Annotations) {
		r = append(r, fmt.Errorf("pod annotations mismatch: %v vs %v", actualAnnots, pm.Annotations))
	}

	return r
}

func validatePodMetadata(metadataURL string, pm *schema.PodManifest) results {
	r := results{}

	uuid, err := metadataGet(metadataURL, "/pod/uuid", "text/plain; charset=us-ascii")
	if err != nil {
		return append(r, err)
	}

	_, err = types.NewUUID(string(uuid))
	if err != nil {
		return append(r, fmt.Errorf("malformed UUID returned (%v): %v", string(uuid), err))
	}

	return append(r, validatePodAnnotations(metadataURL, pm)...)
}

func validateAppAnnotations(metadataURL string, pm *schema.PodManifest, app *schema.RuntimeApp, img *schema.ImageManifest) results {
	r := results{}

	// build a map of expected annotations by merging img.Annotations
	// with PodManifest overrides
	expectedAnnots := img.Annotations
	for _, annot := range app.Annotations {
		expectedAnnots.Set(annot.Name, annot.Value)
	}
	if len(expectedAnnots) == 0 {
		expectedAnnots = nil
	}

	var actualAnnots types.Annotations

	annotJson, err := metadataGet(metadataURL, "/apps/"+string(app.Name)+"/annotations", "application/json")
	if err != nil {
		return append(r, err)
	}

	err = json.Unmarshal(annotJson, &actualAnnots)
	if err != nil {
		return append(r, err)
	}
	if len(actualAnnots) == 0 {
		actualAnnots = nil
	}

	if !reflect.DeepEqual(actualAnnots, expectedAnnots) {
		err := fmt.Errorf("%v annotations mismatch: %v vs %v", app.Name, actualAnnots, expectedAnnots)
		r = append(r, err)
	}

	return r
}

func validateAppMetadata(metadataURL string, pm *schema.PodManifest, app *schema.RuntimeApp) results {
	r := results{}

	am, err := metadataGet(metadataURL, "/apps/"+app.Name.String()+"/image/manifest", "application/json")
	if err != nil {
		return append(r, err)
	}

	img := &schema.ImageManifest{}
	if err = json.Unmarshal(am, img); err != nil {
		return append(r, fmt.Errorf("failed to JSON-decode %q manifest: %v", app.Name.String(), err))
	}

	id, err := metadataGet(metadataURL, "/apps/"+app.Name.String()+"/image/id", "text/plain; charset=us-ascii")
	if err != nil {
		r = append(r, err)
	}

	if string(id) != app.Image.ID.String() {
		err = fmt.Errorf("%q's image id mismatch: %v vs %v", app.Name.String(), id, app.Image.ID)
		r = append(r, err)
	}

	return append(r, validateAppAnnotations(metadataURL, pm, app, img)...)
}

func validateSigning(metadataURL string, pm *schema.PodManifest) results {
	r := results{}

	// Get our UUID
	uuid, err := metadataGet(metadataURL, "/pod/uuid", "text/plain; charset=us-ascii")
	if err != nil {
		return append(r, err)
	}

	plaintext := "Old MacDonald Had A Farm"

	// Sign
	sig, err := metadataPostForm(metadataURL, "/pod/hmac/sign", url.Values{
		"content": []string{plaintext},
	}, "text/plain; charset=us-ascii")
	if err != nil {
		return append(r, err)
	}

	// Verify
	_, err = metadataPostForm(metadataURL, "/pod/hmac/verify", url.Values{
		"content":   []string{plaintext},
		"uuid":      []string{string(uuid)},
		"signature": []string{string(sig)},
	}, "text/plain; charset=us-ascii")

	if err != nil {
		return append(r, err)
	}

	return r
}

func ValidateMetadataSvc() results {
	r := results{}

	metadataURL := os.Getenv("AC_METADATA_URL")
	if metadataURL == "" {
		return append(r, fmt.Errorf("AC_METADATA_URL is not set"))
	}

	pod, err := metadataGet(metadataURL, "/pod/manifest", "application/json")
	if err != nil {
		return append(r, err)
	}

	pm := &schema.PodManifest{}
	if err = json.Unmarshal(pod, pm); err != nil {
		return append(r, fmt.Errorf("failed to JSON-decode pod manifest: %v", err))
	}

	r = append(r, validatePodMetadata(metadataURL, pm)...)

	for _, app := range pm.Apps {
		app := app
		r = append(r, validateAppMetadata(metadataURL, pm, &app)...)
	}

	return append(r, validateSigning(metadataURL, pm)...)
}

// checkMount checks that the given string is a mount point, and that it is
// mounted appropriately read-only or not according to the given bool
func checkMount(d string, readonly bool) error {
	return checkMountImpl(d, readonly)
}

// parseMountinfo parses a Reader representing a /proc/PID/mountinfo file and
// returns whether dir is mounted and if so, whether it is read-only or not
func parseMountinfo(mountinfo io.Reader, dir string) (isMounted bool, readOnly bool, err error) {
	sc := bufio.NewScanner(mountinfo)
	for sc.Scan() {
		var (
			mountID      int
			parentID     int
			majorMinor   string
			root         string
			mountPoint   string
			mountOptions string
		)

		_, err := fmt.Sscanf(sc.Text(), "%d %d %s %s %s %s",
			&mountID, &parentID, &majorMinor, &root, &mountPoint, &mountOptions)
		if err != nil {
			return false, false, err
		}

		if mountPoint == dir {
			isMounted = true
			optionsParts := strings.Split(mountOptions, ",")
			for _, o := range optionsParts {
				switch o {
				case "ro":
					readOnly = true
				case "rw":
					readOnly = false
				}
			}
		}
	}

	return
}

// assertNotExistsAndCreate asserts that a file at the given path does not
// exist, and then proceeds to create (touch) the file. It returns any errors
// encountered at either of these steps.
func assertNotExistsAndCreate(p string) []error {
	var errs []error
	errs = append(errs, assertNotExists(p)...)
	if err := touchFile(p); err != nil {
		errs = append(errs, fmt.Errorf("error touching file %q: %v", p, err))
	}
	return errs
}

// assertNotExists asserts that a file at the given path does not exist. A
// non-empty list of errors is returned if the file exists or any error is
// encountered while checking.
func assertNotExists(p string) []error {
	var errs []error
	e, err := fileExists(p)
	if err != nil {
		errs = append(errs, fmt.Errorf("error checking %q exists: %v", p, err))
	}
	if e {
		errs = append(errs, fmt.Errorf("file %q exists unexpectedly", p))
	}
	return errs
}

// assertExists asserts that a file exists at the given path. A non-empty
// list of errors is returned if the file does not exist or any error is
// encountered while checking.
func assertExists(p string) []error {
	var errs []error
	e, err := fileExists(p)
	if err != nil {
		errs = append(errs, fmt.Errorf("error checking %q exists: %v", p, err))
	}
	if !e {
		errs = append(errs, fmt.Errorf("file %q does not exist as expected", p))
	}
	return errs
}

// touchFile creates an empty file, returning any error encountered
func touchFile(p string) error {
	_, err := os.Create(p)
	return err
}

// fileExists checks whether a file exists at the given path
func fileExists(p string) (bool, error) {
	_, err := os.Stat(p)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

// waitForFile waits for the file at the given path to appear
func waitForFile(p string, to time.Duration) []error {
	done := time.After(to)
	for {
		select {
		case <-done:
			return []error{
				fmt.Errorf("timed out waiting for %s", p),
			}
		case <-time.After(1):
			if ok, _ := fileExists(p); ok {
				return nil
			}
		}
	}
}

func stderr(format string, a ...interface{}) {
	out := fmt.Sprintf(format, a...)
	fmt.Fprintln(os.Stderr, strings.TrimSuffix(out, "\n"))
}
