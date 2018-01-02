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
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/rkt/config"
	"github.com/coreos/rkt/tests/testutils"
	taas "github.com/coreos/rkt/tests/testutils/aci-server"
)

type stubStage1Reference int

type stubStage1Setup struct {
	t       *testing.T
	ctx     *testutils.RktRunCtx
	server  *taas.Server
	name    string
	version string
	path    string
}

const (
	stubStage1UrlHttp stubStage1Reference = iota
	stubStage1UrlHttps
	stubStage1UrlDocker
	stubStage1UrlFile
	stubStage1PathRel
	stubStage1PathAbs
	stubStage1Name
	stubStage1Base
)

const (
	stubStage1Output string = "success, stub stage1 would at this point switch to stage2"
)

// stuff to test: DONE/POSTPONED
// - loading stage1 from config DONE/POSTPONED
//   - url location DONE/POSTPONED
//     - http DONE
//       - check if it gets image from store in the second run DONE
//     - https DONE
//       - check if it gets image from store in the second run DONE
//     - docker POSTPONED
//       - check if it gets image from store in the second run
//       - will not test it until we get some local test docker server
//     - file DONE
//   - path location DONE
//     - bail out on relative path DONE
//     - accept absolute path DONE
//     - check if it falls back to the image in the same directory DONE
//   - name and version from store DONE
//     - make sure that it does not do any discovery and fetch the stuff from remote DONE
// - loading stage1 from flags DONE/POSTPONED
//   - --stage1-path DONE
//     - absolute paths DONE
//     - relative paths DONE
//   - --stage1-url
//     - http DONE
//       - check if it gets image from store in the second run DONE
//     - https DONE
//       - check if it gets image from store in the second run DONE
//     - docker POSTPONED
//       - check if it gets image from store in the second run
//       - will not test it until we get some local test docker server
//     - file DONE
//   - --stage1-name DONE
//       - check if it gets image from store in the second run DONE
//   - --stage1-hash DONE
//   - --stage1-from-dir DONE
//     - needs overriding the directory via config DONE

func TestStage1LoadingFromConfig(t *testing.T) {
	// We don't need discovery in this test, so get random port,
	// but keep https.
	serverSetup := taas.GetDefaultServerSetup()
	serverSetup.Port = taas.PortRandom
	setup := newStubStage1Setup(t, serverSetup)
	defer setup.cleanup()

	tests := []struct {
		name              string
		version           string
		location          string
		expectedFirstRun  string
		expectedSecondRun string
		skipReason        string
	}{
		// typical scenario - config file with correct name,
		// version and location, we expect the stage1 image to
		// be taken from file in the first run, and then from
		// store on subsequent runs
		{
			name:              setup.name,
			version:           setup.version,
			location:          setup.getLocation(stubStage1PathAbs),
			expectedFirstRun:  fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
			expectedSecondRun: fmt.Sprintf("image: using image from local store for image name %s:%s", setup.name, setup.version),
		},

		// wrong name, resulting in always loading the stage1
		// image from disk
		{
			name:              "example.com/stage1",
			version:           setup.version,
			location:          setup.getLocation(stubStage1PathAbs),
			expectedFirstRun:  fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
			expectedSecondRun: fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
		},

		// wrong version, resulting in always loading the stage1
		// image from disk
		{
			name:              setup.name,
			version:           "3.2.1",
			location:          setup.getLocation(stubStage1PathAbs),
			expectedFirstRun:  fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
			expectedSecondRun: fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
		},

		// loading stage1 from http URL
		{
			name:       setup.name,
			version:    setup.version,
			location:   setup.getLocation(stubStage1UrlHttp),
			skipReason: "tested in TestStage1LoadingFromConfigHttp",
		},

		// loading stage1 from docker URL
		{
			name:       setup.name,
			version:    setup.version,
			location:   setup.getLocation(stubStage1UrlDocker),
			skipReason: "tested in TestStage1LoadingFromConfigDocker",
		},

		// loading stage1 from https URL
		{
			name:              setup.name,
			version:           setup.version,
			location:          setup.getLocation(stubStage1UrlHttps),
			expectedFirstRun:  fmt.Sprintf("image: remote fetching from URL %q", setup.getLocation(stubStage1UrlHttps)),
			expectedSecondRun: fmt.Sprintf("image: using image from local store for image name %s:%s", setup.name, setup.version),
		},

		// loading stage1 from file URL
		{
			name:              setup.name,
			version:           setup.version,
			location:          setup.getLocation(stubStage1UrlFile),
			expectedFirstRun:  fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
			expectedSecondRun: fmt.Sprintf("image: using image from local store for image name %s:%s", setup.name, setup.version),
		},
	}

	for _, tt := range tests {
		if tt.skipReason != "" {
			t.Logf("skipping the testcase with name %q, version %q and location %q, reason - %s", tt.name, tt.version, tt.location, tt.skipReason)
			continue
		}
		cfg := &config.Stage1Data{
			Name:     tt.name,
			Version:  tt.version,
			Location: tt.location,
		}
		t.Logf("Generating stage1 configuration file with name %q, version %q and location %q", tt.name, tt.version, tt.location)
		setup.generateStage1Config(cfg)
		setup.check("", tt.expectedFirstRun)
		setup.check("", tt.expectedSecondRun)
		setup.ctx.Reset()
	}
}

func TestStage1LoadingFromConfigHttp(t *testing.T) {
	// We don't need discovery in this test, so get random port.
	serverSetup := taas.GetDefaultServerSetup()
	serverSetup.Protocol = taas.ProtocolHttp
	serverSetup.Port = taas.PortRandom
	setup := newStubStage1Setup(t, serverSetup)
	defer setup.cleanup()
	expectedFirstRun := fmt.Sprintf("image: remote fetching from URL %q", setup.getLocation(stubStage1UrlHttp))
	expectedSecondRun := fmt.Sprintf("image: using image from local store for image name %s:%s", setup.name, setup.version)
	cfg := &config.Stage1Data{
		Name:     setup.name,
		Version:  setup.version,
		Location: setup.getLocation(stubStage1UrlHttp),
	}

	t.Logf("Generating stage1 configuration file with name %q, version %q and location %q", setup.name, setup.version, setup.getLocation(stubStage1UrlHttp))
	setup.generateStage1Config(cfg)
	setup.check("", expectedFirstRun)
	setup.check("", expectedSecondRun)
}

func TestStage1LoadingFromConfigDocker(t *testing.T) {
	t.Skip("no test docker server is available")
}

func TestStage1LoadingFromConfigRelativePathFail(t *testing.T) {
	setup := newStubStage1Setup(t, nil)
	defer setup.cleanup()

	cfg := &config.Stage1Data{
		Name:     setup.name,
		Version:  setup.version,
		Location: setup.getLocation(stubStage1PathRel),
	}
	setup.generateStage1Config(cfg)
	cmd := fmt.Sprintf("%s --insecure-options=image,tls --debug run %s", setup.ctx.Cmd(), getInspectImagePath())
	child := spawnOrFail(setup.t, cmd)
	defer waitOrFail(setup.t, child, 254)
	expectedLine := "default stage1 image location is either a relative path or a URL without scheme"
	setup.getExpectedOrFail(child, expectedLine)
}

func TestStage1LoadingFromConfigFallback(t *testing.T) {
	// This directory will be empty. We just use it to point rkt
	// to the stage1 image that is not there, so rkt will fallback
	// to stage1 image in the same directory as itself is in.
	tmp, err := ioutil.TempDir("", "rkt-config-fallback")
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmp)
	setup := newStubStage1Setup(t, nil)
	defer setup.cleanup()

	fakePath := filepath.Join(tmp, filepath.Base(setup.getLocation(stubStage1Base)))
	cfg := &config.Stage1Data{
		Name:     setup.name,
		Version:  setup.version,
		Location: fakePath,
	}
	setup.generateStage1Config(cfg)
	cmd := fmt.Sprintf("%s --insecure-options=image,tls --debug run %s", setup.ctx.Cmd(), getInspectImagePath())
	child := spawnOrFail(setup.t, cmd)
	defer waitOrFail(setup.t, child, 0)
	setup.getExpectedOrFail(child, fmt.Sprintf("image: using image from file %s", fakePath))
	setup.getExpectedOrFail(child, fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)))
	setup.getExpectedOrFail(child, stubStage1Output)
}

func TestStage1LoadingFromConfigNoDiscovery(t *testing.T) {
	// We shouldn't be using discovery in this setup, but enable
	// it, just in case.
	setup := newStubStage1Setup(t, taas.GetDefaultServerSetup())
	defer setup.cleanup()

	cfg := &config.Stage1Data{
		Name:     setup.name,
		Version:  setup.version,
		Location: setup.getLocation(stubStage1PathAbs),
	}
	setup.generateStage1Config(cfg)
	cmd := fmt.Sprintf("%s --insecure-options=image,tls --debug run %s", setup.ctx.Cmd(), getInspectImagePath())
	child := spawnOrFail(setup.t, cmd)
	defer waitOrFail(setup.t, child, 0)
	discoveringStr := fmt.Sprintf("searching for app image %s", setup.name)
	for {
		matches, output, err := expectRegexWithOutput(child, `(?m)^image:.+$`)
		if err != nil {
			if strings.Index(output, stubStage1Output) < 0 {
				t.Fatalf("got no success notice from stub stage1, output:\n%s", output)
			}
			break
		}
		if strings.Index(matches[0], discoveringStr) >= 0 {
			t.Fatalf("rkt performs discovery for stage1 image, but it should not")
		}
	}
}

func TestStage1LoadingFromFlags(t *testing.T) {
	// We use discovery in this test.
	setup := newStubStage1Setup(t, taas.GetDefaultServerSetup())
	defer setup.cleanup()

	tests := []struct {
		flag              string
		expectedFirstRun  string
		expectedSecondRun string
		skipReason        string
	}{
		// --stage1-path with a relative path
		{
			flag:              fmt.Sprintf("--stage1-path=%q", setup.getLocation(stubStage1PathRel)),
			expectedFirstRun:  fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
			expectedSecondRun: fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
		},

		// --stage1-path with an absolute path
		{
			flag:              fmt.Sprintf("--stage1-path=%q", setup.getLocation(stubStage1PathAbs)),
			expectedFirstRun:  fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
			expectedSecondRun: fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
		},

		// --stage1-url with an http URL
		{
			flag:       "--stage1-url=http://...",
			skipReason: "tested in TestStage1LoadingFromFlagsHttp",
		},

		// --stage1-url with an docker URL
		{
			flag:       "--stage1-url=docker://...",
			skipReason: "tested in TestStage1LoadingFromFlagsDocker",
		},

		// --stage1-url with an https URL
		{
			flag:              fmt.Sprintf("--stage1-url=%q", setup.getLocation(stubStage1UrlHttps)),
			expectedFirstRun:  fmt.Sprintf("image: remote fetching from URL %q", setup.getLocation(stubStage1UrlHttps)),
			expectedSecondRun: fmt.Sprintf("image: using image from local store for url %s", setup.getLocation(stubStage1UrlHttps)),
		},

		// --stage1-url with an file URL
		{
			flag:              fmt.Sprintf("--stage1-url=%q", setup.getLocation(stubStage1UrlFile)),
			expectedFirstRun:  fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
			expectedSecondRun: fmt.Sprintf("image: using image from file %s", setup.getLocation(stubStage1PathAbs)),
		},

		// --stage1-name
		{
			flag:              fmt.Sprintf("--stage1-name=%s", setup.getLocation(stubStage1Name)),
			expectedFirstRun:  fmt.Sprintf("image: searching for app image %s", setup.name),
			expectedSecondRun: fmt.Sprintf("image: using image from local store for image name %s", setup.getLocation(stubStage1Name)),
		},

		// --stage1-hash
		{
			flag:       "--stage1-hash=...",
			skipReason: "tested in TestStage1LoadingFromFlagsHash",
		},

		// --stage1-from-dir
		{
			flag:       "--stage1-from-dir=...",
			skipReason: "tested in TestStage1LoadingFromFlagsFromDir",
		},
	}

	for _, tt := range tests {
		if tt.skipReason != "" {
			t.Logf("skipping the testcase with the flag %s, reason - %s", tt.flag, tt.skipReason)
			continue
		}
		setup.check(tt.flag, tt.expectedFirstRun)
		setup.check(tt.flag, tt.expectedSecondRun)
		setup.ctx.Reset()
	}
}

func TestStage1LoadingFromFlagsHttp(t *testing.T) {
	// We need no discovery, so use random port.
	serverSetup := taas.GetDefaultServerSetup()
	serverSetup.Protocol = taas.ProtocolHttp
	serverSetup.Port = taas.PortRandom
	setup := newStubStage1Setup(t, serverSetup)
	defer setup.cleanup()

	flag := fmt.Sprintf("--stage1-url=%q", setup.getLocation(stubStage1UrlHttp))
	expectedFirstRun := fmt.Sprintf("image: remote fetching from URL %q", setup.getLocation(stubStage1UrlHttp))
	expectedSecondRun := fmt.Sprintf("image: using image from local store for url %s", setup.getLocation(stubStage1UrlHttp))

	setup.check(flag, expectedFirstRun)
	setup.check(flag, expectedSecondRun)
}

func TestStage1LoadingFromFlagsDocker(t *testing.T) {
	t.Skip("no test docker server is available")
}

func TestStage1LoadingFromFlagsHash(t *testing.T) {
	setup := newStubStage1Setup(t, nil)
	defer setup.cleanup()

	stubHash, err := importImageAndFetchHash(setup.t, setup.ctx, "", setup.getLocation(stubStage1PathAbs))
	if err != nil {
		t.Fatalf("%v", err)
	}
	cmd := fmt.Sprintf("%s --insecure-options=image,tls --debug run --stage1-hash=%s %s", setup.ctx.Cmd(), stubHash, getInspectImagePath())
	child := spawnOrFail(setup.t, cmd)
	defer waitOrFail(setup.t, child, 0)
	setup.getExpectedOrFail(child, fmt.Sprintf("using image from the store with hash %s", stubHash))
	setup.getExpectedOrFail(child, stubStage1Output)
}

func TestStage1LoadingFromFlagsFromDir(t *testing.T) {
	setup := newStubStage1Setup(t, nil)
	defer setup.cleanup()
	tmp, err := ioutil.TempDir("", "rkt-config-from-dir")
	if err != nil {
		t.Fatalf("Failed to create a temporary directory: %v", err)
	}
	defer os.RemoveAll(tmp)

	stubCopyPath := filepath.Join(tmp, setup.getLocation(stubStage1Base))
	stubCopy, err := os.Create(stubCopyPath)
	if err != nil {
		t.Fatalf("Failed to create a file in the temporary directory: %v", err)
	}
	defer stubCopy.Close()
	stubStage1, err := os.Open(setup.getLocation(stubStage1PathAbs))
	if err != nil {
		t.Fatalf("Failed to open the stub stage1 image: %v", err)
	}
	defer stubStage1.Close()
	if _, err := io.Copy(stubCopy, stubStage1); err != nil {
		t.Fatalf("Failed to copy the stub stage1 image to the temporary directory: %v", err)
	}

	setup.generateStage1ImagesDirectoryConfig(tmp)
	cmd := fmt.Sprintf("%s --insecure-options=image,tls --debug run --stage1-from-dir=%s %s", setup.ctx.Cmd(), setup.getLocation(stubStage1Base), getInspectImagePath())
	child := spawnOrFail(setup.t, cmd)
	defer waitOrFail(setup.t, child, 0)
	setup.getExpectedOrFail(child, fmt.Sprintf("image: using image from file %s", stubCopyPath))
	setup.getExpectedOrFail(child, stubStage1Output)
}

func newStubStage1Setup(t *testing.T, serverSetup *taas.ServerSetup) *stubStage1Setup {
	ctx := testutils.NewRktRunCtx()
	defer func() {
		if ctx != nil {
			ctx.Cleanup()
		}
	}()

	var server *taas.Server
	stubStage1 := testutils.GetValueFromEnvOrPanic("STUB_STAGE1")
	if serverSetup != nil {
		server = runServer(t, serverSetup)
		defer func() {
			if server != nil {
				server.Close()
			}
		}()
		fileSet := map[string]string{
			filepath.Base(stubStage1): stubStage1,
		}
		if err := server.UpdateFileSet(fileSet); err != nil {
			t.Fatalf("Failed to populate a file list in test aci server: %v", err)
		}
	}

	setup := &stubStage1Setup{
		t:       t,
		ctx:     ctx,
		server:  server,
		name:    "localhost/rkt-stub-stage1",
		version: "0.0.1",
		path:    stubStage1,
	}
	ctx = nil
	server = nil
	return setup
}

func (s *stubStage1Setup) cleanup() {
	s.ctx.Cleanup()
	if s.server != nil {
		s.server.Close()
	}
}

func (s *stubStage1Setup) getLocation(refType stubStage1Reference) string {
	switch refType {
	case stubStage1UrlHttp:
		return fmt.Sprintf("http://%s/%s", s.getServerHost(), filepath.Base(s.path))
	case stubStage1UrlHttps:
		return fmt.Sprintf("https://%s/%s", s.getServerHost(), filepath.Base(s.path))
	case stubStage1UrlDocker:
		// TODO: fix it with the correct URL when we have the
		// test docker server
		return fmt.Sprintf("docker://%s/%s", s.getServerHost(), filepath.Base(s.path))
	case stubStage1UrlFile:
		return fmt.Sprintf("file://%s", s.getAbsPath())
	case stubStage1PathRel:
		return s.getRelPath()
	case stubStage1PathAbs:
		return s.getAbsPath()
	case stubStage1Name:
		return fmt.Sprintf("%s:%s", s.name, s.version)
	case stubStage1Base:
		return filepath.Base(s.path)
	default:
		panic(fmt.Sprintf("Invalid stub stage1 reference type: %d", refType))
	}
}

func (s *stubStage1Setup) getServerHost() string {
	u, err := url.Parse(s.server.URL)
	if err != nil {
		panic(fmt.Sprintf("Invalid server URL %q: %v", s.server.URL, err))
	}
	return u.Host
}

func (s *stubStage1Setup) getAbsPath() string {
	if filepath.IsAbs(s.path) {
		return s.path
	}
	abs, err := filepath.Abs(s.path)
	if err != nil {
		s.t.Fatalf("Failed to get the absolute path to the stub stage1 image (based on %s)", s.path)
	}
	return abs
}

func (s *stubStage1Setup) getRelPath() string {
	if !filepath.IsAbs(s.path) {
		return s.path
	}
	wd, err := os.Getwd()
	if err != nil {
		s.t.Fatalf("Failed to get current working directory")
	}
	rel, err := filepath.Rel(wd, s.path)
	if err != nil {
		s.t.Fatalf("Failed to get the path to stub stage1 image (based on %s) relative to %s", s.path, wd)
	}
	return rel
}

func (s *stubStage1Setup) generateStage1ImagesDirectoryConfig(path string) {
	tmpl := `
{
	"rktKind": "paths",
	"rktVersion": "v1",
	"stage1-images": "^STAGE1_IMAGES_DIR^"
}
`
	replacements := map[string]string{
		"^STAGE1_IMAGES_DIR^": path,
	}
	s.generateConfigContents("paths.d", tmpl, replacements)
}

func (s *stubStage1Setup) generateStage1Config(data *config.Stage1Data) {
	tmpl := `
{
	"rktKind": "stage1",
	"rktVersion": "v1",
	"name": "^NAME^",
	"version": "^VERSION^",
	"location": "^LOCATION^"
}
`
	replacements := map[string]string{
		"^NAME^":     data.Name,
		"^VERSION^":  data.Version,
		"^LOCATION^": data.Location,
	}
	s.generateConfigContents("stage1.d", tmpl, replacements)
}

func (s *stubStage1Setup) generateConfigContents(subdir, tmpl string, replacements map[string]string) {
	dir := filepath.Join(s.ctx.LocalDir(), subdir)
	file := filepath.Join(dir, "cfg.json")
	contents := tmpl
	for k, v := range replacements {
		contents = strings.Replace(contents, k, v, -1)
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		s.t.Fatalf("Failed to create config dir %q: %v", dir, err)
	}
	if err := ioutil.WriteFile(file, []byte(contents), 0644); err != nil {
		s.t.Fatalf("Failed to generate the config in the %q directory: %v", subdir, err)
	}
}

func (s *stubStage1Setup) check(flag, expectedLine string) {
	cmd := fmt.Sprintf("%s --insecure-options=image,tls --debug run %s %s", s.ctx.Cmd(), flag, getInspectImagePath())
	child := spawnOrFail(s.t, cmd)
	defer waitOrFail(s.t, child, 0)
	s.getExpectedOrFail(child, expectedLine)
	s.getExpectedOrFail(child, stubStage1Output)
}

func (s *stubStage1Setup) getExpectedOrFail(child *gexpect.ExpectSubprocess, expectedLine string) {
	if err := expectWithOutput(child, expectedLine); err != nil {
		s.t.Fatalf("Did not get the expected string %q:\n%v", expectedLine, err)
	}
}
