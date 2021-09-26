// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/protobuf/internal/version"
)

var (
	regenerate   = flag.Bool("regenerate", false, "regenerate files")
	buildRelease = flag.Bool("buildRelease", false, "build release binaries")

	protobufVersion = "ef7cc811" // v3.12.0-rc1
	protobufSHA256  = ""         // ignored if protobufVersion is a git hash

	golangVersions = []string{"1.9.7", "1.10.8", "1.11.13", "1.12.17", "1.13.12", "1.14.4"}
	golangLatest   = golangVersions[len(golangVersions)-1]

	staticcheckVersion = "2020.1.4"
	staticcheckSHA256s = map[string]string{
		"darwin/386":   "05ccb332a0c5ba812af165b0e69ffe317cb3e8bb10b0f4b4c4eaaf956ba9a50b",
		"darwin/amd64": "5706d101426c025e8f165309e0cb2932e54809eb035ff23ebe19df0f810699d8",
		"linux/386":    "e4dbf94e940678ae7108f0d22c7c2992339bc10a8fb384e7e734b1531a429a1c",
		"linux/amd64":  "09d2c2002236296de2c757df111fe3ae858b89f9e183f645ad01f8135c83c519",
	}

	// purgeTimeout determines the maximum age of unused sub-directories.
	purgeTimeout = 30 * 24 * time.Hour // 1 month

	// Variables initialized by mustInitDeps.
	goPath       string
	modulePath   string
	protobufPath string
)

func Test(t *testing.T) {
	mustInitDeps(t)
	mustHandleFlags(t)

	// Report dirt in the working tree quickly, rather than after
	// going through all the presubmits.
	//
	// Fail the test late, so we can test uncommitted changes with -failfast.
	gitDiff := mustRunCommand(t, "git", "diff", "HEAD")
	if strings.TrimSpace(gitDiff) != "" {
		fmt.Printf("WARNING: working tree contains uncommitted changes:\n%v\n", gitDiff)
	}
	gitUntracked := mustRunCommand(t, "git", "ls-files", "--others", "--exclude-standard")
	if strings.TrimSpace(gitUntracked) != "" {
		fmt.Printf("WARNING: working tree contains untracked files:\n%v\n", gitUntracked)
	}

	// Do the relatively fast checks up-front.
	t.Run("GeneratedGoFiles", func(t *testing.T) {
		diff := mustRunCommand(t, "go", "run", "-tags", "protolegacy", "./internal/cmd/generate-types")
		if strings.TrimSpace(diff) != "" {
			t.Fatalf("stale generated files:\n%v", diff)
		}
		diff = mustRunCommand(t, "go", "run", "-tags", "protolegacy", "./internal/cmd/generate-protos")
		if strings.TrimSpace(diff) != "" {
			t.Fatalf("stale generated files:\n%v", diff)
		}
	})
	t.Run("FormattedGoFiles", func(t *testing.T) {
		files := strings.Split(strings.TrimSpace(mustRunCommand(t, "git", "ls-files", "*.go")), "\n")
		diff := mustRunCommand(t, append([]string{"gofmt", "-d"}, files...)...)
		if strings.TrimSpace(diff) != "" {
			t.Fatalf("unformatted source files:\n%v", diff)
		}
	})
	t.Run("CopyrightHeaders", func(t *testing.T) {
		files := strings.Split(strings.TrimSpace(mustRunCommand(t, "git", "ls-files", "*.go", "*.proto")), "\n")
		mustHaveCopyrightHeader(t, files)
	})

	var wg sync.WaitGroup
	sema := make(chan bool, (runtime.NumCPU()+1)/2)
	for i := range golangVersions {
		goVersion := golangVersions[i]
		goLabel := "Go" + goVersion
		runGo := func(label, workDir string, args ...string) {
			wg.Add(1)
			sema <- true
			go func() {
				defer wg.Done()
				defer func() { <-sema }()
				t.Run(goLabel+"/"+label, func(t *testing.T) {
					args[0] += goVersion
					command{Dir: workDir}.mustRun(t, args...)
				})
			}()
		}

		workDir := filepath.Join(goPath, "src", modulePath)
		runGo("Normal", workDir, "go", "test", "-race", "./...")
		runGo("PureGo", workDir, "go", "test", "-race", "-tags", "purego", "./...")
		runGo("Reflect", workDir, "go", "test", "-race", "-tags", "protoreflect", "./...")
		if goVersion == golangLatest {
			runGo("ProtoLegacy", workDir, "go", "test", "-race", "-tags", "protolegacy", "./...")
			runGo("ProtocGenGo", "cmd/protoc-gen-go/testdata", "go", "test")
			runGo("Conformance", "internal/conformance", "go", "test", "-execute")
		}
	}
	wg.Wait()

	t.Run("GoStaticCheck", func(t *testing.T) {
		checks := []string{
			"all",     // start with all checks enabled
			"-SA1019", // disable deprecated usage check
			"-S*",     // disable code simplication checks
			"-ST*",    // disable coding style checks
			"-U*",     // disable unused declaration checks
		}
		out := mustRunCommand(t, "staticcheck", "-checks="+strings.Join(checks, ","), "-fail=none", "./...")

		// Filter out findings from certain paths.
		var findings []string
		for _, finding := range strings.Split(strings.TrimSpace(out), "\n") {
			switch {
			case strings.HasPrefix(finding, "internal/testprotos/legacy/"):
			default:
				findings = append(findings, finding)
			}
		}
		if len(findings) > 0 {
			t.Fatalf("staticcheck findings:\n%v", strings.Join(findings, "\n"))
		}
	})
	t.Run("CommittedGitChanges", func(t *testing.T) {
		if strings.TrimSpace(gitDiff) != "" {
			t.Fatalf("uncommitted changes")
		}
	})
	t.Run("TrackedGitFiles", func(t *testing.T) {
		if strings.TrimSpace(gitUntracked) != "" {
			t.Fatalf("untracked files")
		}
	})
}

func mustInitDeps(t *testing.T) {
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}

	// Determine the directory to place the test directory.
	repoRoot, err := os.Getwd()
	check(err)
	testDir := filepath.Join(repoRoot, ".cache")
	check(os.MkdirAll(testDir, 0775))

	// Travis-CI has a hard-coded timeout where it kills the test after
	// 10 minutes of a lack of activity on stdout.
	// We work around this restriction by periodically printing the timestamp.
	ticker := time.NewTicker(5 * time.Minute)
	done := make(chan struct{})
	go func() {
		now := time.Now()
		for {
			select {
			case t := <-ticker.C:
				fmt.Printf("\tt=%0.fmin\n", t.Sub(now).Minutes())
			case <-done:
				return
			}
		}
	}()
	defer close(done)
	defer ticker.Stop()

	// Delete the current directory if non-empty,
	// which only occurs if a dependency failed to initialize properly.
	var workingDir string
	defer func() {
		if workingDir != "" {
			os.RemoveAll(workingDir) // best-effort
		}
	}()

	// Delete other sub-directories that are no longer relevant.
	defer func() {
		subDirs := map[string]bool{"bin": true, "gopath": true}
		subDirs["protobuf-"+protobufVersion] = true
		for _, v := range golangVersions {
			subDirs["go"+v] = true
		}

		now := time.Now()
		fis, _ := ioutil.ReadDir(testDir)
		for _, fi := range fis {
			if subDirs[fi.Name()] {
				os.Chtimes(filepath.Join(testDir, fi.Name()), now, now) // best-effort
				continue
			}
			if now.Sub(fi.ModTime()) < purgeTimeout {
				continue
			}
			fmt.Printf("delete %v\n", fi.Name())
			os.RemoveAll(filepath.Join(testDir, fi.Name())) // best-effort
		}
	}()

	// The bin directory contains symlinks to each tool by version.
	// It is safe to delete this directory and run the test script from scratch.
	binPath := filepath.Join(testDir, "bin")
	check(os.RemoveAll(binPath))
	check(os.Mkdir(binPath, 0775))
	check(os.Setenv("PATH", binPath+":"+os.Getenv("PATH")))
	registerBinary := func(name, path string) {
		check(os.Symlink(path, filepath.Join(binPath, name)))
	}

	// Download and build the protobuf toolchain.
	// We avoid downloading the pre-compiled binaries since they do not contain
	// the conformance test runner.
	workingDir = filepath.Join(testDir, "protobuf-"+protobufVersion)
	protobufPath = workingDir
	if _, err := os.Stat(protobufPath); err != nil {
		fmt.Printf("download %v\n", filepath.Base(protobufPath))
		if isCommit := strings.Trim(protobufVersion, "0123456789abcdef") == ""; isCommit {
			command{Dir: testDir}.mustRun(t, "git", "clone", "https://github.com/protocolbuffers/protobuf", "protobuf-"+protobufVersion)
			command{Dir: protobufPath}.mustRun(t, "git", "checkout", protobufVersion)
		} else {
			url := fmt.Sprintf("https://github.com/google/protobuf/releases/download/v%v/protobuf-all-%v.tar.gz", protobufVersion, protobufVersion)
			downloadArchive(check, protobufPath, url, "protobuf-"+protobufVersion, protobufSHA256)
		}

		fmt.Printf("build %v\n", filepath.Base(protobufPath))
		command{Dir: protobufPath}.mustRun(t, "./autogen.sh")
		command{Dir: protobufPath}.mustRun(t, "./configure")
		command{Dir: protobufPath}.mustRun(t, "make")
		command{Dir: filepath.Join(protobufPath, "conformance")}.mustRun(t, "make")
	}
	check(os.Setenv("PROTOBUF_ROOT", protobufPath)) // for generate-protos
	registerBinary("conform-test-runner", filepath.Join(protobufPath, "conformance", "conformance-test-runner"))
	registerBinary("protoc", filepath.Join(protobufPath, "src", "protoc"))
	workingDir = ""

	// Download each Go toolchain version.
	for _, v := range golangVersions {
		workingDir = filepath.Join(testDir, "go"+v)
		if _, err := os.Stat(workingDir); err != nil {
			fmt.Printf("download %v\n", filepath.Base(workingDir))
			url := fmt.Sprintf("https://dl.google.com/go/go%v.%v-%v.tar.gz", v, runtime.GOOS, runtime.GOARCH)
			downloadArchive(check, workingDir, url, "go", "") // skip SHA256 check as we fetch over https from a trusted domain
		}
		registerBinary("go"+v, filepath.Join(workingDir, "bin", "go"))
	}
	registerBinary("go", filepath.Join(testDir, "go"+golangLatest, "bin", "go"))
	registerBinary("gofmt", filepath.Join(testDir, "go"+golangLatest, "bin", "gofmt"))
	workingDir = ""

	// Download the staticcheck tool.
	workingDir = filepath.Join(testDir, "staticcheck-"+staticcheckVersion)
	if _, err := os.Stat(workingDir); err != nil {
		fmt.Printf("download %v\n", filepath.Base(workingDir))
		url := fmt.Sprintf("https://github.com/dominikh/go-tools/releases/download/%v/staticcheck_%v_%v.tar.gz", staticcheckVersion, runtime.GOOS, runtime.GOARCH)
		downloadArchive(check, workingDir, url, "staticcheck", staticcheckSHA256s[runtime.GOOS+"/"+runtime.GOARCH])
	}
	registerBinary("staticcheck", filepath.Join(workingDir, "staticcheck"))
	workingDir = ""

	// Travis-CI sets GOROOT, which confuses invocations of the Go toolchain.
	// Explicitly clear GOROOT, so each toolchain uses their default GOROOT.
	check(os.Unsetenv("GOROOT"))

	// Set a cache directory outside the test directory.
	check(os.Setenv("GOCACHE", filepath.Join(repoRoot, ".gocache")))

	// Setup GOPATH for pre-module support (i.e., go1.10 and earlier).
	goPath = filepath.Join(testDir, "gopath")
	modulePath = strings.TrimSpace(command{Dir: testDir}.mustRun(t, "go", "list", "-m", "-f", "{{.Path}}"))
	check(os.RemoveAll(filepath.Join(goPath, "src")))
	check(os.MkdirAll(filepath.Join(goPath, "src", filepath.Dir(modulePath)), 0775))
	check(os.Symlink(repoRoot, filepath.Join(goPath, "src", modulePath)))
	command{Dir: repoRoot}.mustRun(t, "go", "mod", "tidy")
	command{Dir: repoRoot}.mustRun(t, "go", "mod", "vendor")
	check(os.Setenv("GOPATH", goPath))
}

func downloadFile(check func(error), dstPath, srcURL string) {
	resp, err := http.Get(srcURL)
	check(err)
	defer resp.Body.Close()

	check(os.MkdirAll(filepath.Dir(dstPath), 0775))
	f, err := os.Create(dstPath)
	check(err)

	_, err = io.Copy(f, resp.Body)
	check(err)
}

func downloadArchive(check func(error), dstPath, srcURL, skipPrefix, wantSHA256 string) {
	check(os.RemoveAll(dstPath))

	resp, err := http.Get(srcURL)
	check(err)
	defer resp.Body.Close()

	var r io.Reader = resp.Body
	if wantSHA256 != "" {
		b, err := ioutil.ReadAll(resp.Body)
		check(err)
		r = bytes.NewReader(b)

		if gotSHA256 := fmt.Sprintf("%x", sha256.Sum256(b)); gotSHA256 != wantSHA256 {
			check(fmt.Errorf("checksum validation error:\ngot  %v\nwant %v", gotSHA256, wantSHA256))
		}
	}

	zr, err := gzip.NewReader(r)
	check(err)

	tr := tar.NewReader(zr)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return
		}
		check(err)

		// Skip directories or files outside the prefix directory.
		if len(skipPrefix) > 0 {
			if !strings.HasPrefix(h.Name, skipPrefix) {
				continue
			}
			if len(h.Name) > len(skipPrefix) && h.Name[len(skipPrefix)] != '/' {
				continue
			}
		}

		path := strings.TrimPrefix(strings.TrimPrefix(h.Name, skipPrefix), "/")
		path = filepath.Join(dstPath, filepath.FromSlash(path))
		mode := os.FileMode(h.Mode & 0777)
		switch h.Typeflag {
		case tar.TypeReg:
			b, err := ioutil.ReadAll(tr)
			check(err)
			check(ioutil.WriteFile(path, b, mode))
		case tar.TypeDir:
			check(os.Mkdir(path, mode))
		}
	}
}

func mustHandleFlags(t *testing.T) {
	if *regenerate {
		t.Run("Generate", func(t *testing.T) {
			fmt.Print(mustRunCommand(t, "go", "run", "-tags", "protolegacy", "./internal/cmd/generate-types", "-execute"))
			fmt.Print(mustRunCommand(t, "go", "run", "-tags", "protolegacy", "./internal/cmd/generate-protos", "-execute"))
			files := strings.Split(strings.TrimSpace(mustRunCommand(t, "git", "ls-files", "*.go")), "\n")
			mustRunCommand(t, append([]string{"gofmt", "-w"}, files...)...)
		})
	}
	if *buildRelease {
		t.Run("BuildRelease", func(t *testing.T) {
			v := version.String()
			for _, goos := range []string{"linux", "darwin", "windows"} {
				for _, goarch := range []string{"386", "amd64"} {
					binPath := filepath.Join("bin", fmt.Sprintf("protoc-gen-go.%v.%v.%v", v, goos, goarch))

					// Build the binary.
					cmd := command{Env: append(os.Environ(), "GOOS="+goos, "GOARCH="+goarch)}
					cmd.mustRun(t, "go", "build", "-trimpath", "-ldflags", "-s -w -buildid=", "-o", binPath, "./cmd/protoc-gen-go")

					// Archive and compress the binary.
					in, err := ioutil.ReadFile(binPath)
					if err != nil {
						t.Fatal(err)
					}
					out := new(bytes.Buffer)
					gz, _ := gzip.NewWriterLevel(out, gzip.BestCompression)
					gz.Comment = fmt.Sprintf("protoc-gen-go VERSION=%v GOOS=%v GOARCH=%v", v, goos, goarch)
					tw := tar.NewWriter(gz)
					tw.WriteHeader(&tar.Header{
						Name: "protoc-gen-go",
						Mode: int64(0775),
						Size: int64(len(in)),
					})
					tw.Write(in)
					tw.Close()
					gz.Close()
					if err := ioutil.WriteFile(binPath+".tar.gz", out.Bytes(), 0664); err != nil {
						t.Fatal(err)
					}
				}
			}
		})
	}
	if *regenerate || *buildRelease {
		t.SkipNow()
	}
}

var copyrightRegex = []*regexp.Regexp{
	regexp.MustCompile(`^// Copyright \d\d\d\d The Go Authors\. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file\.
`),
	// Generated .pb.go files from main protobuf repo.
	regexp.MustCompile(`^// Protocol Buffers - Google's data interchange format
// Copyright \d\d\d\d Google Inc\.  All rights reserved\.
`),
}

func mustHaveCopyrightHeader(t *testing.T, files []string) {
	var bad []string
File:
	for _, file := range files {
		b, err := ioutil.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		for _, re := range copyrightRegex {
			if loc := re.FindIndex(b); loc != nil && loc[0] == 0 {
				continue File
			}
		}
		bad = append(bad, file)
	}
	if len(bad) > 0 {
		t.Fatalf("files with missing/bad copyright headers:\n  %v", strings.Join(bad, "\n  "))
	}
}

type command struct {
	Dir string
	Env []string
}

func (c command) mustRun(t *testing.T, args ...string) string {
	t.Helper()
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = "."
	if c.Dir != "" {
		cmd.Dir = c.Dir
	}
	cmd.Env = os.Environ()
	if c.Env != nil {
		cmd.Env = c.Env
	}
	cmd.Env = append(cmd.Env, "PWD="+cmd.Dir)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("executing (%v): %v\n%s%s", strings.Join(args, " "), err, stdout.String(), stderr.String())
	}
	return stdout.String()
}

func mustRunCommand(t *testing.T, args ...string) string {
	t.Helper()
	return command{}.mustRun(t, args...)
}
