/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// The go2docker command compiles a go main package and forge a minimal
// docker image from the resulting static binary.
//
// usage: go2docker [-image namespace/basename] go/pkg/path | docker load
package main

import (
	"archive/tar"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"time"
)

type Config struct {
	Cmd []string `json:"Cmd"`
}

type Image struct {
	ID            string    `json:"id"`
	Created       time.Time `json:"created"`
	DockerVersion string    `json:"docker_version"`
	Config        Config    `json:"config"`
	Architecture  string    `json:"architecture"`
	OS            string    `json:"os"`
}

var image = flag.String("image", "", "namespace/name for the repository, default to go2docker/$(basename)")

const (
	DockerVersion = "1.4.0"
	Arch          = "amd64"
	OS            = "linux"
	Version       = "1.0"
	Namespace     = "go2docker"
)

func main() {
	flag.Parse()
	args := []string{"."}
	if flag.NArg() > 0 {
		args = flag.Args()
	}

	// check for go
	goBin, err := exec.LookPath("go")
	if err != nil {
		log.Fatalf("`go` executable not found: %v, see %q ", err, "golang.org/doc/install")
	}
	toolPath := filepath.Dir(goBin)
	// check for linux_amd64 toolchain
	crossPath := filepath.Join(toolPath, "linux_amd64")
	if _, crossErr := os.Stat(crossPath); os.IsNotExist(crossErr) {
		// check for make.bash
		makeBash, err := filepath.Abs(filepath.Join(toolPath, "..", "src", "make.bash"))
		if err != nil {
			log.Fatalf("failed to resolve make.bash path: %v", err)
		}
		if _, err := os.Stat(makeBash); os.IsNotExist(err) {
			log.Fatalf("`make.bash` not found %q: %v", err)
		}
		makeBashCmd := fmt.Sprintf("(cd %s; GOOS=linux GOARCH=amd64 ./make.bash --no-clean)", filepath.Dir(makeBash))
		log.Fatalf("`go %s` toolchain not found: %v, run: %q", "linux_amd64", crossErr, makeBashCmd)
	}

	fpath, err := filepath.Abs(args[0])
	ext := filepath.Ext(fpath)
	basename := filepath.Base(fpath[:len(fpath)-len(ext)])

	if *image == "" {
		if err != nil {
			log.Fatalf("failed to get absolute path: %v", err)
		}
		*image = path.Join(Namespace, basename)
	}
	tmpDir, err := ioutil.TempDir("", "")
	if err != nil {
		log.Fatalf("failed to create temp directory: %v", err)
	}
	aout := filepath.Join(tmpDir, basename)
	command := append([]string{"go", "build", "-o", aout, "-a", "-tags", "netgo", "-installsuffix", "netgo"}, args...)
	cmd := exec.Command(command[0], command[1:]...)
	gopath := os.Getenv("GOPATH")
	cmd.Env = []string{
		"GOOS=linux",
		"GOARCH=amd64",
		"GOPATH=" + gopath,
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Fatalf("failed to get process stderr: %v", err)
	}
	if err := cmd.Start(); err != nil {
		log.Fatalf("failed to start command %q: %v", strings.Join(command, " "), err)
	}
	io.Copy(os.Stderr, stderr)
	if err := cmd.Wait(); err != nil {
		log.Fatalf("command %q failed: %v", strings.Join(command, " "), err)
	}

	imageIDBytes := make([]byte, 32)
	if _, err := rand.Read(imageIDBytes); err != nil {
		log.Fatalf("failed to generate ID: %v")
	}
	imageID := hex.EncodeToString(imageIDBytes)
	repo := map[string]map[string]string{
		*image: {
			"latest": imageID,
		},
	}
	repoJSON, err := json.Marshal(repo)
	if err != nil {
		log.Fatalf("failed to serialize repo %#v: %v", repo, err)
	}
	tw := tar.NewWriter(os.Stdout)
	if err := tw.WriteHeader(&tar.Header{
		Name: "repositories",
		Size: int64(len(repoJSON)),
	}); err != nil {
		log.Fatalf("failed to write /repository header: %v", err)
	}
	if _, err := tw.Write(repoJSON); err != nil {
		log.Fatalf(" failed to write /repository body: %v", err)
	}
	if err := tw.WriteHeader(&tar.Header{
		Name: imageID + "/VERSION",
		Size: int64(len(Version)),
	}); err != nil {
		log.Fatalf("failed to write /%s/VERSION header: %v", imageID, err)
	}
	if _, err := tw.Write([]byte(Version)); err != nil {
		log.Fatalf(" failed to write /%s/VERSION body: %v", imageID, err)
	}
	imageJSON, err := json.Marshal(Image{
		ID:            imageID,
		Created:       time.Now().UTC(),
		DockerVersion: Version,
		Config: Config{
			Cmd: []string{"/" + basename},
		},
		Architecture: Arch,
		OS:           OS,
	})
	if err := tw.WriteHeader(&tar.Header{
		Name: imageID + "/json",
		Size: int64(len(imageJSON)),
	}); err != nil {
		log.Fatalf("failed to write /%s/json header: %v", imageID, err)
	}
	if _, err := tw.Write(imageJSON); err != nil {
		log.Fatalf("failed to write /%s/json body: %v", imageID, err)
	}
	var buf bytes.Buffer
	ftw := tar.NewWriter(&buf)
	file, err := os.Open(aout)
	if err != nil {
		log.Fatalf("failed to open %q: %v", aout, err)
	}
	finfo, err := file.Stat()
	if err != nil {
		log.Fatalf("failed to get file info %q: %v", aout, err)
	}
	fheader, err := tar.FileInfoHeader(finfo, "")
	if err != nil {
		log.Fatalf("failed to get file info header %q: %v", aout, err)
	}
	fheader.Name = basename
	if err := ftw.WriteHeader(fheader); err != nil {
		log.Fatalf("failed to write /%s header: %v", aout, err)
	}
	if _, err := io.Copy(ftw, file); err != nil {
		log.Fatalf("failed to write /%s body: %v", aout, err)
	}
	certBytes := []byte(caCerts)
	if err := ftw.WriteHeader(&tar.Header{
		Name: "/etc/ssl/certs/ca-certificates.crt",
		Size: int64(len(certBytes)),
	}); err != nil {
		log.Fatalf("failed to write ca-certificates.crt header: %v", err)
	}
	if _, err := ftw.Write(certBytes); err != nil {
		log.Fatalf("failed to write ca-certificates.crt body: %v", err)
	}
	if err := ftw.Close(); err != nil {
		log.Fatalf("failed to close layer.tar: %v", err)
	}
	layerBytes := buf.Bytes()
	if err := tw.WriteHeader(&tar.Header{
		Name: imageID + "/layer.tar",
		Size: int64(len(layerBytes)),
	}); err != nil {
		log.Fatalf("failed to write /%s/layer.tar header: %v", imageID, err)
	}
	if _, err := tw.Write(layerBytes); err != nil {
		log.Fatalf("failed to write /%s/layer.tar body: %v", imageID, err)
	}
	if err := tw.Close(); err != nil {
		log.Fatalf("failed to close image.tar: %v", err)
	}
}
