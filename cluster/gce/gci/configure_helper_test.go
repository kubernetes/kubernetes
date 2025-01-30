/*
Copyright 2017 The Kubernetes Authors.

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

package gci

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"text/template"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

const (
	envScriptFileName = "kube-env"
)

type ManifestTestCase struct {
	pod                 v1.Pod
	manifest            string
	auxManifests        []string
	kubeHome            string
	manifestSources     string
	manifestDestination string
	manifestTemplateDir string
	manifestTemplate    string
	manifestFuncName    string
	t                   *testing.T
}

func newManifestTestCase(t *testing.T, manifest, funcName string, auxManifests []string) *ManifestTestCase {
	c := &ManifestTestCase{
		t:                t,
		manifest:         manifest,
		auxManifests:     auxManifests,
		manifestFuncName: funcName,
	}

	d, err := os.MkdirTemp("", "configure-helper-test")
	if err != nil {
		c.t.Fatalf("Failed to create temp directory: %v", err)
	}

	c.kubeHome = d
	c.manifestSources = filepath.Join(c.kubeHome, "kube-manifests", "kubernetes", "gci-trusty")

	currentPath, err := os.Getwd()
	if err != nil {
		c.t.Fatalf("Failed to get current directory: %v", err)
	}
	gceDir := filepath.Dir(currentPath)
	c.manifestTemplateDir = filepath.Join(gceDir, "manifests")
	c.manifestTemplate = filepath.Join(c.manifestTemplateDir, c.manifest)
	c.manifestDestination = filepath.Join(c.kubeHome, "etc", "kubernetes", "manifests", c.manifest)

	c.mustCopyFromTemplate()
	c.mustCopyAuxFromTemplate()
	c.mustCreateManifestDstDir()

	return c
}

func (c *ManifestTestCase) mustCopyFromTemplate() {
	if err := os.MkdirAll(c.manifestSources, os.ModePerm); err != nil {
		c.t.Fatalf("Failed to create source directory: %v", err)
	}

	if err := copyFile(c.manifestTemplate, filepath.Join(c.manifestSources, c.manifest)); err != nil {
		c.t.Fatalf("Failed to copy source manifest to KUBE_HOME: %v", err)
	}
}

func (c *ManifestTestCase) mustCopyAuxFromTemplate() {
	for _, m := range c.auxManifests {
		err := copyFile(filepath.Join(c.manifestTemplateDir, m), filepath.Join(c.manifestSources, m))
		if err != nil {
			c.t.Fatalf("Failed to copy source manifest %s to KUBE_HOME: %v", m, err)
		}
	}
}

func (c *ManifestTestCase) mustCreateManifestDstDir() {
	p := filepath.Join(filepath.Join(c.kubeHome, "etc", "kubernetes", "manifests"))
	if err := os.MkdirAll(p, os.ModePerm); err != nil {
		c.t.Fatalf("Failed to create designation folder for kube-apiserver.manifest: %v", err)
	}
}

func (c *ManifestTestCase) mustInvokeFunc(env interface{}, scriptNames []string, targetTemplate string, templates ...string) {
	envScriptPath := c.mustCreateEnv(env, targetTemplate, templates...)
	args := fmt.Sprintf("source %q ;", envScriptPath)
	for _, script := range scriptNames {
		args += fmt.Sprintf("source %q ;", script)
	}
	args += c.manifestFuncName
	cmd := exec.Command("bash", "-c", args)

	bs, err := cmd.CombinedOutput()
	if err != nil {
		c.t.Logf("%q", bs)
		c.t.Fatalf("Failed to run %q: %v", cmd.Args, err)
	}
	c.t.Logf("%s", string(bs))
}

func (c *ManifestTestCase) mustCreateEnv(env interface{}, target string, templates ...string) string {
	f, err := os.Create(filepath.Join(c.kubeHome, envScriptFileName))
	if err != nil {
		c.t.Fatalf("Failed to create envScript: %v", err)
	}
	defer f.Close()

	t, err := template.ParseFiles(templates...)
	if err != nil {
		c.t.Fatalf("Failed to parse files %q, err: %v", templates, err)
	}

	if err = t.ExecuteTemplate(f, target, env); err != nil {
		c.t.Fatalf("Failed to execute template %s, err: %v", target, err)
	}

	return f.Name()
}

func (c *ManifestTestCase) mustLoadPodFromManifest() {
	json, err := os.ReadFile(c.manifestDestination)
	if err != nil {
		c.t.Fatalf("Failed to read manifest: %s, %v", c.manifestDestination, err)
	}

	if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), json, &c.pod); err != nil {
		c.t.Fatalf("Failed to decode manifest:\n%s\nerror: %v", json, err)
	}
}

func (c *ManifestTestCase) tearDown() {
	if err := os.RemoveAll(c.kubeHome); err != nil {
		c.t.Fatalf("Failed to teardown: %s", err)
	}
}

func copyFile(src, dst string) (err error) {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer func() {
		cerr := out.Close()
		if err == nil {
			err = cerr
		}
	}()
	_, err = io.Copy(out, in)
	return err
}
