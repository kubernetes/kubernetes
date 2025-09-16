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

package configfiles

import (
	"errors"
	"fmt"
	"path/filepath"
	goruntime "runtime"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletscheme "k8s.io/kubernetes/pkg/kubelet/apis/config/scheme"
	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

const configDir = "/test-config-dir"
const relativePath = "relative/path/test"
const kubeletFile = "kubelet"

func TestLoad(t *testing.T) {
	tCtx := ktesting.Init(t)
	cases := []struct {
		desc          string
		file          *string
		expect        *kubeletconfig.KubeletConfiguration
		err           string
		strictErr     bool
		skipOnWindows bool
	}{
		// missing file
		{
			desc: "missing file",
			err:  "failed to read",
		},
		// empty file
		{
			desc: "empty file",
			file: ptr.To(``),
			err:  "was empty",
		},
		// invalid format
		{
			desc: "invalid yaml",
			file: ptr.To(`*`),
			err:  "failed to decode",
		},
		{
			desc: "invalid json",
			file: ptr.To(`{*`),
			err:  "failed to decode",
		},
		// invalid object
		{
			desc: "missing kind",
			file: ptr.To(`{"apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			err:  "failed to decode",
		},
		{
			desc: "missing version",
			file: ptr.To(`{"kind":"KubeletConfiguration"}`),
			err:  "failed to decode",
		},
		{
			desc: "unregistered kind",
			file: ptr.To(`{"kind":"BogusKind","apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			err:  "failed to decode",
		},
		{
			desc: "unregistered version",
			file: ptr.To(`{"kind":"KubeletConfiguration","apiVersion":"bogusversion"}`),
			err:  "failed to decode",
		},

		// empty object with correct kind and version should result in the defaults for that kind and version
		{
			desc: "default from yaml",
			file: ptr.To(`kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1`),
			expect:        newConfig(t),
			skipOnWindows: true,
		},
		{
			desc:          "default from json",
			file:          ptr.To(`{"kind":"KubeletConfiguration","apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			expect:        newConfig(t),
			skipOnWindows: true,
		},

		// relative path
		{
			desc: "yaml, relative path is resolved",
			file: ptr.To(fmt.Sprintf(`kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
staticPodPath: %s`, relativePath)),
			expect: func() *kubeletconfig.KubeletConfiguration {
				kc := newConfig(t)
				kc.StaticPodPath = filepath.Join(configDir, relativePath)
				return kc
			}(),
			skipOnWindows: true,
		},
		{
			desc: "json, relative path is resolved",
			file: ptr.To(fmt.Sprintf(`{"kind":"KubeletConfiguration","apiVersion":"kubelet.config.k8s.io/v1beta1","staticPodPath":"%s"}`, relativePath)),
			expect: func() *kubeletconfig.KubeletConfiguration {
				kc := newConfig(t)
				kc.StaticPodPath = filepath.Join(configDir, relativePath)
				return kc
			}(),
			skipOnWindows: true,
		},
		{
			// This should fail from v1beta2+
			desc: "duplicate field",
			file: ptr.To(fmt.Sprintf(`kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
staticPodPath: %s
staticPodPath: %s/foo`, relativePath, relativePath)),
			// err:       `key "staticPodPath" already set`,
			// strictErr: true,
			expect: func() *kubeletconfig.KubeletConfiguration {
				kc := newConfig(t)
				kc.StaticPodPath = filepath.Join(configDir, relativePath, "foo")
				return kc
			}(),
			skipOnWindows: true,
		},
		{
			// This should fail from v1beta2+
			desc: "unknown field",
			file: ptr.To(`kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
foo: bar`),
			// err:       "found unknown field: foo",
			// strictErr: true,
			expect:        newConfig(t),
			skipOnWindows: true,
		},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			// Skip tests that fail on Windows, as discussed during the SIG Testing meeting from January 10, 2023
			if c.skipOnWindows && goruntime.GOOS == "windows" {
				t.Skip("Skipping test that fails on Windows")
			}

			fs := utilfs.NewTempFs()
			fs.MkdirAll(configDir, 0777)
			path := filepath.Join(configDir, kubeletFile)
			if c.file != nil {
				if err := addFile(fs, path, *c.file); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
			loader, err := NewFsLoader(fs, path)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			kc, err := loader.Load(tCtx)

			if c.strictErr && !runtime.IsStrictDecodingError(errors.Unwrap(err)) {
				t.Fatalf("got error: %v, want strict decoding error", err)
			}
			if utiltest.SkipRest(t, c.desc, err, c.err) {
				return
			}
			if !apiequality.Semantic.DeepEqual(c.expect, kc) {
				t.Fatalf("expect %#v but got %#v", *c.expect, *kc)
			}
		})
	}
}

func TestResolveRelativePaths(t *testing.T) {
	absolutePath := filepath.Join(configDir, "absolute")
	cases := []struct {
		desc          string
		path          string
		expect        string
		skipOnWindows bool
	}{
		{"empty path", "", "", false},
		{"absolute path", absolutePath, absolutePath, true},
		{"relative path", relativePath, filepath.Join(configDir, relativePath), false},
	}

	paths := kubeletconfig.KubeletConfigurationPathRefs(newConfig(t))
	if len(paths) == 0 {
		t.Fatalf("requires at least one path field to exist in the KubeletConfiguration type")
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			// Skip tests that fail on Windows, as discussed during the SIG Testing meeting from January 10, 2023
			if c.skipOnWindows && goruntime.GOOS == "windows" {
				t.Skip("Skipping test that fails on Windows")
			}

			// set the path, resolve it, and check if it resolved as we would expect
			*(paths[0]) = c.path
			resolveRelativePaths(paths, configDir)
			if *(paths[0]) != c.expect {
				t.Fatalf("expect %s but got %s", c.expect, *(paths[0]))
			}
		})
	}
}

func addFile(fs utilfs.Filesystem, path string, fileContent string) error {
	dir := filepath.Dir(path)
	tmpFile, err := fs.TempFile(dir, "tmp_"+filepath.Base(path))
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()

	if _, err = tmpFile.Write([]byte(fileContent)); err != nil {
		_ = tmpFile.Close()
		_ = fs.Remove(tmpPath)
		return fmt.Errorf("failed to write to temp file: %w", err)
	}
	if err = tmpFile.Close(); err != nil {
		_ = fs.Remove(tmpPath)
		return fmt.Errorf("failed to close temp file: %w", err)
	}

	return fs.Rename(tmpPath, path)
}

func newConfig(t *testing.T) *kubeletconfig.KubeletConfiguration {
	kubeletScheme, _, err := kubeletscheme.NewSchemeAndCodecs()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// get the built-in default configuration
	external := &kubeletconfigv1beta1.KubeletConfiguration{}
	kubeletScheme.Default(external)
	kc := &kubeletconfig.KubeletConfiguration{}
	err = kubeletScheme.Convert(external, kc, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return kc
}
