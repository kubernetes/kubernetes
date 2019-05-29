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
	"fmt"
	"path/filepath"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletscheme "k8s.io/kubernetes/pkg/kubelet/apis/config/scheme"
	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const configDir = "/test-config-dir"
const relativePath = "relative/path/test"
const kubeletFile = "kubelet"

func TestLoad(t *testing.T) {
	cases := []struct {
		desc   string
		file   *string
		expect *kubeletconfig.KubeletConfiguration
		err    string
	}{
		// missing file
		{
			"missing file",
			nil,
			nil,
			"failed to read",
		},
		// empty file
		{
			"empty file",
			newString(``),
			nil,
			"was empty",
		},
		// invalid format
		{
			"invalid yaml",
			newString(`*`),
			nil,
			"failed to decode",
		},
		{
			"invalid json",
			newString(`{*`),
			nil,
			"failed to decode",
		},
		// invalid object
		{
			"missing kind",
			newString(`{"apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			nil,
			"failed to decode",
		},
		{
			"missing version",
			newString(`{"kind":"KubeletConfiguration"}`),
			nil,
			"failed to decode",
		},
		{
			"unregistered kind",
			newString(`{"kind":"BogusKind","apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			nil,
			"failed to decode",
		},
		{
			"unregistered version",
			newString(`{"kind":"KubeletConfiguration","apiVersion":"bogusversion"}`),
			nil,
			"failed to decode",
		},

		// empty object with correct kind and version should result in the defaults for that kind and version
		{
			"default from yaml",
			newString(`kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1`),
			newConfig(t),
			"",
		},
		{
			"default from json",
			newString(`{"kind":"KubeletConfiguration","apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			newConfig(t),
			"",
		},

		// relative path
		{
			"yaml, relative path is resolved",
			newString(fmt.Sprintf(`kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
staticPodPath: %s`, relativePath)),
			func() *kubeletconfig.KubeletConfiguration {
				kc := newConfig(t)
				kc.StaticPodPath = filepath.Join(configDir, relativePath)
				return kc
			}(),
			"",
		},
		{
			"json, relative path is resolved",
			newString(fmt.Sprintf(`{"kind":"KubeletConfiguration","apiVersion":"kubelet.config.k8s.io/v1beta1","staticPodPath":"%s"}`, relativePath)),
			func() *kubeletconfig.KubeletConfiguration {
				kc := newConfig(t)
				kc.StaticPodPath = filepath.Join(configDir, relativePath)
				return kc
			}(),
			"",
		},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			fs := utilfs.NewFakeFs()
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
			kc, err := loader.Load()
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
		desc   string
		path   string
		expect string
	}{
		{"empty path", "", ""},
		{"absolute path", absolutePath, absolutePath},
		{"relative path", relativePath, filepath.Join(configDir, relativePath)},
	}

	paths := kubeletconfig.KubeletConfigurationPathRefs(newConfig(t))
	if len(paths) == 0 {
		t.Fatalf("requires at least one path field to exist in the KubeletConfiguration type")
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			// set the path, resolve it, and check if it resolved as we would expect
			*(paths[0]) = c.path
			resolveRelativePaths(paths, configDir)
			if *(paths[0]) != c.expect {
				t.Fatalf("expect %s but got %s", c.expect, *(paths[0]))
			}
		})
	}
}

func newString(s string) *string {
	return &s
}

func addFile(fs utilfs.Filesystem, path string, file string) error {
	if err := utilfiles.EnsureDir(fs, filepath.Dir(path)); err != nil {
		return err
	}
	return utilfiles.ReplaceFile(fs, path, []byte(file))
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
