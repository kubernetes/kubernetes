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
	"github.com/pkg/errors"

	"path/filepath"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletscheme "k8s.io/kubernetes/pkg/kubelet/apis/config/scheme"
	utilcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"
	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const configDir = "/test-config-dir"
const relativePath = "relative/path/test"
const kubeletFile = "kubelet"
const fakeAddress = "127.0.0.1"

func TestLoadAndLoadInstanceFile(t *testing.T) {
	cases := []struct {
		desc string
		file *string
		err  string
	}{
		{
			desc: "missing file",
			err:  "failed to read",
		},
		{
			desc: "empty file",
			file: newString(``),
			err:  "was empty",
		},
		{
			desc: "invalid yaml",
			file: newString(`*`),
			err:  "failed to decode",
		},
		{
			desc: "invalid json",
			file: newString(`{*`),
			err:  "failed to decode",
		},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			loader, err := loadConfigFile(t, c.file)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			_, err = loader.Load()
			utiltest.ExpectError(t, err, c.err)

			_, err = loader.LoadInstance(&kubeletconfigv1beta1.KubeletInstanceConfiguration{})
			utiltest.ExpectError(t, err, c.err)
		})
	}
}

func TestLoad(t *testing.T) {
	cases := []struct {
		desc      string
		file      *string
		expect    *kubeletconfig.KubeletConfiguration
		err       string
		strictErr bool
	}{
		{
			desc: "missing kind",
			file: newString(`{"apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			err:  "failed to decode",
		},
		{
			desc: "missing version",
			file: newString(`{"kind":"KubeletConfiguration"}`),
			err:  "failed to decode",
		},
		{
			desc: "unregistered kind",
			file: newString(`{"kind":"BogusKind","apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			err:  "failed to decode",
		},
		{
			desc: "unregistered version",
			file: newString(`{"kind":"KubeletConfiguration","apiVersion":"bogusversion"}`),
			err:  "failed to decode",
		},
		// empty object with correct kind and version should result in the defaults for that kind and version
		{
			desc: "default from yaml",
			file: newString(`kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1`),
			expect: newConfig(t),
		},
		{
			desc:   "default from json",
			file:   newString(`{"kind":"KubeletConfiguration","apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			expect: newConfig(t),
		},
		// relative path
		{
			desc: "yaml, relative path is resolved",
			file: newString(fmt.Sprintf(`kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
staticPodPath: %s`, filepath.Join(configDir, relativePath))),
			expect: func() *kubeletconfig.KubeletConfiguration {
				kc := newConfig(t)
				kc.StaticPodPath = filepath.Join(configDir, relativePath)
				return kc
			}(),
		},
		{
			desc: "json, relative path is resolved",
			file: newString(fmt.Sprintf(`{"kind":"KubeletConfiguration","apiVersion":"kubelet.config.k8s.io/v1beta1","staticPodPath":"%s"}`, filepath.Join(configDir, relativePath))),
			expect: func() *kubeletconfig.KubeletConfiguration {
				kc := newConfig(t)
				kc.StaticPodPath = filepath.Join(configDir, relativePath)
				return kc
			}(),
		},
		{
			// This should fail from v1beta2+
			desc: "duplicate field",
			file: newString(fmt.Sprintf(`kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
staticPodPath: %s
staticPodPath: %s/foo`, filepath.Join(configDir, relativePath), filepath.Join(configDir, relativePath))),
			// err:       `key "staticPodPath" already set`,
			// strictErr: true,
			expect: func() *kubeletconfig.KubeletConfiguration {
				kc := newConfig(t)
				kc.StaticPodPath = filepath.Join(configDir, relativePath, "foo")
				return kc
			}(),
		},
		{
			// This should fail from v1beta2+
			desc: "unknown field",
			file: newString(`kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
foo: bar`),
			// err:       "found unknown field: foo",
			// strictErr: true,
			expect: newConfig(t),
		},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			loader, err := loadConfigFile(t, c.file)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			kc, err := loader.Load()
			if c.strictErr && !runtime.IsStrictDecodingError(errors.Cause(err)) {
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

func TestLoadInstance(t *testing.T) {
	cases := []struct {
		desc      string
		file      *string
		expect    *kubeletconfigv1beta1.KubeletInstanceConfiguration
		err       string
		strictErr bool
	}{
		{
			desc: "unregistered kind",
			file: newString(`{"kind":"BogusKind","apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			err:  "failed to decode",
		},
		{
			desc: "unregistered version",
			file: newString(`{"kind":"KubeletInstanceConfiguration","apiVersion":"bogusversion"}`),
			err:  "failed to decode",
		},
		{
			desc: "empty correct object with default from yaml",
			file: newString(`kind: KubeletInstanceConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1`),
			expect: newInstanceConfig(t),
		},
		{
			desc:   "default from json",
			file:   newString(`{"kind":"KubeletInstanceConfiguration","apiVersion":"kubelet.config.k8s.io/v1beta1"}`),
			expect: newInstanceConfig(t),
		},
		{
			desc: "yaml, relative path is resolved",
			file: newString(fmt.Sprintf(`kind: KubeletInstanceConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
address: %s`, fakeAddress)),
			expect: func() *kubeletconfigv1beta1.KubeletInstanceConfiguration {
				kic := newInstanceConfig(t)
				kic.Address = fakeAddress
				return kic
			}(),
		},
		{
			desc: "json, relative path is resolved",
			file: newString(fmt.Sprintf(`{"kind":"KubeletInstanceConfiguration","apiVersion":"kubelet.config.k8s.io/v1beta1","address": "%s"}`, fakeAddress)),
			expect: func() *kubeletconfigv1beta1.KubeletInstanceConfiguration {
				kic := newInstanceConfig(t)
				kic.Address = fakeAddress
				return kic
			}(),
		},
		{
			// This should fail from v1beta2+
			desc: "duplicate field",
			file: newString(fmt.Sprintf(`kind: KubeletInstanceConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
address: 0.0.0.0
address: %s`, fakeAddress)),
			err: `key "address" already set`,
			expect: func() *kubeletconfigv1beta1.KubeletInstanceConfiguration {
				kic := newInstanceConfig(t)
				kic.Address = fakeAddress
				return kic
			}(),
		},
		{
			desc: "unknown field",
			file: newString(`kind: KubeletInstanceConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
foo: bar`),
			err:    "found unknown field: foo",
			expect: newInstanceConfig(t),
		},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			loader, err := loadConfigFile(t, c.file)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			defaultConfig := kubeletconfigv1beta1.KubeletInstanceConfiguration{}
			kic, err := loader.LoadInstance(&defaultConfig)
			if c.strictErr && !runtime.IsStrictDecodingError(errors.Cause(err)) {
				t.Fatalf("got error: %v, want strict decoding error", err)
			}

			if utiltest.SkipRest(t, c.desc, err, c.err) {
				return
			}
			if !apiequality.Semantic.DeepEqual(c.expect, kic) {
				t.Fatalf("expect %#v but got %#v", *c.expect, *kic)
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

	kcE, _ := utilcodec.NewKubeletConfigurationE()
	kc, _ := utilcodec.ConvertKubeletConfigurationI(kcE)
	paths := kubeletconfig.KubeletConfigurationPathRefs(kc)

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

func loadConfigFile(t *testing.T, file *string) (Loader, error) {
	fs := utilfs.NewFakeFs()
	path := filepath.Join(configDir, kubeletFile)
	if file != nil {
		if err := addFile(fs, path, *file); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	loader, err := NewFsLoader(fs, path)
	if err != nil {
		return nil, err

	}
	return loader, nil
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

func newInstanceConfig(t *testing.T) *kubeletconfigv1beta1.KubeletInstanceConfiguration {
	kind := "KubeletInstanceConfiguration"
	kcE, err := utilcodec.NewKubeletInstanceConfigurationE()
	kcE.TypeMeta = metav1.TypeMeta{APIVersion: kubeletconfigv1beta1.SchemeGroupVersion.String(), Kind: kind}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	return kcE
}
