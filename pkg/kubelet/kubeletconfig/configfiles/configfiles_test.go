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
	"strings"
	"testing"

	"github.com/davecgh/go-spew/spew"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletscheme "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme"
	kubeletconfigv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1alpha1"
	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

func addFile(fs utilfs.Filesystem, path string, file string) error {
	if err := utilfiles.EnsureDir(fs, filepath.Dir(path)); err != nil {
		return err
	}
	if err := utilfiles.ReplaceFile(fs, path, []byte(file)); err != nil {
		return err
	}
	return nil
}

func TestLoad(t *testing.T) {
	kubeletScheme, _, err := kubeletscheme.NewSchemeAndCodecs()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// get the built-in default configuration
	external := &kubeletconfigv1alpha1.KubeletConfiguration{}
	kubeletScheme.Default(external)
	defaultConfig := &kubeletconfig.KubeletConfiguration{}
	err = kubeletScheme.Convert(external, defaultConfig, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cases := []struct {
		desc   string
		file   string
		expect *kubeletconfig.KubeletConfiguration
		err    string
	}{
		{"empty data", ``, nil, "was empty"},
		// invalid format
		{"invalid yaml", `*`, nil, "failed to decode"},
		{"invalid json", `{*`, nil, "failed to decode"},
		// invalid object
		{"missing kind", `{"apiVersion":"kubeletconfig/v1alpha1"}`, nil, "failed to decode"},
		{"missing version", `{"kind":"KubeletConfiguration"}`, nil, "failed to decode"},
		{"unregistered kind", `{"kind":"BogusKind","apiVersion":"kubeletconfig/v1alpha1"}`, nil, "failed to decode"},
		{"unregistered version", `{"kind":"KubeletConfiguration","apiVersion":"bogusversion"}`, nil, "failed to decode"},
		// empty object with correct kind and version should result in the defaults for that kind and version
		{"default from yaml", `kind: KubeletConfiguration
apiVersion: kubeletconfig/v1alpha1`, defaultConfig, ""},
		{"default from json", `{"kind":"KubeletConfiguration","apiVersion":"kubeletconfig/v1alpha1"}`, defaultConfig, ""},
	}

	fs := utilfs.NewFakeFs()
	for i := range cases {
		dir := fmt.Sprintf("/%d", i)
		if err := addFile(fs, filepath.Join(dir, kubeletFile), cases[i].file); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		loader, err := NewFsLoader(fs, dir)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		kc, err := loader.Load()
		if utiltest.SkipRest(t, cases[i].desc, err, cases[i].err) {
			continue
		}
		// we expect the parsed configuration to match what we described in the ConfigMap
		if !apiequality.Semantic.DeepEqual(cases[i].expect, kc) {
			t.Errorf("case %q, expect config %s but got %s", cases[i].desc, spew.Sdump(cases[i].expect), spew.Sdump(kc))
		}
	}

	// finally test for a missing file
	desc := "missing kubelet file"
	contains := "failed to read"
	loader, err := NewFsLoader(fs, "/fake")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = loader.Load()
	if err == nil {
		t.Errorf("case %q, expect error to contain %q but got nil error", desc, contains)
	} else if !strings.Contains(err.Error(), contains) {
		t.Errorf("case %q, expect error to contain %q but got %q", desc, contains, err.Error())
	}
}
