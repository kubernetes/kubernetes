/*
Copyright 2019 The Kubernetes Authors.

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

package config

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	kubectrlmgrconfigv1alpha1 "k8s.io/kube-controller-manager/config/v1alpha1"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
	kubectrlmgrsscheme "k8s.io/kubernetes/pkg/controller/apis/config/scheme"
	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const (
	configDir   = "/test-cmc-dir"
	ctrlmgrFile = "controller-manager-file"
)

func TestConfigLoad(t *testing.T) {
	var simpleErr = func(err error) bool {
		if err == nil {
			return false
		}
		return true
	}

	testCases := []struct {
		desc   string
		file   *string
		expect *kubectrlmgrconfig.KubeControllerManagerConfiguration
		errFn  func(err error) bool
	}{
		{
			desc:  "missing file",
			errFn: simpleErr,
		},
		{
			desc:  "empty file",
			file:  newString(``),
			errFn: simpleErr,
		},
		{
			desc:  "invalid yaml",
			file:  newString(`*`),
			errFn: simpleErr,
		},
		{
			desc:  "invalid json",
			file:  newString(`{*`),
			errFn: simpleErr,
		},
		{
			desc:  "missing kind",
			file:  newString(`{"apiVersion":"kubecontrollermanager.config.k8s.io.config.k8s.io/v1alpha1"}`),
			errFn: simpleErr,
		},
		{
			desc:  "missing version",
			file:  newString(`{"kind":"KubeControllerManagerConfiguration"}`),
			errFn: simpleErr,
		},
		{
			desc:  "unregistered kind",
			file:  newString(`{"kind":"boguskind","apiVersion":"kubecontrollermanager.config.k8s.io.config.k8s.io/v1alpha1"}`),
			errFn: simpleErr,
		},
		{
			desc:  "unregistered version",
			file:  newString(`{"kind":"KubeControllerManagerConfiguration","apiVersion":"bogusversion"}`),
			errFn: simpleErr,
		},
		{
			desc: "default from yaml",
			file: newString(`kind: KubeControllerManagerConfiguration
apiVersion: kubecontrollermanager.config.k8s.io/v1alpha1`),
			expect: newConfig(t),
		},
		{
			desc:   "default from json",
			file:   newString(`{"kind": "KubeControllerManagerConfiguration", "apiVersion": "kubecontrollermanager.config.k8s.io/v1alpha1"}`),
			expect: newConfig(t),
		},
		{
			desc: "full from yaml",
			file: customConfig(),
			expect: func() *kubectrlmgrconfig.KubeControllerManagerConfiguration {
				conf := newConfig(t)
				conf.Generic.Port = 1234
				conf.Generic.ClientConnection.Burst = 10
				conf.ResourceQuotaController.ConcurrentResourceQuotaSyncs = 15
				return conf
			}(),
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			fs := utilfs.NewFakeFs()
			path := filepath.Join(configDir, ctrlmgrFile)
			if test.file != nil {
				if err := addFile(fs, path, *test.file); err != nil {
					t.Fatalf("unexpected error adding %q: %v", path, err)
				}
			}
			loader, err := NewFsLoader(fs, path)
			if err != nil {
				t.Fatalf("unexpected error creating fsLoader: %v", err)
			}
			cmc, err := loader.Load()
			if err != nil {
				if test.errFn != nil {
					assert.True(t, test.errFn(err), test.desc)
				} else {
					t.Fatalf("unexepected error: %v", err)
				}

			}
			assert.Equal(t, test.expect, cmc, test.desc)
		})
	}
}

func newString(s string) *string {
	return &s
}

func newConfig(t *testing.T) *kubectrlmgrconfig.KubeControllerManagerConfiguration {
	kcmScheme, _, err := kubectrlmgrsscheme.NewSchemeAndCodecs()
	if err != nil {
		t.Fatalf("unexpected error creating kcm scheme: %v", err)
	}
	external := &kubectrlmgrconfigv1alpha1.KubeControllerManagerConfiguration{}
	kcmScheme.Default(external)
	internal := &kubectrlmgrconfig.KubeControllerManagerConfiguration{}
	err = kcmScheme.Convert(external, internal, nil)
	if err != nil {
		t.Fatalf("unexpected error while trying to convert from external to internal kcm config: %v", err)
	}
	return internal
}

func addFile(fs utilfs.Filesystem, path string, file string) error {
	if err := utilfiles.EnsureDir(fs, filepath.Dir(path)); err != nil {
		return err
	}
	return utilfiles.ReplaceFile(fs, path, []byte(file))
}

func customConfig() *string {
	c := `kind: KubeControllerManagerConfiguration
apiVersion: kubecontrollermanager.config.k8s.io/v1alpha1
Generic: 					# GenericControllerConfiguration
  Port: 1234
  Address: 0.0.0.0
  ClientConnection:			# ClientConnectionConfiguration
    burst: 10
ResourceQuotaController:	# ResourceQuotaControllerConfiguration
  ConcurrentResourceQuotaSyncs: 15`
	return &c
}
