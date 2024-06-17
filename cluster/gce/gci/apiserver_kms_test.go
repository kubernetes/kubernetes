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
	"bytes"
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
)

const (
	kubeAPIServerManifestFileName = "kube-apiserver.manifest"
	kubeAPIServerConfigScriptName = "configure-kubeapiserver.sh"
	kubeAPIServerStartFuncName    = "start-kube-apiserver"
)

type kubeAPIServerEnv struct {
	KubeHome                     string
	KubeAPIServerRunAsUser       string
	EncryptionProviderConfigPath string
	EncryptionProviderConfig     string
	CloudKMSIntegration          bool
}

func TestEncryptionProviderFlag(t *testing.T) {
	var (
		encryptionConfigFlag = "--encryption-provider-config"
	)

	testCases := []struct {
		desc                     string
		encryptionProviderConfig string
		wantFlag                 bool
	}{
		{
			desc:                     "ENCRYPTION_PROVIDER_CONFIG is set",
			encryptionProviderConfig: base64.StdEncoding.EncodeToString([]byte("foo")),
			wantFlag:                 true,
		},
		{
			desc:                     "ENCRYPTION_PROVIDER_CONFIG is not set",
			encryptionProviderConfig: "",
			wantFlag:                 false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			c := newManifestTestCase(t, kubeAPIServerManifestFileName, kubeAPIServerStartFuncName, nil)
			defer c.tearDown()

			e := kubeAPIServerEnv{
				KubeHome:                     c.kubeHome,
				KubeAPIServerRunAsUser:       strconv.Itoa(os.Getuid()),
				EncryptionProviderConfigPath: filepath.Join(c.kubeHome, "encryption-provider-config.yaml"),
				EncryptionProviderConfig:     tc.encryptionProviderConfig,
			}

			c.mustInvokeFunc(
				e,
				[]string{"configure-helper.sh", kubeAPIServerConfigScriptName},
				"kms.template",
				"testdata/kube-apiserver/base.template",
				"testdata/kube-apiserver/kms.template")
			c.mustLoadPodFromManifest()

			execArgs := strings.Join(c.pod.Spec.Containers[0].Command, " ")
			flagIsInArg := strings.Contains(execArgs, encryptionConfigFlag)
			flag := fmt.Sprintf("%s=%s", encryptionConfigFlag, e.EncryptionProviderConfigPath)

			switch {
			case tc.wantFlag && !flagIsInArg:
				t.Fatalf("Got %q,\n want flags to contain %q", execArgs, flag)
			case !tc.wantFlag && flagIsInArg:
				t.Fatalf("Got %q,\n do not want flags to contain %q", execArgs, encryptionConfigFlag)
			case tc.wantFlag && flagIsInArg && !strings.Contains(execArgs, flag):
				t.Fatalf("Got flags: %q, want it to contain %q", execArgs, flag)
			}
		})
	}
}

func TestEncryptionProviderConfig(t *testing.T) {
	c := newManifestTestCase(t, kubeAPIServerManifestFileName, kubeAPIServerStartFuncName, nil)
	defer c.tearDown()

	p := filepath.Join(c.kubeHome, "encryption-provider-config.yaml")
	e := kubeAPIServerEnv{
		KubeHome:                     c.kubeHome,
		KubeAPIServerRunAsUser:       strconv.Itoa(os.Getuid()),
		EncryptionProviderConfigPath: p,
		EncryptionProviderConfig:     base64.StdEncoding.EncodeToString([]byte("foo")),
	}

	c.mustInvokeFunc(
		e,
		[]string{"configure-helper.sh", kubeAPIServerConfigScriptName},
		"kms.template",

		"testdata/kube-apiserver/base.template",
		"testdata/kube-apiserver/kms.template",
	)

	if _, err := os.Stat(p); err != nil {
		c.t.Fatalf("Expected encryption provider config to be written to %s, but stat failed with error: %v", p, err)
	}

	got, err := os.ReadFile(p)
	if err != nil {
		c.t.Fatalf("Failed to read encryption provider config %s", p)
	}

	want := []byte("foo")
	if !bytes.Equal(got, want) {
		c.t.Fatalf("got encryptionConfig:\n%q\n, want encryptionConfig:\n%q", got, want)
	}
}

func TestKMSIntegration(t *testing.T) {
	var (
		socketPath  = "/var/run/kmsplugin"
		dirOrCreate = v1.HostPathType(v1.HostPathDirectoryOrCreate)
		socketName  = "kmssocket"
	)
	testCases := []struct {
		desc                string
		cloudKMSIntegration bool
		wantVolume          v1.Volume
		wantVolMount        v1.VolumeMount
	}{
		{
			desc:                "CLOUD_KMS_INTEGRATION is set",
			cloudKMSIntegration: true,
			wantVolume: v1.Volume{
				Name: socketName,
				VolumeSource: v1.VolumeSource{
					HostPath: &v1.HostPathVolumeSource{
						Path: socketPath,
						Type: &dirOrCreate,
					},
				},
			},
			wantVolMount: v1.VolumeMount{
				Name:      socketName,
				MountPath: socketPath,
			},
		},
		{
			desc:                "CLOUD_KMS_INTEGRATION is not set",
			cloudKMSIntegration: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			c := newManifestTestCase(t, kubeAPIServerManifestFileName, kubeAPIServerStartFuncName, nil)
			defer c.tearDown()

			var e = kubeAPIServerEnv{
				KubeHome:                     c.kubeHome,
				KubeAPIServerRunAsUser:       strconv.Itoa(os.Getuid()),
				EncryptionProviderConfigPath: filepath.Join(c.kubeHome, "encryption-provider-config.yaml"),
				EncryptionProviderConfig:     base64.StdEncoding.EncodeToString([]byte("foo")),
				CloudKMSIntegration:          tc.cloudKMSIntegration,
			}

			c.mustInvokeFunc(
				e,
				[]string{"configure-helper.sh", kubeAPIServerConfigScriptName},
				"kms.template",

				"testdata/kube-apiserver/base.template",
				"testdata/kube-apiserver/kms.template",
			)
			c.mustLoadPodFromManifest()
			// By this point, we can be sure that kube-apiserver manifest is a valid POD.

			var gotVolume v1.Volume
			for _, v := range c.pod.Spec.Volumes {
				if v.Name == socketName {
					gotVolume = v
					break
				}
			}

			if !reflect.DeepEqual(gotVolume, tc.wantVolume) {
				t.Errorf("got volume %v, want %v", gotVolume, tc.wantVolume)
			}

			var gotVolumeMount v1.VolumeMount
			for _, v := range c.pod.Spec.Containers[0].VolumeMounts {
				if v.Name == socketName {
					gotVolumeMount = v
					break
				}
			}

			if !reflect.DeepEqual(gotVolumeMount, tc.wantVolMount) {
				t.Errorf("got volumeMount %v, want %v", gotVolumeMount, tc.wantVolMount)
			}
		})
	}
}
