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
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
)

const (
	/*
	   Template for defining the environment state of configure-helper.sh
	   The environment of configure-helper.sh is initially configured via kube-env file. However, as deploy-helper
	   executes new variables are created. ManifestTestCase does not care where a variable came from. However, future
	   test scenarios, may require such a distinction.
	   The list of variables is, by no means, complete - this is what is required to run currently defined tests.
	*/
	deployHelperEnv = `
readonly KUBE_HOME={{.KubeHome}}
readonly KUBE_API_SERVER_LOG_PATH=${KUBE_HOME}/kube-apiserver.log
readonly KUBE_API_SERVER_AUDIT_LOG_PATH=${KUBE_HOME}/kube-apiserver-audit.log
readonly CLOUD_CONFIG_OPT=--cloud-config=/etc/gce.conf
readonly CA_CERT_BUNDLE_PATH=/foo/bar
readonly APISERVER_SERVER_CERT_PATH=/foo/bar
readonly APISERVER_SERVER_KEY_PATH=/foo/bar
readonly APISERVER_CLIENT_CERT_PATH=/foo/bar
readonly CLOUD_CONFIG_MOUNT="{\"name\": \"cloudconfigmount\",\"mountPath\": \"/etc/gce.conf\", \"readOnly\": true},"
readonly CLOUD_CONFIG_VOLUME="{\"name\": \"cloudconfigmount\",\"hostPath\": {\"path\": \"/etc/gce.conf\", \"type\": \"FileOrCreate\"}},"
readonly INSECURE_PORT_MAPPING="{ \"name\": \"local\", \"containerPort\": 8080, \"hostPort\": 8080},"
readonly DOCKER_REGISTRY="k8s.gcr.io"
readonly ENABLE_LEGACY_ABAC=false
readonly ETC_MANIFESTS=${KUBE_HOME}/etc/kubernetes/manifests
readonly KUBE_API_SERVER_DOCKER_TAG=v1.11.0-alpha.0.1808_3c7452dc11645d-dirty
readonly LOG_OWNER_USER=$(id -un)
readonly LOG_OWNER_GROUP=$(id -gn)
readonly SERVICEACCOUNT_ISSUER=https://foo.bar.baz
readonly SERVICEACCOUNT_KEY_PATH=/foo/bar/baz.key
{{if .EncryptionProviderConfig}}
ENCRYPTION_PROVIDER_CONFIG={{.EncryptionProviderConfig}}
{{end}}
ENCRYPTION_PROVIDER_CONFIG_PATH={{.EncryptionProviderConfigPath}}
{{if .CloudKMSIntegration}}
readonly CLOUD_KMS_INTEGRATION=true
{{end}}
`
	kubeAPIServerManifestFileName = "kube-apiserver.manifest"
	kubeAPIServerStartFuncName    = "start-kube-apiserver"
)

type kubeAPIServerEnv struct {
	KubeHome                     string
	EncryptionProviderConfigPath string
	EncryptionProviderConfig     string
	CloudKMSIntegration          bool
}

type kubeAPIServerManifestTestCase struct {
	*ManifestTestCase
}

func newKubeAPIServerManifestTestCase(t *testing.T) *kubeAPIServerManifestTestCase {
	return &kubeAPIServerManifestTestCase{
		ManifestTestCase: newManifestTestCase(t, kubeAPIServerManifestFileName, kubeAPIServerStartFuncName, nil),
	}
}

func (c *kubeAPIServerManifestTestCase) invokeTest(e kubeAPIServerEnv, kubeEnv string) {
	c.mustInvokeFunc(kubeEnv, e)
	c.mustLoadPodFromManifest()
}

func TestEncryptionProviderFlag(t *testing.T) {
	var (
		//	command": [
		//   "/bin/sh", - Index 0
		//   "-c",      - Index 1
		//   "exec /usr/local/bin/kube-apiserver " - Index 2
		execArgsIndex        = 2
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
			c := newKubeAPIServerManifestTestCase(t)
			defer c.tearDown()

			e := kubeAPIServerEnv{
				KubeHome:                     c.kubeHome,
				EncryptionProviderConfigPath: filepath.Join(c.kubeHome, "encryption-provider-config.yaml"),
				EncryptionProviderConfig:     tc.encryptionProviderConfig,
			}

			c.invokeTest(e, deployHelperEnv)

			execArgs := c.pod.Spec.Containers[0].Command[execArgsIndex]
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
	c := newKubeAPIServerManifestTestCase(t)
	defer c.tearDown()

	p := filepath.Join(c.kubeHome, "encryption-provider-config.yaml")
	e := kubeAPIServerEnv{
		KubeHome:                     c.kubeHome,
		EncryptionProviderConfigPath: p,
		EncryptionProviderConfig:     base64.StdEncoding.EncodeToString([]byte("foo")),
	}

	c.mustInvokeFunc(deployHelperEnv, e)

	if _, err := os.Stat(p); err != nil {
		c.t.Fatalf("Expected encryption provider config to be written to %s, but stat failed with error: %v", p, err)
	}

	got, err := ioutil.ReadFile(p)
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
			c := newKubeAPIServerManifestTestCase(t)
			defer c.tearDown()

			var e = kubeAPIServerEnv{
				KubeHome:                     c.kubeHome,
				EncryptionProviderConfigPath: filepath.Join(c.kubeHome, "encryption-provider-config.yaml"),
				EncryptionProviderConfig:     base64.StdEncoding.EncodeToString([]byte("foo")),
				CloudKMSIntegration:          tc.cloudKMSIntegration,
			}

			c.invokeTest(e, deployHelperEnv)
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
