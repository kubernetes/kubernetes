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
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
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
readonly DOCKER_REGISTRY="k8s.gcr.io"
readonly ENABLE_LEGACY_ABAC=false
readonly ETC_MANIFESTS=${KUBE_HOME}/etc/kubernetes/manifests
readonly KUBE_API_SERVER_DOCKER_TAG=v1.11.0-alpha.0.1808_3c7452dc11645d-dirty
readonly LOG_OWNER_USER=$(whoami)
readonly LOG_OWNER_GROUP=$(id -gn)
ENCRYPTION_PROVIDER_CONFIG={{.EncryptionProviderConfig}}
ENCRYPTION_PROVIDER_CONFIG_PATH={{.EncryptionProviderConfigPath}}
readonly ETCD_KMS_KEY_ID={{.ETCDKMSKeyID}}
`
	kubeAPIServerManifestFileName = "kube-apiserver.manifest"
	kmsPluginManifestFileName     = "kms-plugin-container.manifest"
	kubeAPIServerStartFuncName    = "start-kube-apiserver"

	// Position of containers within a pod manifest
	kmsPluginContainerIndex        = 0
	apiServerContainerIndexNoKMS   = 0
	apiServerContainerIndexWithKMS = 1

	//	command": [
	//   "/bin/sh", - Index 0
	//   "-c",      - Index 1
	//   "exec /usr/local/bin/kube-apiserver " - Index 2
	execArgsIndex = 2

	socketVolumeMountIndexKMSPlugin = 1
	socketVolumeMountIndexAPIServer = 0
)

type kubeAPIServerEnv struct {
	KubeHome                     string
	EncryptionProviderConfig     string
	EncryptionProviderConfigPath string
	ETCDKMSKeyID                 string
}

type kubeAPIServerManifestTestCase struct {
	*ManifestTestCase
	apiServerContainer v1.Container
	kmsPluginContainer v1.Container
}

func newKubeAPIServerManifestTestCase(t *testing.T) *kubeAPIServerManifestTestCase {
	return &kubeAPIServerManifestTestCase{
		ManifestTestCase: newManifestTestCase(t, kubeAPIServerManifestFileName, kubeAPIServerStartFuncName, []string{kmsPluginManifestFileName}),
	}
}

func (c *kubeAPIServerManifestTestCase) mustLoadContainers() {
	c.mustLoadPodFromManifest()

	switch len(c.pod.Spec.Containers) {
	case 1:
		c.apiServerContainer = c.pod.Spec.Containers[apiServerContainerIndexNoKMS]
	case 2:
		c.apiServerContainer = c.pod.Spec.Containers[apiServerContainerIndexWithKMS]
		c.kmsPluginContainer = c.pod.Spec.Containers[kmsPluginContainerIndex]
	default:
		c.t.Fatalf("got %d containers in apiserver pod, want 1 or 2", len(c.pod.Spec.Containers))
	}
}

func (c *kubeAPIServerManifestTestCase) invokeTest(e kubeAPIServerEnv) {
	c.mustInvokeFunc(deployHelperEnv, e)
	c.mustLoadContainers()
}

func getEncryptionProviderConfigFlag(path string) string {
	return fmt.Sprintf("--experimental-encryption-provider-config=%s", path)
}

func TestEncryptionProviderFlag(t *testing.T) {
	c := newKubeAPIServerManifestTestCase(t)
	defer c.tearDown()

	e := kubeAPIServerEnv{
		KubeHome:                     c.kubeHome,
		EncryptionProviderConfig:     base64.StdEncoding.EncodeToString([]byte("FOO")),
		EncryptionProviderConfigPath: filepath.Join(c.kubeHome, "encryption-provider-config.yaml"),
	}

	c.invokeTest(e)

	expectedFlag := getEncryptionProviderConfigFlag(e.EncryptionProviderConfigPath)
	execArgs := c.apiServerContainer.Command[execArgsIndex]
	if !strings.Contains(execArgs, expectedFlag) {
		c.t.Fatalf("Got %q, wanted the flag to contain %q", execArgs, expectedFlag)
	}
}

func TestEncryptionProviderConfig(t *testing.T) {
	c := newKubeAPIServerManifestTestCase(t)
	defer c.tearDown()

	p := filepath.Join(c.kubeHome, "encryption-provider-config.yaml")
	e := kubeAPIServerEnv{
		KubeHome:                     c.kubeHome,
		EncryptionProviderConfig:     base64.StdEncoding.EncodeToString([]byte("FOO")),
		EncryptionProviderConfigPath: p,
	}

	c.mustInvokeFunc(deployHelperEnv, e)

	if _, err := os.Stat(p); err != nil {
		c.t.Fatalf("Expected encryption provider config to be written to %s, but stat failed with error: %v", p, err)
	}
}

// TestKMSEncryptionProviderConfig asserts that if ETCD_KMS_KEY_ID is set then start-kube-apiserver will produce
// EncryptionProviderConfig file of type KMS and inject experimental-encryption-provider-config startup flag.
func TestKMSEncryptionProviderConfig(t *testing.T) {
	c := newKubeAPIServerManifestTestCase(t)
	defer c.tearDown()

	e := kubeAPIServerEnv{
		KubeHome:                     c.kubeHome,
		EncryptionProviderConfigPath: filepath.Join(c.kubeHome, "encryption-provider-config.yaml"),
		ETCDKMSKeyID:                 "FOO",
	}

	c.invokeTest(e)

	expectedFlag := getEncryptionProviderConfigFlag(e.EncryptionProviderConfigPath)
	execArgs := c.apiServerContainer.Command[execArgsIndex]
	if !strings.Contains(execArgs, expectedFlag) {
		c.t.Fatalf("Got %q, wanted the flag to contain %q", execArgs, expectedFlag)
	}

	p := filepath.Join(c.kubeHome, "encryption-provider-config.yaml")
	if _, err := os.Stat(p); err != nil {
		c.t.Fatalf("Expected encryption provider config to be written to %s, but stat failed with error: %v", p, err)
	}

	d, err := ioutil.ReadFile(p)
	if err != nil {
		c.t.Fatalf("Failed to read encryption provider config %s", p)
	}

	if !strings.Contains(string(d), "name: grpc-kms-provider") {
		c.t.Fatalf("Got %s\n, wanted encryption provider config to be of type grpc-kms", string(d))
	}
}

func TestKMSPluginAndAPIServerSharedVolume(t *testing.T) {
	c := newKubeAPIServerManifestTestCase(t)
	defer c.tearDown()

	var e = kubeAPIServerEnv{
		KubeHome:                     c.kubeHome,
		EncryptionProviderConfigPath: filepath.Join(c.kubeHome, "encryption-provider-config.yaml"),
		ETCDKMSKeyID:                 "FOO",
	}

	c.invokeTest(e)

	k := c.kmsPluginContainer.VolumeMounts[socketVolumeMountIndexKMSPlugin].MountPath
	a := c.apiServerContainer.VolumeMounts[socketVolumeMountIndexAPIServer].MountPath

	if k != a {
		t.Fatalf("Got %s!=%s, wanted KMSPlugin VolumeMount #1:%s to be equal to kube-apiserver VolumeMount #0:%s",
			k, a, k, a)
	}
}
