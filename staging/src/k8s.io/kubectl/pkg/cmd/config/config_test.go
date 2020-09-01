/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
)

func newRedFederalCowHammerConfig() clientcmdapi.Config {
	return clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"red-user": {Token: "red-token"}},
		Clusters: map[string]*clientcmdapi.Cluster{
			"cow-cluster": {Server: "http://cow.org:8080"}},
		Contexts: map[string]*clientcmdapi.Context{
			"federal-context": {AuthInfo: "red-user", Cluster: "cow-cluster"}},
		CurrentContext: "federal-context",
	}
}

func Example_view() {
	expectedConfig := newRedFederalCowHammerConfig()
	test := configCommandTest{
		args:           []string{"view"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	output := test.run(nil)
	fmt.Printf("%v", output)
	// Output:
	// apiVersion: v1
	// clusters:
	// - cluster:
	//     server: http://cow.org:8080
	//   name: cow-cluster
	// contexts:
	// - context:
	//     cluster: cow-cluster
	//     user: red-user
	//   name: federal-context
	// current-context: federal-context
	// kind: Config
	// preferences: {}
	// users:
	// - name: red-user
	//   user:
	//     token: REDACTED
}

func TestCurrentContext(t *testing.T) {
	startingConfig := newRedFederalCowHammerConfig()
	test := configCommandTest{
		args:            []string{"current-context"},
		startingConfig:  startingConfig,
		expectedConfig:  startingConfig,
		expectedOutputs: []string{startingConfig.CurrentContext},
	}
	test.run(t)
}

func TestSetCurrentContext(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	startingConfig := newRedFederalCowHammerConfig()

	newContextName := "the-new-context"

	startingConfig.Contexts[newContextName] = clientcmdapi.NewContext()
	expectedConfig.Contexts[newContextName] = clientcmdapi.NewContext()

	expectedConfig.CurrentContext = newContextName

	test := configCommandTest{
		args:           []string{"use-context", "the-new-context"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestSetNonExistentContext(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()

	test := configCommandTest{
		args:           []string{"use-context", "non-existent-config"},
		startingConfig: expectedConfig,
		expectedConfig: expectedConfig,
	}

	func() {
		defer func() {
			// Restore cmdutil behavior.
			cmdutil.DefaultBehaviorOnFatal()
		}()

		// Check exit code.
		cmdutil.BehaviorOnFatal(func(e string, code int) {
			if code != 1 {
				t.Errorf("The exit code is %d, expected 1", code)
			}
			expectedOutputs := []string{`no context exists with the name: "non-existent-config"`}
			test.checkOutput(e, expectedOutputs, t)
		})

		test.run(t)
	}()
}

func TestSetIntoExistingStruct(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.AuthInfos["red-user"].Password = "new-path-value" // Fake value for testing.
	test := configCommandTest{
		args:           []string{"set", "users.red-user.password", "new-path-value"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestSetWithPathPrefixIntoExistingStruct(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.Clusters["cow-cluster"].Server = "http://cow.org:8080/foo/baz"
	test := configCommandTest{
		args:           []string{"set", "clusters.cow-cluster.server", "http://cow.org:8080/foo/baz"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)

	dc := clientcmd.NewDefaultClientConfig(expectedConfig, &clientcmd.ConfigOverrides{})
	dcc, err := dc.ClientConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedHost := "http://cow.org:8080/foo/baz"
	if expectedHost != dcc.Host {
		t.Fatalf("expected client.Config.Host = %q instead of %q", expectedHost, dcc.Host)
	}
}

func TestUnsetStruct(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	delete(expectedConfig.AuthInfos, "red-user")
	test := configCommandTest{
		args:           []string{"unset", "users.red-user"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestUnsetField(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.AuthInfos["red-user"] = clientcmdapi.NewAuthInfo()
	test := configCommandTest{
		args:           []string{"unset", "users.red-user.token"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestSetIntoNewStruct(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	cluster := clientcmdapi.NewCluster()
	cluster.Server = "new-server-value"
	expectedConfig.Clusters["big-cluster"] = cluster
	test := configCommandTest{
		args:           []string{"set", "clusters.big-cluster.server", "new-server-value"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestSetBoolean(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	cluster := clientcmdapi.NewCluster()
	cluster.InsecureSkipTLSVerify = true
	expectedConfig.Clusters["big-cluster"] = cluster
	test := configCommandTest{
		args:           []string{"set", "clusters.big-cluster.insecure-skip-tls-verify", "true"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestSetIntoNewConfig(t *testing.T) {
	expectedConfig := *clientcmdapi.NewConfig()
	context := clientcmdapi.NewContext()
	context.AuthInfo = "fake-user"
	expectedConfig.Contexts["new-context"] = context
	test := configCommandTest{
		args:           []string{"set", "contexts.new-context.user", "fake-user"},
		startingConfig: *clientcmdapi.NewConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestNewEmptyAuth(t *testing.T) {
	expectedConfig := *clientcmdapi.NewConfig()
	expectedConfig.AuthInfos["the-user-name"] = clientcmdapi.NewAuthInfo()
	test := configCommandTest{
		args:           []string{"set-credentials", "the-user-name"},
		startingConfig: *clientcmdapi.NewConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestAdditionalAuth(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.Token = "token"
	expectedConfig.AuthInfos["another-user"] = authInfo
	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagBearerToken + "=token"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestEmbedClientCert(t *testing.T) {
	fakeCertFile, _ := ioutil.TempFile(os.TempDir(), "")
	defer os.Remove(fakeCertFile.Name())
	fakeData := []byte("fake-data")
	ioutil.WriteFile(fakeCertFile.Name(), fakeData, 0600)
	expectedConfig := newRedFederalCowHammerConfig()
	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.ClientCertificateData = fakeData
	expectedConfig.AuthInfos["another-user"] = authInfo

	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagCertFile + "=" + fakeCertFile.Name(), "--" + clientcmd.FlagEmbedCerts + "=true"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestEmbedClientKey(t *testing.T) {
	fakeKeyFile, _ := ioutil.TempFile(os.TempDir(), "")
	defer os.Remove(fakeKeyFile.Name())
	fakeData := []byte("fake-data")
	ioutil.WriteFile(fakeKeyFile.Name(), fakeData, 0600)
	expectedConfig := newRedFederalCowHammerConfig()
	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.ClientKeyData = fakeData
	expectedConfig.AuthInfos["another-user"] = authInfo

	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagKeyFile + "=" + fakeKeyFile.Name(), "--" + clientcmd.FlagEmbedCerts + "=true"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestEmbedNoKeyOrCertDisallowed(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagEmbedCerts + "=true"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	func() {
		defer func() {
			// Restore cmdutil behavior.
			cmdutil.DefaultBehaviorOnFatal()
		}()

		// Check exit code.
		cmdutil.BehaviorOnFatal(func(e string, code int) {
			if code != 1 {
				t.Errorf("The exit code is %d, expected 1", code)
			}
			expectedOutputs := []string{"--client-certificate", "--client-key", "embed"}
			test.checkOutput(e, expectedOutputs, t)
		})

		test.run(t)
	}()
}

func TestEmptyTokenAndCertAllowed(t *testing.T) {
	fakeCertFile, _ := ioutil.TempFile(os.TempDir(), "cert-file")
	defer os.Remove(fakeCertFile.Name())
	expectedConfig := newRedFederalCowHammerConfig()
	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.ClientCertificate = path.Base(fakeCertFile.Name())
	expectedConfig.AuthInfos["another-user"] = authInfo

	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagCertFile + "=" + fakeCertFile.Name(), "--" + clientcmd.FlagBearerToken + "="},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestTokenAndCertAllowed(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.Token = "token"
	authInfo.ClientCertificate = "/cert-file"
	expectedConfig.AuthInfos["another-user"] = authInfo
	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagCertFile + "=/cert-file", "--" + clientcmd.FlagBearerToken + "=token"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestTokenAndBasicDisallowed(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagUsername + "=myuser", "--" + clientcmd.FlagBearerToken + "=token"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	func() {
		defer func() {
			// Restore cmdutil behavior.
			cmdutil.DefaultBehaviorOnFatal()
		}()

		// Check exit code.
		cmdutil.BehaviorOnFatal(func(e string, code int) {
			if code != 1 {
				t.Errorf("The exit code is %d, expected 1", code)
			}

			expectedOutputs := []string{"--token", "--username"}
			test.checkOutput(e, expectedOutputs, t)
		})

		test.run(t)
	}()
}

func TestBasicClearsToken(t *testing.T) {
	authInfoWithToken := clientcmdapi.NewAuthInfo()
	authInfoWithToken.Token = "token"

	authInfoWithBasic := clientcmdapi.NewAuthInfo()
	authInfoWithBasic.Username = "myuser"
	authInfoWithBasic.Password = "mypass" // Fake value for testing.

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.AuthInfos["another-user"] = authInfoWithToken

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.AuthInfos["another-user"] = authInfoWithBasic

	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagUsername + "=myuser", "--" + clientcmd.FlagPassword + "=mypass"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestTokenClearsBasic(t *testing.T) {
	authInfoWithBasic := clientcmdapi.NewAuthInfo()
	authInfoWithBasic.Username = "myuser"
	authInfoWithBasic.Password = "mypass" // Fake value for testing.

	authInfoWithToken := clientcmdapi.NewAuthInfo()
	authInfoWithToken.Token = "token"

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.AuthInfos["another-user"] = authInfoWithBasic

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.AuthInfos["another-user"] = authInfoWithToken

	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagBearerToken + "=token"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestTokenLeavesCert(t *testing.T) {
	authInfoWithCerts := clientcmdapi.NewAuthInfo()
	authInfoWithCerts.ClientCertificate = "cert"
	authInfoWithCerts.ClientCertificateData = []byte("certdata")
	authInfoWithCerts.ClientKey = "key"
	authInfoWithCerts.ClientKeyData = []byte("keydata")

	authInfoWithTokenAndCerts := clientcmdapi.NewAuthInfo()
	authInfoWithTokenAndCerts.Token = "token"
	authInfoWithTokenAndCerts.ClientCertificate = "cert"
	authInfoWithTokenAndCerts.ClientCertificateData = []byte("certdata")
	authInfoWithTokenAndCerts.ClientKey = "key"
	authInfoWithTokenAndCerts.ClientKeyData = []byte("keydata")

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.AuthInfos["another-user"] = authInfoWithCerts

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.AuthInfos["another-user"] = authInfoWithTokenAndCerts

	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagBearerToken + "=token"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestCertLeavesToken(t *testing.T) {
	authInfoWithToken := clientcmdapi.NewAuthInfo()
	authInfoWithToken.Token = "token"

	authInfoWithTokenAndCerts := clientcmdapi.NewAuthInfo()
	authInfoWithTokenAndCerts.Token = "token"
	authInfoWithTokenAndCerts.ClientCertificate = "/cert"
	authInfoWithTokenAndCerts.ClientKey = "/key"

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.AuthInfos["another-user"] = authInfoWithToken

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.AuthInfos["another-user"] = authInfoWithTokenAndCerts

	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagCertFile + "=/cert", "--" + clientcmd.FlagKeyFile + "=/key"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestSetBytesBad(t *testing.T) {
	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.Clusters["another-cluster"] = clientcmdapi.NewCluster()

	test := configCommandTest{
		args:           []string{"set", "clusters.another-cluster.certificate-authority-data", "cadata"},
		startingConfig: startingConfig,
		expectedConfig: startingConfig,
	}

	func() {
		defer func() {
			// Restore cmdutil behavior.
			cmdutil.DefaultBehaviorOnFatal()
		}()

		// Check exit code.
		cmdutil.BehaviorOnFatal(func(e string, code int) {
			if code != 1 {
				t.Errorf("The exit code is %d, expected 1", code)
			}
		})

		test.run(t)
	}()
}

func TestSetBytes(t *testing.T) {
	clusterInfoWithCAData := clientcmdapi.NewCluster()
	clusterInfoWithCAData.CertificateAuthorityData = []byte("cadata")

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.Clusters["another-cluster"] = clientcmdapi.NewCluster()

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.Clusters["another-cluster"] = clusterInfoWithCAData

	test := configCommandTest{
		args:           []string{"set", "clusters.another-cluster.certificate-authority-data", "cadata", "--set-raw-bytes"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestSetBase64Bytes(t *testing.T) {
	clusterInfoWithCAData := clientcmdapi.NewCluster()
	clusterInfoWithCAData.CertificateAuthorityData = []byte("cadata")

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.Clusters["another-cluster"] = clientcmdapi.NewCluster()

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.Clusters["another-cluster"] = clusterInfoWithCAData

	test := configCommandTest{
		args:           []string{"set", "clusters.another-cluster.certificate-authority-data", "Y2FkYXRh"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestUnsetBytes(t *testing.T) {
	clusterInfoWithCAData := clientcmdapi.NewCluster()
	clusterInfoWithCAData.CertificateAuthorityData = []byte("cadata")

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.Clusters["another-cluster"] = clusterInfoWithCAData

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.Clusters["another-cluster"] = clientcmdapi.NewCluster()

	test := configCommandTest{
		args:           []string{"unset", "clusters.another-cluster.certificate-authority-data"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestCAClearsInsecure(t *testing.T) {
	fakeCAFile, _ := ioutil.TempFile(os.TempDir(), "ca-file")
	defer os.Remove(fakeCAFile.Name())
	clusterInfoWithInsecure := clientcmdapi.NewCluster()
	clusterInfoWithInsecure.InsecureSkipTLSVerify = true

	clusterInfoWithCA := clientcmdapi.NewCluster()
	clusterInfoWithCA.CertificateAuthority = path.Base(fakeCAFile.Name())

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.Clusters["another-cluster"] = clusterInfoWithInsecure

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.Clusters["another-cluster"] = clusterInfoWithCA

	test := configCommandTest{
		args:           []string{"set-cluster", "another-cluster", "--" + clientcmd.FlagCAFile + "=" + fakeCAFile.Name()},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestCAClearsCAData(t *testing.T) {
	clusterInfoWithCAData := clientcmdapi.NewCluster()
	clusterInfoWithCAData.CertificateAuthorityData = []byte("cadata")

	clusterInfoWithCA := clientcmdapi.NewCluster()
	clusterInfoWithCA.CertificateAuthority = "/cafile"

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.Clusters["another-cluster"] = clusterInfoWithCAData

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.Clusters["another-cluster"] = clusterInfoWithCA

	test := configCommandTest{
		args:           []string{"set-cluster", "another-cluster", "--" + clientcmd.FlagCAFile + "=/cafile", "--" + clientcmd.FlagInsecure + "=false"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestInsecureClearsCA(t *testing.T) {
	clusterInfoWithInsecure := clientcmdapi.NewCluster()
	clusterInfoWithInsecure.InsecureSkipTLSVerify = true

	clusterInfoWithCA := clientcmdapi.NewCluster()
	clusterInfoWithCA.CertificateAuthority = "cafile"
	clusterInfoWithCA.CertificateAuthorityData = []byte("cadata")

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.Clusters["another-cluster"] = clusterInfoWithCA

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.Clusters["another-cluster"] = clusterInfoWithInsecure

	test := configCommandTest{
		args:           []string{"set-cluster", "another-cluster", "--" + clientcmd.FlagInsecure + "=true"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestCADataClearsCA(t *testing.T) {
	fakeCAFile, _ := ioutil.TempFile(os.TempDir(), "")
	defer os.Remove(fakeCAFile.Name())
	fakeData := []byte("cadata")
	ioutil.WriteFile(fakeCAFile.Name(), fakeData, 0600)

	clusterInfoWithCAData := clientcmdapi.NewCluster()
	clusterInfoWithCAData.CertificateAuthorityData = fakeData

	clusterInfoWithCA := clientcmdapi.NewCluster()
	clusterInfoWithCA.CertificateAuthority = "cafile"

	startingConfig := newRedFederalCowHammerConfig()
	startingConfig.Clusters["another-cluster"] = clusterInfoWithCA

	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.Clusters["another-cluster"] = clusterInfoWithCAData

	test := configCommandTest{
		args:           []string{"set-cluster", "another-cluster", "--" + clientcmd.FlagCAFile + "=" + fakeCAFile.Name(), "--" + clientcmd.FlagEmbedCerts + "=true"},
		startingConfig: startingConfig,
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestEmbedNoCADisallowed(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	test := configCommandTest{
		args:           []string{"set-cluster", "another-cluster", "--" + clientcmd.FlagEmbedCerts + "=true"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	func() {
		defer func() {
			// Restore cmdutil behavior.
			cmdutil.DefaultBehaviorOnFatal()
		}()

		// Check exit code.
		cmdutil.BehaviorOnFatal(func(e string, code int) {
			if code != 1 {
				t.Errorf("The exit code is %d, expected 1", code)
			}

			expectedOutputs := []string{"--certificate-authority", "embed"}
			test.checkOutput(e, expectedOutputs, t)
		})

		test.run(t)
	}()
}

func TestCAAndInsecureDisallowed(t *testing.T) {
	test := configCommandTest{
		args:           []string{"set-cluster", "another-cluster", "--" + clientcmd.FlagCAFile + "=cafile", "--" + clientcmd.FlagInsecure + "=true"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: newRedFederalCowHammerConfig(),
	}

	func() {
		defer func() {
			// Restore cmdutil behavior.
			cmdutil.DefaultBehaviorOnFatal()
		}()

		// Check exit code.
		cmdutil.BehaviorOnFatal(func(e string, code int) {
			if code != 1 {
				t.Errorf("The exit code is %d, expected 1", code)
			}

			expectedOutputs := []string{"certificate", "insecure"}
			test.checkOutput(e, expectedOutputs, t)
		})

		test.run(t)
	}()
}

func TestMergeExistingAuth(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	authInfo := expectedConfig.AuthInfos["red-user"]
	authInfo.ClientKey = "/key"
	expectedConfig.AuthInfos["red-user"] = authInfo
	test := configCommandTest{
		args:           []string{"set-credentials", "red-user", "--" + clientcmd.FlagKeyFile + "=/key"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestNewEmptyCluster(t *testing.T) {
	expectedConfig := *clientcmdapi.NewConfig()
	expectedConfig.Clusters["new-cluster"] = clientcmdapi.NewCluster()
	test := configCommandTest{
		args:           []string{"set-cluster", "new-cluster"},
		startingConfig: *clientcmdapi.NewConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestAdditionalCluster(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	cluster := clientcmdapi.NewCluster()
	cluster.CertificateAuthority = "/ca-location"
	cluster.InsecureSkipTLSVerify = false
	cluster.Server = "serverlocation"
	expectedConfig.Clusters["different-cluster"] = cluster
	test := configCommandTest{
		args:           []string{"set-cluster", "different-cluster", "--" + clientcmd.FlagAPIServer + "=serverlocation", "--" + clientcmd.FlagInsecure + "=false", "--" + clientcmd.FlagCAFile + "=/ca-location"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestOverwriteExistingCluster(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	cluster := clientcmdapi.NewCluster()
	cluster.Server = "serverlocation"
	expectedConfig.Clusters["cow-cluster"] = cluster

	test := configCommandTest{
		args:           []string{"set-cluster", "cow-cluster", "--" + clientcmd.FlagAPIServer + "=serverlocation"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestNewEmptyContext(t *testing.T) {
	expectedConfig := *clientcmdapi.NewConfig()
	expectedConfig.Contexts["new-context"] = clientcmdapi.NewContext()
	test := configCommandTest{
		args:           []string{"set-context", "new-context"},
		startingConfig: *clientcmdapi.NewConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestAdditionalContext(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	context := clientcmdapi.NewContext()
	context.Cluster = "some-cluster"
	context.AuthInfo = "some-user"
	context.Namespace = "different-namespace"
	expectedConfig.Contexts["different-context"] = context
	test := configCommandTest{
		args:           []string{"set-context", "different-context", "--" + clientcmd.FlagClusterName + "=some-cluster", "--" + clientcmd.FlagAuthInfoName + "=some-user", "--" + clientcmd.FlagNamespace + "=different-namespace"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestMergeExistingContext(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	context := expectedConfig.Contexts["federal-context"]
	context.Namespace = "hammer"
	expectedConfig.Contexts["federal-context"] = context

	test := configCommandTest{
		args:           []string{"set-context", "federal-context", "--" + clientcmd.FlagNamespace + "=hammer"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestToBool(t *testing.T) {
	type test struct {
		in  string
		out bool
		err string
	}

	tests := []test{
		{"", false, ""},
		{"true", true, ""},
		{"on", false, `strconv.ParseBool: parsing "on": invalid syntax`},
	}

	for _, curr := range tests {
		b, err := toBool(curr.in)
		if (len(curr.err) != 0) && err == nil {
			t.Errorf("Expected error: %v, but got nil", curr.err)
		}
		if (len(curr.err) == 0) && err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if (err != nil) && (err.Error() != curr.err) {
			t.Errorf("Expected %v, got %v", curr.err, err)

		}
		if b != curr.out {
			t.Errorf("Expected %v, got %v", curr.out, b)
		}
	}

}

func testConfigCommand(args []string, startingConfig clientcmdapi.Config, t *testing.T) (string, clientcmdapi.Config) {
	fakeKubeFile, _ := ioutil.TempFile(os.TempDir(), "")
	defer os.Remove(fakeKubeFile.Name())
	err := clientcmd.WriteToFile(startingConfig, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	argsToUse := make([]string, 0, 2+len(args))
	argsToUse = append(argsToUse, "--kubeconfig="+fakeKubeFile.Name())
	argsToUse = append(argsToUse, args...)

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdConfig(cmdutil.NewFactory(genericclioptions.NewTestConfigFlags()), clientcmd.NewDefaultPathOptions(), streams)
	// "context" is a global flag, inherited from base kubectl command in the real world
	cmd.PersistentFlags().String("context", "", "The name of the kubeconfig context to use")
	cmd.SetArgs(argsToUse)
	cmd.Execute()

	config := clientcmd.GetConfigFromFileOrDie(fakeKubeFile.Name())
	return buf.String(), *config
}

type configCommandTest struct {
	args            []string
	startingConfig  clientcmdapi.Config
	expectedConfig  clientcmdapi.Config
	expectedOutputs []string
}

func (test configCommandTest) checkOutput(out string, expectedOutputs []string, t *testing.T) {
	for _, expectedOutput := range expectedOutputs {
		if !strings.Contains(out, expectedOutput) {
			t.Errorf("expected '%s' in output, got '%s'", expectedOutput, out)
		}
	}
}

func (test configCommandTest) run(t *testing.T) string {
	out, actualConfig := testConfigCommand(test.args, test.startingConfig, t)

	testSetNilMapsToEmpties(reflect.ValueOf(&test.expectedConfig))
	testSetNilMapsToEmpties(reflect.ValueOf(&actualConfig))
	testClearLocationOfOrigin(&actualConfig)

	if !apiequality.Semantic.DeepEqual(test.expectedConfig, actualConfig) {
		t.Errorf("diff: %v", diff.ObjectDiff(test.expectedConfig, actualConfig))
		t.Errorf("expected: %#v\n actual:   %#v", test.expectedConfig, actualConfig)
	}

	test.checkOutput(out, test.expectedOutputs, t)

	return out
}
func testClearLocationOfOrigin(config *clientcmdapi.Config) {
	for key, obj := range config.AuthInfos {
		obj.LocationOfOrigin = ""
		config.AuthInfos[key] = obj
	}
	for key, obj := range config.Clusters {
		obj.LocationOfOrigin = ""
		config.Clusters[key] = obj
	}
	for key, obj := range config.Contexts {
		obj.LocationOfOrigin = ""
		config.Contexts[key] = obj
	}
}
func testSetNilMapsToEmpties(curr reflect.Value) {
	actualCurrValue := curr
	if curr.Kind() == reflect.Ptr {
		actualCurrValue = curr.Elem()
	}

	switch actualCurrValue.Kind() {
	case reflect.Map:
		for _, mapKey := range actualCurrValue.MapKeys() {
			currMapValue := actualCurrValue.MapIndex(mapKey)
			testSetNilMapsToEmpties(currMapValue)
		}

	case reflect.Struct:
		for fieldIndex := 0; fieldIndex < actualCurrValue.NumField(); fieldIndex++ {
			currFieldValue := actualCurrValue.Field(fieldIndex)

			if currFieldValue.Kind() == reflect.Map && currFieldValue.IsNil() {
				newValue := reflect.MakeMap(currFieldValue.Type())
				currFieldValue.Set(newValue)
			} else {
				testSetNilMapsToEmpties(currFieldValue.Addr())
			}
		}

	}

}
