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

package clientcmd

import (
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/imdario/mergo"

	restclient "k8s.io/client-go/rest"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func TestMergoSemantics(t *testing.T) {
	type U struct {
		A string
		B int64
	}
	type T struct {
		S []string
		X string
		Y int64
		U U
	}
	var testDataStruct = []struct {
		dst      T
		src      T
		expected T
	}{
		{
			dst:      T{X: "one"},
			src:      T{X: "two"},
			expected: T{X: "two"},
		},
		{
			dst:      T{X: "one", Y: 5, U: U{A: "four", B: 6}},
			src:      T{X: "two", U: U{A: "three", B: 4}},
			expected: T{X: "two", Y: 5, U: U{A: "three", B: 4}},
		},
		{
			dst:      T{S: []string{"test3", "test4", "test5"}},
			src:      T{S: []string{"test1", "test2", "test3"}},
			expected: T{S: []string{"test1", "test2", "test3"}},
		},
	}
	for _, data := range testDataStruct {
		err := mergo.MergeWithOverwrite(&data.dst, &data.src)
		if err != nil {
			t.Errorf("error while merging: %s", err)
		}
		if !reflect.DeepEqual(data.dst, data.expected) {
			// The mergo library has previously changed in a an incompatible way.
			// example:
			//
			//   https://github.com/imdario/mergo/commit/d304790b2ed594794496464fadd89d2bb266600a
			//
			// This test verifies that the semantics of the merge are what we expect.
			// If they are not, the mergo library may have been updated and broken
			// unexpectedly.
			t.Errorf("mergo.MergeWithOverwrite did not provide expected output: %+v doesn't match %+v", data.dst, data.expected)
		}
	}

	var testDataMap = []struct {
		dst      map[string]int
		src      map[string]int
		expected map[string]int
	}{
		{
			dst:      map[string]int{"rsc": 6543, "r": 2138, "gri": 1908, "adg": 912, "prt": 22},
			src:      map[string]int{"rsc": 3711, "r": 2138, "gri": 1908, "adg": 912},
			expected: map[string]int{"rsc": 3711, "r": 2138, "gri": 1908, "adg": 912, "prt": 22},
		},
	}
	for _, data := range testDataMap {
		err := mergo.MergeWithOverwrite(&data.dst, &data.src)
		if err != nil {
			t.Errorf("error while merging: %s", err)
		}
		if !reflect.DeepEqual(data.dst, data.expected) {
			// The mergo library has previously changed in a an incompatible way.
			// example:
			//
			//   https://github.com/imdario/mergo/commit/d304790b2ed594794496464fadd89d2bb266600a
			//
			// This test verifies that the semantics of the merge are what we expect.
			// If they are not, the mergo library may have been updated and broken
			// unexpectedly.
			t.Errorf("mergo.MergeWithOverwrite did not provide expected output: %+v doesn't match %+v", data.dst, data.expected)
		}
	}
}

func createValidTestConfig() *clientcmdapi.Config {
	const (
		server = "https://anything.com:8080"
		token  = "the-token"
	)

	config := clientcmdapi.NewConfig()
	config.Clusters["clean"] = &clientcmdapi.Cluster{
		Server: server,
	}
	config.AuthInfos["clean"] = &clientcmdapi.AuthInfo{
		Token: token,
	}
	config.Contexts["clean"] = &clientcmdapi.Context{
		Cluster:  "clean",
		AuthInfo: "clean",
	}
	config.CurrentContext = "clean"

	return config
}

func createCAValidTestConfig() *clientcmdapi.Config {

	config := createValidTestConfig()
	config.Clusters["clean"].CertificateAuthorityData = []byte{0, 0}
	return config
}

func TestInsecureOverridesCA(t *testing.T) {
	config := createCAValidTestConfig()
	clientBuilder := NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{
		ClusterInfo: clientcmdapi.Cluster{
			InsecureSkipTLSVerify: true,
		},
	}, nil)

	actualCfg, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	matchBoolArg(true, actualCfg.Insecure, t)
	matchStringArg("", actualCfg.TLSClientConfig.CAFile, t)
	matchByteArg(nil, actualCfg.TLSClientConfig.CAData, t)
}

func TestCAOverridesCAData(t *testing.T) {
	file, err := ioutil.TempFile("", "my.ca")
	if err != nil {
		t.Fatalf("could not create tempfile: %v", err)
	}
	defer os.Remove(file.Name())

	config := createCAValidTestConfig()
	clientBuilder := NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{
		ClusterInfo: clientcmdapi.Cluster{
			CertificateAuthority: file.Name(),
		},
	}, nil)

	actualCfg, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	matchBoolArg(false, actualCfg.Insecure, t)
	matchStringArg(file.Name(), actualCfg.TLSClientConfig.CAFile, t)
	matchByteArg(nil, actualCfg.TLSClientConfig.CAData, t)
}

func TestMergeContext(t *testing.T) {
	const namespace = "overridden-namespace"

	config := createValidTestConfig()
	clientBuilder := NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{}, nil)

	_, overridden, err := clientBuilder.Namespace()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if overridden {
		t.Error("Expected namespace to not be overridden")
	}

	clientBuilder = NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{
		Context: clientcmdapi.Context{
			Namespace: namespace,
		},
	}, nil)

	actual, overridden, err := clientBuilder.Namespace()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !overridden {
		t.Error("Expected namespace to be overridden")
	}

	matchStringArg(namespace, actual, t)
}

func TestModifyContext(t *testing.T) {
	expectedCtx := map[string]bool{
		"updated": true,
		"clean":   true,
	}

	tempPath, err := ioutil.TempFile("", "testclientcmd-")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(tempPath.Name())

	pathOptions := NewDefaultPathOptions()
	config := createValidTestConfig()

	pathOptions.GlobalFile = tempPath.Name()

	// define new context and assign it - our path options config
	config.Contexts["updated"] = &clientcmdapi.Context{
		Cluster:  "updated",
		AuthInfo: "updated",
	}
	config.CurrentContext = "updated"

	if err := ModifyConfig(pathOptions, *config, true); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	startingConfig, err := pathOptions.GetStartingConfig()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// make sure the current context was updated
	matchStringArg("updated", startingConfig.CurrentContext, t)

	// there should now be two contexts
	if len(startingConfig.Contexts) != len(expectedCtx) {
		t.Fatalf("unexpected nuber of contexts, expecting %v, but found %v", len(expectedCtx), len(startingConfig.Contexts))
	}

	for key := range startingConfig.Contexts {
		if !expectedCtx[key] {
			t.Fatalf("expected context %q to exist", key)
		}
	}
}

func TestCertificateData(t *testing.T) {
	caData := []byte("ca-data")
	certData := []byte("cert-data")
	keyData := []byte("key-data")

	config := clientcmdapi.NewConfig()
	config.Clusters["clean"] = &clientcmdapi.Cluster{
		Server:                   "https://localhost:8443",
		CertificateAuthorityData: caData,
	}
	config.AuthInfos["clean"] = &clientcmdapi.AuthInfo{
		ClientCertificateData: certData,
		ClientKeyData:         keyData,
	}
	config.Contexts["clean"] = &clientcmdapi.Context{
		Cluster:  "clean",
		AuthInfo: "clean",
	}
	config.CurrentContext = "clean"

	clientBuilder := NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{}, nil)

	clientConfig, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Make sure cert data gets into config (will override file paths)
	matchByteArg(caData, clientConfig.TLSClientConfig.CAData, t)
	matchByteArg(certData, clientConfig.TLSClientConfig.CertData, t)
	matchByteArg(keyData, clientConfig.TLSClientConfig.KeyData, t)
}

func TestBasicAuthData(t *testing.T) {
	username := "myuser"
	password := "mypass"

	config := clientcmdapi.NewConfig()
	config.Clusters["clean"] = &clientcmdapi.Cluster{
		Server: "https://localhost:8443",
	}
	config.AuthInfos["clean"] = &clientcmdapi.AuthInfo{
		Username: username,
		Password: password,
	}
	config.Contexts["clean"] = &clientcmdapi.Context{
		Cluster:  "clean",
		AuthInfo: "clean",
	}
	config.CurrentContext = "clean"

	clientBuilder := NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{}, nil)

	clientConfig, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Make sure basic auth data gets into config
	matchStringArg(username, clientConfig.Username, t)
	matchStringArg(password, clientConfig.Password, t)
}

func TestBasicTokenFile(t *testing.T) {
	token := "exampletoken"
	f, err := ioutil.TempFile("", "tokenfile")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
		return
	}
	defer os.Remove(f.Name())
	if err := ioutil.WriteFile(f.Name(), []byte(token), 0644); err != nil {
		t.Errorf("Unexpected error: %v", err)
		return
	}

	config := clientcmdapi.NewConfig()
	config.Clusters["clean"] = &clientcmdapi.Cluster{
		Server: "https://localhost:8443",
	}
	config.AuthInfos["clean"] = &clientcmdapi.AuthInfo{
		TokenFile: f.Name(),
	}
	config.Contexts["clean"] = &clientcmdapi.Context{
		Cluster:  "clean",
		AuthInfo: "clean",
	}
	config.CurrentContext = "clean"

	clientBuilder := NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{}, nil)

	clientConfig, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	matchStringArg(token, clientConfig.BearerToken, t)
}

func TestPrecedenceTokenFile(t *testing.T) {
	token := "exampletoken"
	f, err := ioutil.TempFile("", "tokenfile")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
		return
	}
	defer os.Remove(f.Name())
	if err := ioutil.WriteFile(f.Name(), []byte(token), 0644); err != nil {
		t.Errorf("Unexpected error: %v", err)
		return
	}

	config := clientcmdapi.NewConfig()
	config.Clusters["clean"] = &clientcmdapi.Cluster{
		Server: "https://localhost:8443",
	}
	expectedToken := "expected"
	config.AuthInfos["clean"] = &clientcmdapi.AuthInfo{
		Token:     expectedToken,
		TokenFile: f.Name(),
	}
	config.Contexts["clean"] = &clientcmdapi.Context{
		Cluster:  "clean",
		AuthInfo: "clean",
	}
	config.CurrentContext = "clean"

	clientBuilder := NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{}, nil)

	clientConfig, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	matchStringArg(expectedToken, clientConfig.BearerToken, t)
}

func TestCreateClean(t *testing.T) {
	config := createValidTestConfig()
	clientBuilder := NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{}, nil)

	clientConfig, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	matchStringArg(config.Clusters["clean"].Server, clientConfig.Host, t)
	matchStringArg("", clientConfig.APIPath, t)
	matchBoolArg(config.Clusters["clean"].InsecureSkipTLSVerify, clientConfig.Insecure, t)
	matchStringArg(config.AuthInfos["clean"].Token, clientConfig.BearerToken, t)
}

func TestCreateCleanWithPrefix(t *testing.T) {
	tt := []struct {
		server string
		host   string
	}{
		{"https://anything.com:8080/foo/bar", "https://anything.com:8080/foo/bar"},
		{"http://anything.com:8080/foo/bar", "http://anything.com:8080/foo/bar"},
		{"http://anything.com:8080/foo/bar/", "http://anything.com:8080/foo/bar/"},
		{"http://anything.com:8080/", "http://anything.com:8080/"},
		{"http://anything.com:8080//", "http://anything.com:8080//"},
		{"anything.com:8080/foo/bar", "anything.com:8080/foo/bar"},
		{"anything.com:8080", "anything.com:8080"},
		{"anything.com", "anything.com"},
		{"anything", "anything"},
	}

	tt = append(tt, struct{ server, host string }{"", "http://localhost:8080"})

	for _, tc := range tt {
		config := createValidTestConfig()

		cleanConfig := config.Clusters["clean"]
		cleanConfig.Server = tc.server
		config.Clusters["clean"] = cleanConfig

		clientBuilder := NewNonInteractiveClientConfig(*config, "clean", &ConfigOverrides{
			ClusterDefaults: clientcmdapi.Cluster{Server: "http://localhost:8080"},
		}, nil)

		clientConfig, err := clientBuilder.ClientConfig()
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		matchStringArg(tc.host, clientConfig.Host, t)
	}
}

func TestCreateCleanDefault(t *testing.T) {
	config := createValidTestConfig()
	clientBuilder := NewDefaultClientConfig(*config, &ConfigOverrides{})

	clientConfig, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	matchStringArg(config.Clusters["clean"].Server, clientConfig.Host, t)
	matchBoolArg(config.Clusters["clean"].InsecureSkipTLSVerify, clientConfig.Insecure, t)
	matchStringArg(config.AuthInfos["clean"].Token, clientConfig.BearerToken, t)
}

func TestCreateCleanDefaultCluster(t *testing.T) {
	config := createValidTestConfig()
	clientBuilder := NewDefaultClientConfig(*config, &ConfigOverrides{
		ClusterDefaults: clientcmdapi.Cluster{Server: "http://localhost:8080"},
	})

	clientConfig, err := clientBuilder.ClientConfig()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	matchStringArg(config.Clusters["clean"].Server, clientConfig.Host, t)
	matchBoolArg(config.Clusters["clean"].InsecureSkipTLSVerify, clientConfig.Insecure, t)
	matchStringArg(config.AuthInfos["clean"].Token, clientConfig.BearerToken, t)
}

func TestCreateMissingContextNoDefault(t *testing.T) {
	config := createValidTestConfig()
	clientBuilder := NewNonInteractiveClientConfig(*config, "not-present", &ConfigOverrides{}, nil)

	_, err := clientBuilder.ClientConfig()
	if err == nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestCreateMissingContext(t *testing.T) {
	const expectedErrorContains = "context was not found for specified context: not-present"
	config := createValidTestConfig()
	clientBuilder := NewNonInteractiveClientConfig(*config, "not-present", &ConfigOverrides{
		ClusterDefaults: clientcmdapi.Cluster{Server: "http://localhost:8080"},
	}, nil)

	_, err := clientBuilder.ClientConfig()
	if err == nil {
		t.Fatalf("Expected error: %v", expectedErrorContains)
	}
	if !strings.Contains(err.Error(), expectedErrorContains) {
		t.Fatalf("Expected error: %v, but got %v", expectedErrorContains, err)
	}
}

func TestInClusterClientConfigPrecedence(t *testing.T) {
	tt := []struct {
		overrides *ConfigOverrides
	}{
		{
			overrides: &ConfigOverrides{
				ClusterInfo: clientcmdapi.Cluster{
					Server: "https://host-from-overrides.com",
				},
			},
		},
		{
			overrides: &ConfigOverrides{
				AuthInfo: clientcmdapi.AuthInfo{
					Token: "https://host-from-overrides.com",
				},
			},
		},
		{
			overrides: &ConfigOverrides{
				ClusterInfo: clientcmdapi.Cluster{
					CertificateAuthority: "/path/to/ca-from-overrides.crt",
				},
			},
		},
		{
			overrides: &ConfigOverrides{
				ClusterInfo: clientcmdapi.Cluster{
					Server: "https://host-from-overrides.com",
				},
				AuthInfo: clientcmdapi.AuthInfo{
					Token: "https://host-from-overrides.com",
				},
			},
		},
		{
			overrides: &ConfigOverrides{
				ClusterInfo: clientcmdapi.Cluster{
					Server:               "https://host-from-overrides.com",
					CertificateAuthority: "/path/to/ca-from-overrides.crt",
				},
			},
		},
		{
			overrides: &ConfigOverrides{
				ClusterInfo: clientcmdapi.Cluster{
					CertificateAuthority: "/path/to/ca-from-overrides.crt",
				},
				AuthInfo: clientcmdapi.AuthInfo{
					Token: "https://host-from-overrides.com",
				},
			},
		},
		{
			overrides: &ConfigOverrides{
				ClusterInfo: clientcmdapi.Cluster{
					Server:               "https://host-from-overrides.com",
					CertificateAuthority: "/path/to/ca-from-overrides.crt",
				},
				AuthInfo: clientcmdapi.AuthInfo{
					Token: "https://host-from-overrides.com",
				},
			},
		},
		{
			overrides: &ConfigOverrides{
				ClusterInfo: clientcmdapi.Cluster{
					Server:               "https://host-from-overrides.com",
					CertificateAuthority: "/path/to/ca-from-overrides.crt",
				},
				AuthInfo: clientcmdapi.AuthInfo{
					Token:     "token-from-override",
					TokenFile: "tokenfile-from-override",
				},
			},
		},
		{
			overrides: &ConfigOverrides{
				ClusterInfo: clientcmdapi.Cluster{
					Server:               "https://host-from-overrides.com",
					CertificateAuthority: "/path/to/ca-from-overrides.crt",
				},
				AuthInfo: clientcmdapi.AuthInfo{
					Token:     "",
					TokenFile: "tokenfile-from-override",
				},
			},
		},
		{
			overrides: &ConfigOverrides{},
		},
	}

	for _, tc := range tt {
		expectedServer := "https://host-from-cluster.com"
		expectedToken := "token-from-cluster"
		expectedTokenFile := "tokenfile-from-cluster"
		expectedCAFile := "/path/to/ca-from-cluster.crt"

		icc := &inClusterClientConfig{
			inClusterConfigProvider: func() (*restclient.Config, error) {
				return &restclient.Config{
					Host:            expectedServer,
					BearerToken:     expectedToken,
					BearerTokenFile: expectedTokenFile,
					TLSClientConfig: restclient.TLSClientConfig{
						CAFile: expectedCAFile,
					},
				}, nil
			},
			overrides: tc.overrides,
		}

		clientConfig, err := icc.ClientConfig()
		if err != nil {
			t.Fatalf("Unxpected error: %v", err)
		}

		if overridenServer := tc.overrides.ClusterInfo.Server; len(overridenServer) > 0 {
			expectedServer = overridenServer
		}
		if len(tc.overrides.AuthInfo.Token) > 0 || len(tc.overrides.AuthInfo.TokenFile) > 0 {
			expectedToken = tc.overrides.AuthInfo.Token
			expectedTokenFile = tc.overrides.AuthInfo.TokenFile
		}
		if overridenCAFile := tc.overrides.ClusterInfo.CertificateAuthority; len(overridenCAFile) > 0 {
			expectedCAFile = overridenCAFile
		}

		if clientConfig.Host != expectedServer {
			t.Errorf("Expected server %v, got %v", expectedServer, clientConfig.Host)
		}
		if clientConfig.BearerToken != expectedToken {
			t.Errorf("Expected token %v, got %v", expectedToken, clientConfig.BearerToken)
		}
		if clientConfig.BearerTokenFile != expectedTokenFile {
			t.Errorf("Expected tokenfile %v, got %v", expectedTokenFile, clientConfig.BearerTokenFile)
		}
		if clientConfig.TLSClientConfig.CAFile != expectedCAFile {
			t.Errorf("Expected Certificate Authority %v, got %v", expectedCAFile, clientConfig.TLSClientConfig.CAFile)
		}
	}
}

func matchBoolArg(expected, got bool, t *testing.T) {
	if expected != got {
		t.Errorf("Expected %v, got %v", expected, got)
	}
}

func matchStringArg(expected, got string, t *testing.T) {
	if expected != got {
		t.Errorf("Expected %q, got %q", expected, got)
	}
}

func matchByteArg(expected, got []byte, t *testing.T) {
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("Expected %v, got %v", expected, got)
	}
}

func TestNamespaceOverride(t *testing.T) {
	config := &DirectClientConfig{
		overrides: &ConfigOverrides{
			Context: clientcmdapi.Context{
				Namespace: "foo",
			},
		},
	}

	ns, overridden, err := config.Namespace()

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !overridden {
		t.Errorf("Expected overridden = true")
	}

	matchStringArg("foo", ns, t)
}

func TestAuthConfigMerge(t *testing.T) {
	content := `
apiVersion: v1
clusters:
- cluster:
    server: https://localhost:8080
  name: foo-cluster
contexts:
- context:
    cluster: foo-cluster
    user: foo-user
    namespace: bar
  name: foo-context
current-context: foo-context
kind: Config
users:
- name: foo-user
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1alpha1
      args:
      - arg-1
      - arg-2
      command: foo-command
`
	tmpfile, err := ioutil.TempFile("", "kubeconfig")
	if err != nil {
		t.Error(err)
	}
	defer os.Remove(tmpfile.Name())
	if err := ioutil.WriteFile(tmpfile.Name(), []byte(content), 0666); err != nil {
		t.Error(err)
	}
	config, err := BuildConfigFromFlags("", tmpfile.Name())
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(config.ExecProvider.Args, []string{"arg-1", "arg-2"}) {
		t.Errorf("Got args %v when they should be %v\n", config.ExecProvider.Args, []string{"arg-1", "arg-2"})
	}

}
