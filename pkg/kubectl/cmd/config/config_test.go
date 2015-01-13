/*
Copyright 2014 Google Inc. All rights reserved.

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
	"bytes"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func newRedFederalCowHammerConfig() clientcmdapi.Config {
	return clientcmdapi.Config{
		AuthInfos: map[string]clientcmdapi.AuthInfo{
			"red-user": {Token: "red-token"}},
		Clusters: map[string]clientcmdapi.Cluster{
			"cow-cluster": {Server: "http://cow.org:8080"}},
		Contexts: map[string]clientcmdapi.Context{
			"federal-context": {AuthInfo: "red-user", Cluster: "cow-cluster", Namespace: "hammer-ns"}},
	}
}

type configCommandTest struct {
	args           []string
	startingConfig clientcmdapi.Config
	expectedConfig clientcmdapi.Config
}

func TestSetCurrentContext(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	expectedConfig.CurrentContext = "the-new-context"
	test := configCommandTest{
		args:           []string{"use-context", "the-new-context"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestSetIntoExistingStruct(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	a := expectedConfig.AuthInfos["red-user"]
	authInfo := &a
	authInfo.AuthPath = "new-path-value"
	expectedConfig.AuthInfos["red-user"] = *authInfo
	test := configCommandTest{
		args:           []string{"set", "users.red-user.auth-path", "new-path-value"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
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
	expectedConfig.AuthInfos["red-user"] = *clientcmdapi.NewAuthInfo()
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
	expectedConfig.Clusters["big-cluster"] = *cluster
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
	expectedConfig.Clusters["big-cluster"] = *cluster
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
	expectedConfig.Contexts["new-context"] = *context
	test := configCommandTest{
		args:           []string{"set", "contexts.new-context.user", "fake-user"},
		startingConfig: *clientcmdapi.NewConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestNewEmptyAuth(t *testing.T) {
	expectedConfig := *clientcmdapi.NewConfig()
	expectedConfig.AuthInfos["the-user-name"] = *clientcmdapi.NewAuthInfo()
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
	authInfo.AuthPath = "auth-path"
	authInfo.ClientKey = "client-key"
	authInfo.Token = "token"
	expectedConfig.AuthInfos["another-user"] = *authInfo
	test := configCommandTest{
		args:           []string{"set-credentials", "another-user", "--" + clientcmd.FlagAuthPath + "=auth-path", "--" + clientcmd.FlagKeyFile + "=client-key", "--" + clientcmd.FlagBearerToken + "=token"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestOverwriteExistingAuth(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	authInfo := clientcmdapi.NewAuthInfo()
	authInfo.AuthPath = "auth-path"
	expectedConfig.AuthInfos["red-user"] = *authInfo
	test := configCommandTest{
		args:           []string{"set-credentials", "red-user", "--" + clientcmd.FlagAuthPath + "=auth-path"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestNewEmptyCluster(t *testing.T) {
	expectedConfig := *clientcmdapi.NewConfig()
	expectedConfig.Clusters["new-cluster"] = *clientcmdapi.NewCluster()
	test := configCommandTest{
		args:           []string{"set-cluster", "new-cluster"},
		startingConfig: *clientcmdapi.NewConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestAdditionalCluster(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	cluster := *clientcmdapi.NewCluster()
	cluster.APIVersion = "v1beta1"
	cluster.CertificateAuthority = "ca-location"
	cluster.InsecureSkipTLSVerify = true
	cluster.Server = "serverlocation"
	expectedConfig.Clusters["different-cluster"] = cluster
	test := configCommandTest{
		args:           []string{"set-cluster", "different-cluster", "--" + clientcmd.FlagAPIServer + "=serverlocation", "--" + clientcmd.FlagInsecure + "=true", "--" + clientcmd.FlagCAFile + "=ca-location", "--" + clientcmd.FlagAPIVersion + "=v1beta1"},
		startingConfig: newRedFederalCowHammerConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestOverwriteExistingCluster(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	cluster := *clientcmdapi.NewCluster()
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
	expectedConfig.Contexts["new-context"] = *clientcmdapi.NewContext()
	test := configCommandTest{
		args:           []string{"set-context", "new-context"},
		startingConfig: *clientcmdapi.NewConfig(),
		expectedConfig: expectedConfig,
	}

	test.run(t)
}

func TestAdditionalContext(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	context := *clientcmdapi.NewContext()
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

func TestOverwriteExistingContext(t *testing.T) {
	expectedConfig := newRedFederalCowHammerConfig()
	context := *clientcmdapi.NewContext()
	context.Cluster = "clustername"
	expectedConfig.Contexts["federal-context"] = context

	test := configCommandTest{
		args:           []string{"set-context", "federal-context", "--" + clientcmd.FlagClusterName + "=clustername"},
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

func testConfigCommand(args []string, startingConfig clientcmdapi.Config) (string, clientcmdapi.Config) {
	fakeKubeFile, _ := ioutil.TempFile("", "")
	defer os.Remove(fakeKubeFile.Name())
	clientcmd.WriteToFile(startingConfig, fakeKubeFile.Name())

	argsToUse := make([]string, 0, 2+len(args))
	argsToUse = append(argsToUse, "--kubeconfig="+fakeKubeFile.Name())
	argsToUse = append(argsToUse, args...)

	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdConfig(buf)
	cmd.SetArgs(argsToUse)
	cmd.Execute()

	// outBytes, _ := ioutil.ReadFile(fakeKubeFile.Name())
	config := getConfigFromFileOrDie(fakeKubeFile.Name())

	return buf.String(), *config
}

func (test configCommandTest) run(t *testing.T) {
	_, actualConfig := testConfigCommand(test.args, test.startingConfig)

	testSetNilMapsToEmpties(reflect.ValueOf(&test.expectedConfig))
	testSetNilMapsToEmpties(reflect.ValueOf(&actualConfig))

	if !reflect.DeepEqual(test.expectedConfig, actualConfig) {
		t.Errorf("diff: %v", util.ObjectDiff(test.expectedConfig, actualConfig))
		t.Errorf("expected: %#v\n actual:   %#v", test.expectedConfig, actualConfig)
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

			// our maps do not hold pointers to structs, they hold the structs themselves.  This means that MapIndex returns the struct itself
			// That in turn means that they have kinds of type.Struct, which is not a settable type.  Because of this, we need to make new struct of that type
			// copy all the data from the old value into the new value, then take the .addr of the new value to modify it in the next recursion.
			// clear as mud
			modifiableMapValue := reflect.New(currMapValue.Type()).Elem()
			modifiableMapValue.Set(currMapValue)

			if modifiableMapValue.Kind() == reflect.Struct {
				modifiableMapValue = modifiableMapValue.Addr()
			}

			testSetNilMapsToEmpties(modifiableMapValue)
			actualCurrValue.SetMapIndex(mapKey, reflect.Indirect(modifiableMapValue))
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
