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

package clientcmd

import (
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"testing"

	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var (
	testConfigAlfa = clientcmdapi.Config{
		AuthInfos: map[string]clientcmdapi.AuthInfo{
			"red-user": {Token: "red-token"}},
		Clusters: map[string]clientcmdapi.Cluster{
			"cow-cluster": {Server: "http://cow.org:8080"}},
		Contexts: map[string]clientcmdapi.Context{
			"federal-context": {AuthInfo: "red-user", Cluster: "cow-cluster", Namespace: "hammer-ns"}},
	}
	testConfigBravo = clientcmdapi.Config{
		AuthInfos: map[string]clientcmdapi.AuthInfo{
			"black-user": {Token: "black-token"}},
		Clusters: map[string]clientcmdapi.Cluster{
			"pig-cluster": {Server: "http://pig.org:8080"}},
		Contexts: map[string]clientcmdapi.Context{
			"queen-anne-context": {AuthInfo: "black-user", Cluster: "pig-cluster", Namespace: "saw-ns"}},
	}
)

func init() {
	testSetNilMapsToEmpties(reflect.ValueOf(&testConfigAlfa))
	testSetNilMapsToEmpties(reflect.ValueOf(&testConfigBravo))
}

func TestNoFilesFound(t *testing.T) {
	loadingOrder := ClientConfigLoadingOrder([]string{"this/is/a/fake/file"})

	loadedConfig, filename, err := loadingOrder.Load()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if len(filename) > 0 {
		t.Errorf("Expected %v, got %v", "", filename)
	}

	empty := clientcmdapi.NewConfig()
	if !reflect.DeepEqual(*empty, *loadedConfig) {
		t.Errorf("Values are not equal, diff: %v", util.ObjectGoPrintDiff(empty, loadedConfig))
	}
}

func TestSkipMissingFile(t *testing.T) {
	configFile, _ := ioutil.TempFile("", "")
	defer os.Remove(configFile.Name())

	WriteToFile(testConfigAlfa, configFile.Name())

	loadingOrder := ClientConfigLoadingOrder([]string{"this/is/a/fake/file", configFile.Name()})

	loadedConfig, filename, err := loadingOrder.Load()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if filename != configFile.Name() {
		t.Errorf("Expected %v, got %v", configFile, filename)
	}

	if !reflect.DeepEqual(testConfigAlfa, *loadedConfig) {
		t.Errorf("Values are not equal, diff: %v", util.ObjectGoPrintDiff(testConfigAlfa, *loadedConfig))
	}
}

func TestResolveRelativePaths(t *testing.T) {
	pathResolutionConfig1 := clientcmdapi.Config{
		AuthInfos: map[string]clientcmdapi.AuthInfo{
			"relative-user-1": {ClientCertificate: "relative/client/cert", ClientKey: "../relative/client/key", AuthPath: "../../relative/auth/path"},
			"absolute-user-1": {ClientCertificate: "/absolute/client/cert", ClientKey: "/absolute/client/key", AuthPath: "/absolute/auth/path"},
		},
		Clusters: map[string]clientcmdapi.Cluster{
			"relative-server-1": {CertificateAuthority: "../relative/ca"},
			"absolute-server-1": {CertificateAuthority: "/absolute/ca"},
		},
	}

	configDir1, _ := ioutil.TempDir("", "")
	configFile1 := path.Join(configDir1, ".kubeconfig")
	configDir1, _ = filepath.Abs(configDir1)
	defer os.Remove(configFile1)

	WriteToFile(pathResolutionConfig1, configFile1)

	loadingOrder := ClientConfigLoadingOrder([]string{configFile1})

	loadedConfig, _, err := loadingOrder.Load()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	foundClusterCount := 0
	for key, cluster := range loadedConfig.Clusters {
		if key == "relative-server-1" {
			foundClusterCount++
			matchStringArg(path.Join(configDir1, pathResolutionConfig1.Clusters["relative-server-1"].CertificateAuthority), cluster.CertificateAuthority, t)
		}
		if key == "absolute-server-1" {
			foundClusterCount++
			matchStringArg(pathResolutionConfig1.Clusters["absolute-server-1"].CertificateAuthority, cluster.CertificateAuthority, t)
		}
	}
	if foundClusterCount != 2 {
		t.Errorf("Expected 2 clusters, found %v: %v", foundClusterCount, loadedConfig.Clusters)
	}

	foundAuthInfoCount := 0
	for key, authInfo := range loadedConfig.AuthInfos {
		if key == "relative-user-1" {
			foundAuthInfoCount++
			matchStringArg(path.Join(configDir1, pathResolutionConfig1.AuthInfos["relative-user-1"].ClientCertificate), authInfo.ClientCertificate, t)
			matchStringArg(path.Join(configDir1, pathResolutionConfig1.AuthInfos["relative-user-1"].ClientKey), authInfo.ClientKey, t)
			matchStringArg(path.Join(configDir1, pathResolutionConfig1.AuthInfos["relative-user-1"].AuthPath), authInfo.AuthPath, t)
		}
		if key == "absolute-user-1" {
			foundAuthInfoCount++
			matchStringArg(pathResolutionConfig1.AuthInfos["absolute-user-1"].ClientCertificate, authInfo.ClientCertificate, t)
			matchStringArg(pathResolutionConfig1.AuthInfos["absolute-user-1"].ClientKey, authInfo.ClientKey, t)
			matchStringArg(pathResolutionConfig1.AuthInfos["absolute-user-1"].AuthPath, authInfo.AuthPath, t)
		}
	}
	if foundAuthInfoCount != 2 {
		t.Errorf("Expected 2 users, found %v: %v", foundAuthInfoCount, loadedConfig.AuthInfos)
	}

}

func TestChooseFirstFile(t *testing.T) {
	firstFile, _ := ioutil.TempFile("", "")
	defer os.Remove(firstFile.Name())
	secondFile, _ := ioutil.TempFile("", "")
	defer os.Remove(secondFile.Name())

	WriteToFile(testConfigAlfa, firstFile.Name())
	WriteToFile(testConfigBravo, secondFile.Name())
	loadingOrder := ClientConfigLoadingOrder([]string{firstFile.Name(), secondFile.Name()})

	loadedConfig, filename, err := loadingOrder.Load()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if filename != firstFile.Name() {
		t.Errorf("Expected %v, got %v", firstFile.Name(), filename)
	}

	if !reflect.DeepEqual(testConfigAlfa, *loadedConfig) {
		t.Errorf("Values are not equal, diff: %v", util.ObjectGoPrintDiff(testConfigAlfa, loadedConfig))
	}
}

func TestEmptyFileName(t *testing.T) {
	bravoFile, _ := ioutil.TempFile("", "")
	defer os.Remove(bravoFile.Name())

	WriteToFile(testConfigBravo, bravoFile.Name())

	loadingOrder := ClientConfigLoadingOrder([]string{"", bravoFile.Name()})

	loadedConfig, filename, err := loadingOrder.Load()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if filename != bravoFile.Name() {
		t.Errorf("Expected %v, got %v", bravoFile.Name(), filename)
	}

	if !reflect.DeepEqual(testConfigBravo, *loadedConfig) {
		t.Errorf("Values are not equal, diff: %v", util.ObjectGoPrintDiff(testConfigBravo, loadedConfig))
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
