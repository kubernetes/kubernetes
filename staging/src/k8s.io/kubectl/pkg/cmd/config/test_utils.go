/*
Copyright 2020 The Kubernetes Authors.

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
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

// Write out a fake kube config for testing and return the file object
func generateTestKubeConfig(config clientcmdapi.Config) (*os.File, error) {
	fakeKubeFile, err := ioutil.TempFile(os.TempDir(), "")
	if err != nil {
		return nil, err
	}
	err = clientcmd.WriteToFile(config, fakeKubeFile.Name())
	if err != nil {
		return nil, err
	}
	return fakeKubeFile, nil
}

// Set locations of origin for the provided fake kubefile name so that deep equals works as expected
func completeConfig(conf *clientcmdapi.Config, locOfOrigin string) *clientcmdapi.Config {
	for _, contexts := range conf.Contexts {
		contexts.LocationOfOrigin = locOfOrigin
		contexts.Extensions = make(map[string]runtime.Object)
	}
	for _, cluster := range conf.Clusters {
		cluster.LocationOfOrigin = locOfOrigin
		cluster.Extensions = make(map[string]runtime.Object)
	}
	for _, authInfo := range conf.AuthInfos {
		authInfo.LocationOfOrigin = locOfOrigin
		authInfo.Extensions = make(map[string]runtime.Object)
	}

	return conf
}

// Check if the provided string has a substring and log out a consistently formatted testing error if not
func checkOutputResults(t *testing.T, output, expectedOutput string) {
	if !strings.Contains(output, expectedOutput) {
		t.Fatalf("did not get expected output:\nwant: %v\ngot:  %v", expectedOutput, output)
	}
}

// Check if the provided config is the same as the expected configuration, using the provided comparison options
func checkOutputConfig(t *testing.T, configAccess clientcmd.ConfigAccess, wantConfig *clientcmdapi.Config, cmpOptions cmp.Option) {
	gotConfig, configFileName, err := loadConfig(configAccess)
	if err != nil {
		t.Fatalf("unexpected error loading temp config from file")
	}
	expectedConfig := completeConfig(wantConfig, configFileName)
	if cmp.Diff(expectedConfig, gotConfig, cmpOptions) != "" {
		t.Fatalf("expected config did not match actual config (-want, +got):\n%v", cmp.Diff(expectedConfig, gotConfig, cmpOptions))
	}
}

// Wrapper function to remove a temp file, to be used for defer lines
func removeTempFile(t *testing.T, name string) {
	err := os.Remove(name)
	if err != nil {
		t.Fatalf("unexpected error removing fake kube config file: %v", err)
	}
}
