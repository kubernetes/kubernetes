/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type deleteAuthInfoTest struct {
	config            clientcmdapi.Config
	authInfoToDelete  string
	expectedAuthInfos []string
	expectedOut       string
}

func TestDeleteAuthInfo(t *testing.T) {
	conf := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"me": {Token: "sdff94380092j2309jf203458hf5io45fj4589"},
			"other": {
				Username: "ivansmil",
				Password: "foobar",
			},
		},
	}
	test := deleteAuthInfoTest{
		config:            conf,
		authInfoToDelete:  "me",
		expectedAuthInfos: []string{"other"},
		expectedOut:       "deleted user me from %s\n",
	}

	test.run(t)
}

func TestDeleteEmptyAuthInfo(t *testing.T) {
	conf := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"me": {Token: "3f2323f23f2c23c23f5asdasdasdj4589"},
		},
	}
	test := deleteAuthInfoTest{
		config:            conf,
		authInfoToDelete:  "me",
		expectedAuthInfos: []string{},
		expectedOut:       "deleted user me from %s\n",
	}

	test.run(t)
}

func (test deleteAuthInfoTest) run(t *testing.T) {
	fakeKubeFile, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(fakeKubeFile.Name())
	err = clientcmd.WriteToFile(test.config, fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	pathOptions := clientcmd.NewDefaultPathOptions()
	pathOptions.GlobalFile = fakeKubeFile.Name()
	pathOptions.EnvVar = ""

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdConfigDeleteAuthInfo(buf, pathOptions)
	cmd.SetArgs([]string{test.authInfoToDelete})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v", err)
	}

	expectedOutWithFile := fmt.Sprintf(test.expectedOut, fakeKubeFile.Name())
	if expectedOutWithFile != buf.String() {
		t.Errorf("expected output %s, but got %s", expectedOutWithFile, buf.String())
		return
	}

	// Verify authInfo was removed from kubeconfig file
	config, err := clientcmd.LoadFromFile(fakeKubeFile.Name())
	if err != nil {
		t.Fatalf("unexpected error loading kubeconfig file: %v", err)
	}

	authInfos := make([]string, 0, len(config.AuthInfos))
	for k := range config.AuthInfos {
		authInfos = append(authInfos, k)
	}

	if !reflect.DeepEqual(test.expectedAuthInfos, authInfos) {
		t.Errorf("expected authInfos %v, but found %v in kubeconfig", test.expectedAuthInfos, authInfos)
	}
}
