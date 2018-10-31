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
	"bytes"
	"io/ioutil"
	"os"
	"testing"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

type getAuthInfosTest struct {
	config   clientcmdapi.Config
	expected string
}

func TestGetAuthInfos(t *testing.T) {
	conf := clientcmdapi.Config{
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"some-username": {Token: "d209caf7c447ad03c8c2621916ab4a9j"},
		},
	}
	test := getAuthInfosTest{
		config: conf,
		expected: `NAME
some-username
`,
	}

	test.run(t)
}

func TestGetAuthInfosEmpty(t *testing.T) {
	test := getAuthInfosTest{
		config:   clientcmdapi.Config{},
		expected: "NAME\n",
	}

	test.run(t)
}

func (test getAuthInfosTest) run(t *testing.T) {
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
	cmd := NewCmdConfigGetAuthInfos(buf, pathOptions)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v", err)
	}
	if len(test.expected) != 0 {
		if buf.String() != test.expected {
			t.Errorf("expected %v, but got %v", test.expected, buf.String())
		}
		return
	}
}
