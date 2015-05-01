/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"io/ioutil"
	"os"
	"testing"

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
)

// Verifies that referencing an old .kubernetes_auth file respects all fields
func TestAuthPathUpdatesBothClusterAndUser(t *testing.T) {
	authFile, _ := ioutil.TempFile("", "")
	defer os.Remove(authFile.Name())

	insecure := true
	auth := &clientauth.Info{
		User:        "user",
		Password:    "password",
		CAFile:      "ca-file",
		CertFile:    "cert-file",
		KeyFile:     "key-file",
		BearerToken: "bearer-token",
		Insecure:    &insecure,
	}
	err := testWriteAuthInfoFile(*auth, authFile.Name())
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}

	cmd := &cobra.Command{
		Run: func(cmd *cobra.Command, args []string) {
		},
	}
	clientConfig := testBindClientConfig(cmd)
	cmd.ParseFlags([]string{"--server=https://localhost", "--auth-path=" + authFile.Name()})

	config, err := clientConfig.ClientConfig()
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}

	matchStringArg(auth.User, config.Username, t)
	matchStringArg(auth.Password, config.Password, t)
	matchStringArg(auth.CAFile, config.CAFile, t)
	matchStringArg(auth.CertFile, config.CertFile, t)
	matchStringArg(auth.KeyFile, config.KeyFile, t)
	matchStringArg(auth.BearerToken, config.BearerToken, t)
	matchBoolArg(*auth.Insecure, config.Insecure, t)
}

func testWriteAuthInfoFile(auth clientauth.Info, filename string) error {
	data, err := json.Marshal(auth)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(filename, data, 0600)
	return err
}

func testBindClientConfig(cmd *cobra.Command) ClientConfig {
	loadingRules := &ClientConfigLoadingRules{}
	cmd.PersistentFlags().StringVar(&loadingRules.ExplicitPath, "kubeconfig", "", "Path to the kubeconfig file to use for CLI requests.")

	overrides := &ConfigOverrides{}
	BindOverrideFlags(overrides, cmd.PersistentFlags(), RecommendedConfigOverrideFlags(""))
	clientConfig := NewInteractiveDeferredLoadingClientConfig(loadingRules, overrides, os.Stdin)

	return clientConfig
}
