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
	"fmt"

	"gopkg.in/v2/yaml"
)

func ExampleEmptyConfig() {
	defaultConfig := NewConfig()

	output, err := yaml.Marshal(defaultConfig)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}

	fmt.Printf("%v", string(output))
	// Output:
	// preferences: {}
	// clusters: {}
	// users: {}
	// contexts: {}
	// current-context: ""
}

func ExampleOfOptionsConfig() {
	defaultConfig := NewConfig()
	defaultConfig.Preferences.Colors = true
	defaultConfig.Clusters["alfa"] = Cluster{
		Server:                "https://alfa.org:8080",
		APIVersion:            "v1beta2",
		InsecureSkipTLSVerify: true,
		CertificateAuthority:  "path/to/my/cert-ca-filename",
	}
	defaultConfig.Clusters["bravo"] = Cluster{
		Server:                "https://bravo.org:8080",
		APIVersion:            "v1beta1",
		InsecureSkipTLSVerify: false,
	}
	defaultConfig.AuthInfos["black-mage-via-file"] = AuthInfo{
		AuthPath: "path/to/my/.kubernetes_auth",
	}
	defaultConfig.AuthInfos["white-mage-via-cert"] = AuthInfo{
		ClientCertificate: "path/to/my/client-cert-filename",
		ClientKey:         "path/to/my/client-key-filename",
	}
	defaultConfig.AuthInfos["red-mage-via-token"] = AuthInfo{
		Token: "my-secret-token",
	}
	defaultConfig.Contexts["bravo-as-black-mage"] = Context{
		Cluster:   "bravo",
		AuthInfo:  "black-mage-via-file",
		Namespace: "yankee",
	}
	defaultConfig.Contexts["alfa-as-black-mage"] = Context{
		Cluster:   "alfa",
		AuthInfo:  "black-mage-via-file",
		Namespace: "zulu",
	}
	defaultConfig.Contexts["alfa-as-white-mage"] = Context{
		Cluster:  "alfa",
		AuthInfo: "white-mage-via-cert",
	}
	defaultConfig.CurrentContext = "alfa-as-white-mage"

	output, err := yaml.Marshal(defaultConfig)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}

	fmt.Printf("%v", string(output))
	// Output:
	// preferences:
	//   colors: true
	// clusters:
	//   alfa:
	//     server: https://alfa.org:8080
	//     api-version: v1beta2
	//     insecure-skip-tls-verify: true
	//     certificate-authority: path/to/my/cert-ca-filename
	//   bravo:
	//     server: https://bravo.org:8080
	//     api-version: v1beta1
	// users:
	//   black-mage-via-file:
	//     auth-path: path/to/my/.kubernetes_auth
	//   red-mage-via-token:
	//     token: my-secret-token
	//   white-mage-via-cert:
	//     client-certificate: path/to/my/client-cert-filename
	//     client-key: path/to/my/client-key-filename
	// contexts:
	//   alfa-as-black-mage:
	//     cluster: alfa
	//     user: black-mage-via-file
	//     namespace: zulu
	//   alfa-as-white-mage:
	//     cluster: alfa
	//     user: white-mage-via-cert
	//   bravo-as-black-mage:
	//     cluster: bravo
	//     user: black-mage-via-file
	//     namespace: yankee
	// current-context: alfa-as-white-mage
}
