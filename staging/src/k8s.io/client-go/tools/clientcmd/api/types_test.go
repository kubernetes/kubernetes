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

package api

import (
	"fmt"

	"sigs.k8s.io/yaml"
)

func Example_emptyConfig() {
	defaultConfig := NewConfig()

	output, err := yaml.Marshal(defaultConfig)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}

	fmt.Printf("%v", string(output))
	// Output:
	// clusters: {}
	// contexts: {}
	// current-context: ""
	// users: {}
}

func Example_ofOptionsConfig() {
	defaultConfig := NewConfig()
	defaultConfig.Preferences.Colors = true
	defaultConfig.Clusters["alfa"] = &Cluster{
		Server:                "https://alfa.org:8080",
		InsecureSkipTLSVerify: true,
		CertificateAuthority:  "path/to/my/cert-ca-filename",
		DisableCompression:    true,
	}
	defaultConfig.Clusters["bravo"] = &Cluster{
		Server:                "https://bravo.org:8080",
		InsecureSkipTLSVerify: false,
		DisableCompression:    false,
	}
	defaultConfig.AuthInfos["white-mage-via-cert"] = &AuthInfo{
		ClientCertificate: "path/to/my/client-cert-filename",
		ClientKey:         "path/to/my/client-key-filename",
	}
	defaultConfig.AuthInfos["red-mage-via-token"] = &AuthInfo{
		Token: "my-secret-token",
	}
	defaultConfig.AuthInfos["black-mage-via-auth-provider"] = &AuthInfo{
		AuthProvider: &AuthProviderConfig{
			Name: "gcp",
			Config: map[string]string{
				"foo":   "bar",
				"token": "s3cr3t-t0k3n",
			},
		},
	}
	defaultConfig.Contexts["bravo-as-black-mage"] = &Context{
		Cluster:   "bravo",
		AuthInfo:  "black-mage-via-auth-provider",
		Namespace: "yankee",
	}
	defaultConfig.Contexts["alfa-as-black-mage"] = &Context{
		Cluster:   "alfa",
		AuthInfo:  "black-mage-via-auth-provider",
		Namespace: "zulu",
	}
	defaultConfig.Contexts["alfa-as-white-mage"] = &Context{
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
	// clusters:
	//   alfa:
	//     certificate-authority: path/to/my/cert-ca-filename
	//     disable-compression: true
	//     insecure-skip-tls-verify: true
	//     server: https://alfa.org:8080
	//   bravo:
	//     server: https://bravo.org:8080
	// contexts:
	//   alfa-as-black-mage:
	//     cluster: alfa
	//     namespace: zulu
	//     user: black-mage-via-auth-provider
	//   alfa-as-white-mage:
	//     cluster: alfa
	//     user: white-mage-via-cert
	//   bravo-as-black-mage:
	//     cluster: bravo
	//     namespace: yankee
	//     user: black-mage-via-auth-provider
	// current-context: alfa-as-white-mage
	// preferences:
	//   colors: true
	// users:
	//   black-mage-via-auth-provider:
	//     auth-provider:
	//       config:
	//         foo: bar
	//         token: s3cr3t-t0k3n
	//       name: gcp
	//   red-mage-via-token:
	//     token: my-secret-token
	//   white-mage-via-cert:
	//     client-certificate: path/to/my/client-cert-filename
	//     client-key: path/to/my/client-key-filename
}
