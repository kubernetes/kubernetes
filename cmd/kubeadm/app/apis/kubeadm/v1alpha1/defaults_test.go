/*
Copyright 2017 The Kubernetes Authors.

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

package v1alpha1

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

// Register API defaulters functions
func registerAPIDefFunctions(t *testing.T) {
	if err := addDefaultingFuncs(api.Scheme); err != nil {
		t.Errorf("Error registering defaulters functions: '%v'", err)
	}
}

func TestMasterConfigDefaults(t *testing.T) {
	registerAPIDefFunctions(t)

	// Create a master configuration and set defaults
	cfg := &MasterConfiguration{}
	api.Scheme.Default(cfg)

	// Check default string fields
	stringFields := []struct {
		name       string
		valPtr     *string
		expDefault string
	}{
		{
			name:       "DNSDomain",
			valPtr:     &cfg.Networking.DNSDomain,
			expDefault: DefaultServiceDNSDomain,
		}, {
			name:       "ServiceSubnet",
			valPtr:     &cfg.Networking.ServiceSubnet,
			expDefault: DefaultServicesSubnet,
		}, {
			name:       "KubernetesVersion",
			valPtr:     &cfg.KubernetesVersion,
			expDefault: DefaultKubernetesVersion,
		}, {
			name:       "CertificatesDir",
			valPtr:     &cfg.CertificatesDir,
			expDefault: DefaultCertificatesDir,
		}, {
			name:       "EtcdDataDir",
			valPtr:     &cfg.Etcd.DataDir,
			expDefault: DefaultEtcdDataDir,
		}, {
			name:       "ImageRepository",
			valPtr:     &cfg.ImageRepository,
			expDefault: DefaultImageRepository,
		},
	}
	for _, field := range stringFields {
		val := *field.valPtr
		if *field.valPtr != field.expDefault {
			t.Errorf("Wrong default for %s, got: '%s', expected: '%s'", field.name, val, field.expDefault)
		}
	}

	// Check default API bind port
	if cfg.API.BindPort != DefaultAPIBindPort {
		t.Errorf("Wrong default for APIBindPort, got: '%d', expected: '%d'", cfg.API.BindPort, DefaultAPIBindPort)
	}

	// Check default AuthorizationModes
	expModes := strings.Split(DefaultAuthorizationModes, ",")
	if !reflect.DeepEqual(cfg.AuthorizationModes, expModes) {
		t.Errorf("Wrong default for AuthorizationModes, got: %v, expected: %v", cfg.AuthorizationModes, expModes)
	}
}

func TestNodeConfigDefaults(t *testing.T) {
	registerAPIDefFunctions(t)

	discoveryFiles := []string{
		"",
		"http://example.com/foobar",
		"file:/temp/foobar",
	}

	for _, discoveryFile := range discoveryFiles {
		// Create a node configuration and set token and discovery file
		cfg := &NodeConfiguration{}
		token := "testToken"
		cfg.Token = token
		cfg.DiscoveryFile = discoveryFile

		// Set defaults
		api.Scheme.Default(cfg)

		// Check default values
		if cfg.CACertPath != DefaultCACertPath {
			t.Errorf("Wrong default for CACertPath, got: '%s', expected: '%s'", cfg.CACertPath, DefaultCACertPath)
		}
		if cfg.TLSBootstrapToken != token {
			t.Errorf("Wrong default for TLSBootstrapToken, got: '%s', expected: '%s'", cfg.TLSBootstrapToken, token)
		}
		if discoveryFile == "" {
			if cfg.DiscoveryToken != token {
				t.Errorf("Wrong default for DiscoveryToken, got: '%s', expected: '%s'", cfg.DiscoveryToken, token)
			}
		} else {
			// Strip "file:" prefix from discovery file if necessary
			if strings.HasPrefix(discoveryFile, "file:") {
				discoveryFile = discoveryFile[len("file:"):]
				if cfg.DiscoveryFile != discoveryFile {
					t.Errorf("Wrong default for DiscoveryFile, got: '%s', expected: '%s'", cfg.DiscoveryFile, discoveryFile)
				}
			}
		}
	}
}
