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

package config

import (
	"bytes"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cliflag "k8s.io/component-base/cli/flag"
)

type setClusterRunTest struct {
	name           string
	startingConfig *clientcmdapi.Config
	options        *SetClusterOptions
	expectedConfig *clientcmdapi.Config
	expectedOut    string
	runError       string
}

type setClusterToOptionsTest struct {
	name            string
	args            []string
	flags           *SetClusterFlags
	expectedOptions *SetClusterOptions
	expectedError   string
}

func TestRunSetCluster(t *testing.T) {
	t.Parallel()

	startingConfigEmpty := clientcmdapi.NewConfig()

	expectedConfig := clientcmdapi.NewConfig()
	testCluster1 := clientcmdapi.NewCluster()
	testCluster1.Server = "https://1.2.3.4:8283"
	expectedConfig.Clusters = map[string]*clientcmdapi.Cluster{
		"my-cluster": testCluster1,
	}

	serverFlag := cliflag.StringFlag{}
	if err := serverFlag.Set("https://1.2.3.4:8283"); err != nil {
		t.Errorf("unexpected error setting serverFlag: %v", err)
	}

	for _, test := range []setClusterRunTest{
		{
			name:           "CurrentContext",
			startingConfig: startingConfigEmpty,
			options: &SetClusterOptions{
				Name:   "my-cluster",
				Server: serverFlag,
			},
			expectedConfig: expectedConfig,
			expectedOut:    "Cluster \"my-cluster\" set.\n",
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			fakeKubeFile, err := generateTestKubeConfig(*test.startingConfig)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			defer removeTempFile(t, fakeKubeFile.Name())

			pathOptions := clientcmd.NewDefaultPathOptions()
			pathOptions.GlobalFile = fakeKubeFile.Name()
			pathOptions.EnvVar = ""

			streams, _, buffOut, _ := genericclioptions.NewTestIOStreams()

			test.options.IOStreams = streams
			test.options.ConfigAccess = pathOptions

			err = test.options.RunSetCluster()
			if len(test.runError) != 0 && err != nil {
				checkOutputResults(t, err.Error(), test.runError)
				return
			} else if len(test.runError) != 0 && err == nil {
				t.Errorf("expected error %q running to options but non received", test.runError)
				return
			} else if err != nil {
				t.Errorf("unexpected error running to options: %v", err)
			}

			if len(test.expectedOut) != 0 {
				checkOutputResults(t, buffOut.String(), test.expectedOut)
				checkOutputConfig(t, test.options.ConfigAccess, test.expectedConfig, cmp.Options{})
			}
		})
	}
}

func TestSetClusterToOptions(t *testing.T) {
	t.Parallel()

	serverFlag := cliflag.StringFlag{}
	if err := serverFlag.Set("https://1.2.3.4:8283"); err != nil {
		t.Errorf("unexpected error setting serverFlag: %v", err)
	}

	certificateAuthorityFlag := cliflag.StringFlag{}
	if err := certificateAuthorityFlag.Set("cert.ca"); err != nil {
		t.Errorf("unexpected error setting serverFlag: %v", err)
	}

	proxyURLFlag := cliflag.StringFlag{}
	if err := proxyURLFlag.Set("http://proxy.fake/"); err != nil {
		t.Errorf("unexpected error setting serverFlag: %v", err)
	}

	tlsServerNameFlag := cliflag.StringFlag{}
	if err := tlsServerNameFlag.Set("tls-server.fake"); err != nil {
		t.Errorf("unexpected error setting serverFlag: %v", err)
	}

	for _, test := range []setClusterToOptionsTest{
		{
			name: "DefaultBoolsSetValues",
			args: []string{"my-cluster"},
			flags: &SetClusterFlags{
				certificateAuthority:  certificateAuthorityFlag,
				embedCAData:           false,
				insecureSkipTLSVerify: false,
				proxyURL:              proxyURLFlag,
				server:                serverFlag,
				tlsServerName:         tlsServerNameFlag,
			},
			expectedOptions: &SetClusterOptions{
				Name:                  "my-cluster",
				CertificateAuthority:  certificateAuthorityFlag,
				EmbedCAData:           false,
				InsecureSkipTLSVerify: false,
				ProxyURL:              proxyURLFlag,
				Server:                serverFlag,
				TlsServerName:         tlsServerNameFlag,
			},
		}, {
			name:          "ErrorZeroArgs",
			args:          []string{},
			flags:         &SetClusterFlags{},
			expectedError: "unexpected args: ",
		}, {
			name:          "ErrorTwoArgs",
			args:          []string{"my-cluster", "bad-arg"},
			flags:         &SetClusterFlags{},
			expectedError: "unexpected args: [my-cluster bad-arg]",
		}, {
			name: "ErrorInsecureAndCertAuthority",
			args: []string{"my-cluster"},
			flags: &SetClusterFlags{
				insecureSkipTLSVerify: true,
				certificateAuthority:  certificateAuthorityFlag,
			},
			expectedError: "you cannot specify a certificate authority and insecure mode at the same time",
		}, {
			name: "ErrorEmbededCADataNoCertAuthority",
			args: []string{"my-cluster"},
			flags: &SetClusterFlags{
				embedCAData: true,
			},
			expectedError: "you must specify a --certificate-authority to embed",
		}, {
			name: "ErrorCertFileNotFound",
			args: []string{"my-cluster"},
			flags: &SetClusterFlags{
				embedCAData:          true,
				certificateAuthority: certificateAuthorityFlag,
			},
			expectedError: "could not stat certificate-authority file",
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			fakeKubeFile, err := generateTestKubeConfig(*clientcmdapi.NewConfig())
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			defer removeTempFile(t, fakeKubeFile.Name())

			pathOptions := clientcmd.NewDefaultPathOptions()
			pathOptions.GlobalFile = fakeKubeFile.Name()
			pathOptions.EnvVar = ""

			streams, _, _, _ := genericclioptions.NewTestIOStreams()

			test.flags.configAccess = pathOptions
			test.flags.ioStreams = streams

			options, err := test.flags.ToOptions(test.args)
			if len(test.expectedError) != 0 && err != nil {
				checkOutputResults(t, err.Error(), test.expectedError)
				return
			} else if len(test.expectedError) != 0 && err == nil {
				t.Errorf("expected error %q running to options but non received", test.expectedError)
				return
			} else if err != nil {
				t.Errorf("unexpected error running to options: %v", err)
			}

			// finish options for proper comparison
			test.expectedOptions.IOStreams = streams
			test.expectedOptions.ConfigAccess = pathOptions

			cmpOptions := cmpopts.IgnoreUnexported(
				cliflag.StringFlag{},
				bytes.Buffer{})
			if cmp.Diff(test.expectedOptions, options, cmpOptions) != "" {
				t.Errorf("expected options did not match actual options (-want, +got):\n%v", cmp.Diff(test.expectedOptions, options, cmpOptions))
			}
		})
	}
}

// To avoid complication I've broken out testing when a cert file that needs to be written out into this function
func TestCertAuthorityFile(t *testing.T) {
	t.Parallel()

	certificateAuthorityFlag := cliflag.StringFlag{}
	if err := certificateAuthorityFlag.Set("cert.ca"); err != nil {
		t.Errorf("unexpected error setting serverFlag: %v", err)
	}

	for _, test := range []setClusterToOptionsTest{
		{
			name: "EmbedCertData",
			args: []string{"my-cluster"},
			flags: &SetClusterFlags{
				certificateAuthority: certificateAuthorityFlag,
				embedCAData:          true,
			},
			expectedOptions: &SetClusterOptions{
				Name:                  "my-cluster",
				CertificateAuthority:  certificateAuthorityFlag,
				EmbedCAData:           true,
				InsecureSkipTLSVerify: false,
				ProxyURL:              cliflag.StringFlag{},
				Server:                cliflag.StringFlag{},
				TlsServerName:         cliflag.StringFlag{},
			},
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			fakeKubeFile, err := generateTestKubeConfig(*clientcmdapi.NewConfig())
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			defer removeTempFile(t, fakeKubeFile.Name())

			fakeCAFile, err := os.CreateTemp(os.TempDir(), "")
			if err != nil {
				t.Errorf("error creating fake certificate authority file: %v", err)
			}

			defer removeTempFile(t, fakeCAFile.Name())

			if err := test.flags.certificateAuthority.Set(fakeCAFile.Name()); err != nil {
				t.Errorf("error overwriting provided CA flag: %v", err)
			}
			fakeCertData := []byte("I guess it doesn't matter what goes in here\n")
			if _, err := fakeCAFile.Write(fakeCertData); err != nil {
				t.Errorf("error writing fake certificate authority file: %v", err)
			}

			pathOptions := clientcmd.NewDefaultPathOptions()
			pathOptions.GlobalFile = fakeKubeFile.Name()
			pathOptions.EnvVar = ""

			streams, _, _, _ := genericclioptions.NewTestIOStreams()

			test.flags.configAccess = pathOptions
			test.flags.ioStreams = streams

			options, err := test.flags.ToOptions(test.args)
			if len(test.expectedError) != 0 && err != nil {
				checkOutputResults(t, err.Error(), test.expectedError)
				return
			} else if len(test.expectedError) != 0 && err == nil {
				t.Errorf("expected error %q running command but non received", test.expectedError)
				return
			} else if err != nil {
				t.Errorf("unexpected error running to options: %v", err)
			}

			// finish options for proper comparison
			test.expectedOptions.IOStreams = streams
			test.expectedOptions.ConfigAccess = pathOptions

			// set CA to the real name of the fake CA file that was generated for proper DeepEqual testing
			if err := options.CertificateAuthority.Set(fakeCAFile.Name()); err != nil {
				t.Errorf("unexpected error overwriting CA file name: %v", err)
			}

			cmpOptions := cmpopts.IgnoreUnexported(
				cliflag.StringFlag{},
				bytes.Buffer{})
			if cmp.Diff(test.expectedOptions, options, cmpOptions) != "" {
				t.Errorf("expected options did not match actual options (-want, +got):\n%v", cmp.Diff(test.expectedOptions, options, cmpOptions))
			}
		})
	}
}
