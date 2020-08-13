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

package token

import (
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/clientcmd"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	tokenjws "k8s.io/cluster-bootstrap/token/jws"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"

	"github.com/pmezard/go-difflib/difflib"
)

func TestRetrieveValidatedConfigInfo(t *testing.T) {
	const (
		caCert = `-----BEGIN CERTIFICATE-----
MIICyDCCAbCgAwIBAgIBADANBgkqhkiG9w0BAQsFADAVMRMwEQYDVQQDEwprdWJl
cm5ldGVzMB4XDTE5MTEyMDAwNDk0MloXDTI5MTExNzAwNDk0MlowFTETMBEGA1UE
AxMKa3ViZXJuZXRlczCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMqQ
ctECzA8yFSuVYupOUYgrTmfQeKe/9BaDWagaq7ow9+I2IvsfWFvlrD8QQr8sea6q
xjq7TV67Vb4RxBaoYDA+yI5vIcujWUxULun64lu3Q6iC1sj2UnmUpIdgazRXXEkZ
vxA6EbAnoxA0+lBOn1CZWl23IQ4s70o2hZ7wIp/vevB88RRRjqtvgc5elsjsbmDF
LS7L1Zuye8c6gS93bR+VjVmSIfr1IEq0748tIIyXjAVCWPVCvuP41MlfPc/JVpZD
uD2+pO6ZYREcdAnOf2eD4/eLOMKko4L1dSFy9JKM5PLnOC0Zk0AYOd1vS8DTAfxj
XPEIY8OBYFhlsxf4TE8CAwEAAaMjMCEwDgYDVR0PAQH/BAQDAgKkMA8GA1UdEwEB
/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBAH/OYq8zyl1+zSTmuow3yI/15PL1
dl8hB7IKnZNWmC/LTdm/+noh3Sb1IdRv6HkKg/GUn0UMuRUngLhju3EO4ozJPQcX
quaxzgmTKNWJ6ErDvRvWhGX0ZcbdBfZv+dowyRqzd5nlJ49hC+NrtFFQq6P05BYn
7SemguqeXmXwIj2Sa+1DeR6lRm9o8shAYjnyThUFqaMn18kI3SANJ5vk/3DFrPEO
CKC9EzFku2kuxg2dM12PbRGZQ2o0K6HEZgrrIKTPOy3ocb8r9M0aSFhjOV/NqGA4
SaupXSW6XfvIi/UHoIbU3pNcsnUJGnQfQvip95XKk/gqcUr+m50vxgumxtA=
-----END CERTIFICATE-----`

		caCertHash = "sha256:98be2e6d4d8a89aa308fb15de0c07e2531ce549c68dec1687cdd5c06f0826658"

		expectedKubeconfig = `apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUN5RENDQWJDZ0F3SUJBZ0lCQURBTkJna3Foa2lHOXcwQkFRc0ZBREFWTVJNd0VRWURWUVFERXdwcmRXSmwKY201bGRHVnpNQjRYRFRFNU1URXlNREF3TkRrME1sb1hEVEk1TVRFeE56QXdORGswTWxvd0ZURVRNQkVHQTFVRQpBeE1LYTNWaVpYSnVaWFJsY3pDQ0FTSXdEUVlKS29aSWh2Y05BUUVCQlFBRGdnRVBBRENDQVFvQ2dnRUJBTXFRCmN0RUN6QTh5RlN1Vll1cE9VWWdyVG1mUWVLZS85QmFEV2FnYXE3b3c5K0kySXZzZldGdmxyRDhRUXI4c2VhNnEKeGpxN1RWNjdWYjRSeEJhb1lEQSt5STV2SWN1aldVeFVMdW42NGx1M1E2aUMxc2oyVW5tVXBJZGdhelJYWEVrWgp2eEE2RWJBbm94QTArbEJPbjFDWldsMjNJUTRzNzBvMmhaN3dJcC92ZXZCODhSUlJqcXR2Z2M1ZWxzanNibURGCkxTN0wxWnV5ZThjNmdTOTNiUitWalZtU0lmcjFJRXEwNzQ4dElJeVhqQVZDV1BWQ3Z1UDQxTWxmUGMvSlZwWkQKdUQyK3BPNlpZUkVjZEFuT2YyZUQ0L2VMT01La280TDFkU0Z5OUpLTTVQTG5PQzBaazBBWU9kMXZTOERUQWZ4agpYUEVJWThPQllGaGxzeGY0VEU4Q0F3RUFBYU1qTUNFd0RnWURWUjBQQVFIL0JBUURBZ0trTUE4R0ExVWRFd0VCCi93UUZNQU1CQWY4d0RRWUpLb1pJaHZjTkFRRUxCUUFEZ2dFQkFIL09ZcTh6eWwxK3pTVG11b3czeUkvMTVQTDEKZGw4aEI3SUtuWk5XbUMvTFRkbS8rbm9oM1NiMUlkUnY2SGtLZy9HVW4wVU11UlVuZ0xoanUzRU80b3pKUFFjWApxdWF4emdtVEtOV0o2RXJEdlJ2V2hHWDBaY2JkQmZaditkb3d5UnF6ZDVubEo0OWhDK05ydEZGUXE2UDA1QlluCjdTZW1ndXFlWG1Yd0lqMlNhKzFEZVI2bFJtOW84c2hBWWpueVRoVUZxYU1uMThrSTNTQU5KNXZrLzNERnJQRU8KQ0tDOUV6Rmt1Mmt1eGcyZE0xMlBiUkdaUTJvMEs2SEVaZ3JySUtUUE95M29jYjhyOU0wYVNGaGpPVi9OcUdBNApTYXVwWFNXNlhmdklpL1VIb0liVTNwTmNzblVKR25RZlF2aXA5NVhLay9ncWNVcittNTB2eGd1bXh0QT0KLS0tLS1FTkQgQ0VSVElGSUNBVEUtLS0tLQ==
    server: https://127.0.0.1
  name: somecluster
contexts:
- context:
    cluster: somecluster
    user: token-bootstrap-client
  name: token-bootstrap-client@somecluster
current-context: token-bootstrap-client@somecluster
kind: Config
preferences: {}
users: null
`
	)

	tests := []struct {
		name                     string
		tokenID                  string
		tokenSecret              string
		cfg                      *kubeadmapi.Discovery
		configMap                *fakeConfigMap
		delayedJWSSignaturePatch bool
		expectedError            bool
	}{
		{
			// This is the default behavior. The JWS signature is patched after the cluster-info ConfigMap is created
			name:        "valid: retrieve a valid kubeconfig with CA verification and delayed JWS signature",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token:        "123456.abcdef1234567890",
					CACertHashes: []string{caCertHash},
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: map[string]string{},
			},
			delayedJWSSignaturePatch: true,
		},
		{
			// Same as above expect this test creates the ConfigMap with the JWS signature
			name:        "valid: retrieve a valid kubeconfig with CA verification",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token:        "123456.abcdef1234567890",
					CACertHashes: []string{caCertHash},
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: nil,
			},
		},
		{
			// Skipping CA verification is also supported
			name:        "valid: retrieve a valid kubeconfig without CA verification",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "123456.abcdef1234567890",
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: nil,
			},
		},
		{
			name:        "invalid: token format is invalid",
			tokenID:     "foo",
			tokenSecret: "bar",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "foo.bar",
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: nil,
			},
			expectedError: true,
		},
		{
			name:        "invalid: missing cluster-info ConfigMap",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "123456.abcdef1234567890",
				},
			},
			configMap: &fakeConfigMap{
				name: "baz",
				data: nil,
			},
			expectedError: true,
		},
		{
			name:        "invalid: wrong JWS signature",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "123456.abcdef1234567890",
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: map[string]string{
					bootstrapapi.KubeConfigKey:                    "foo",
					bootstrapapi.JWSSignatureKeyPrefix + "123456": "bar",
				},
			},
			expectedError: true,
		},
		{
			name:        "invalid: missing key for JWSSignatureKeyPrefix",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token: "123456.abcdef1234567890",
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: map[string]string{
					bootstrapapi.KubeConfigKey: "foo",
				},
			},
			expectedError: true,
		},
		{
			name:        "invalid: wrong CA cert hash",
			tokenID:     "123456",
			tokenSecret: "abcdef1234567890",
			cfg: &kubeadmapi.Discovery{
				BootstrapToken: &kubeadmapi.BootstrapTokenDiscovery{
					Token:        "123456.abcdef1234567890",
					CACertHashes: []string{"foo"},
				},
			},
			configMap: &fakeConfigMap{
				name: bootstrapapi.ConfigMapClusterInfo,
				data: nil,
			},
			expectedError: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			kubeconfig := buildSecureBootstrapKubeConfig("127.0.0.1", []byte(caCert), "somecluster")
			kubeconfigBytes, err := clientcmd.Write(*kubeconfig)
			if err != nil {
				t.Fatalf("cannot marshal kubeconfig %v", err)
			}

			// Generate signature of the insecure kubeconfig
			sig, err := tokenjws.ComputeDetachedSignature(string(kubeconfigBytes), test.tokenID, test.tokenSecret)
			if err != nil {
				t.Fatalf("cannot compute detached JWS signature: %v", err)
			}

			// If the JWS signature is delayed, only add the kubeconfig
			if test.delayedJWSSignaturePatch {
				test.configMap.data = map[string]string{}
				test.configMap.data[bootstrapapi.KubeConfigKey] = string(kubeconfigBytes)
			}

			// Populate the default cluster-info data
			if test.configMap.data == nil {
				test.configMap.data = map[string]string{}
				test.configMap.data[bootstrapapi.KubeConfigKey] = string(kubeconfigBytes)
				test.configMap.data[bootstrapapi.JWSSignatureKeyPrefix+test.tokenID] = sig
			}

			// Create a fake client and create the cluster-info ConfigMap
			client := fakeclient.NewSimpleClientset()
			if err = test.configMap.createOrUpdate(client); err != nil {
				t.Fatalf("could not create ConfigMap: %v", err)
			}

			// Set arbitrary discovery timeout and retry interval
			test.cfg.Timeout = &metav1.Duration{Duration: time.Millisecond * 200}
			interval := time.Millisecond * 20

			// Patch the JWS signature after a short delay
			if test.delayedJWSSignaturePatch {
				test.configMap.data[bootstrapapi.JWSSignatureKeyPrefix+test.tokenID] = sig
				go func() {
					time.Sleep(time.Millisecond * 60)
					if err := test.configMap.createOrUpdate(client); err != nil {
						t.Errorf("could not update the cluster-info ConfigMap with a JWS signature: %v", err)
					}
				}()
			}

			// Retrieve validated configuration
			kubeconfig, err = retrieveValidatedConfigInfo(client, test.cfg, interval)
			if (err != nil) != test.expectedError {
				t.Errorf("expected error %v, got %v, error: %v", test.expectedError, err != nil, err)
			}

			// Return if an error is expected
			if test.expectedError {
				return
			}

			// Validate the resulted kubeconfig
			kubeconfigBytes, err = clientcmd.Write(*kubeconfig)
			if err != nil {
				t.Fatalf("cannot marshal resulted kubeconfig %v", err)
			}
			if string(kubeconfigBytes) != expectedKubeconfig {
				t.Error("unexpected kubeconfig")
				diff := difflib.UnifiedDiff{
					A:        difflib.SplitLines(expectedKubeconfig),
					B:        difflib.SplitLines(string(kubeconfigBytes)),
					FromFile: "expected",
					ToFile:   "got",
					Context:  10,
				}
				diffstr, err := difflib.GetUnifiedDiffString(diff)
				if err != nil {
					t.Fatalf("error generating unified diff string: %v", err)
				}
				t.Errorf("\n%s", diffstr)
			}
		})
	}
}

type fakeConfigMap struct {
	name string
	data map[string]string
}

func (c *fakeConfigMap) createOrUpdate(client clientset.Interface) error {
	return apiclient.CreateOrUpdateConfigMap(client, &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      c.name,
			Namespace: metav1.NamespacePublic,
		},
		Data: c.data,
	})
}
