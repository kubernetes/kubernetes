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

package webhook

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/tools/clientcmd/api/v1"
	"k8s.io/kubernetes/pkg/apis/admission/v1alpha1"

	_ "k8s.io/kubernetes/pkg/apis/admission/install"
)

const (
	errAdmitPrefix = `pods "my-pod" is forbidden: `
)

var (
	requestCount int
	testRoot     string
	testServer   *httptest.Server
)

type genericAdmissionWebhookTest struct {
	test   string
	config *GenericAdmissionWebhookConfig
	value  interface{}
}

func TestMain(m *testing.M) {
	tmpRoot, err := ioutil.TempDir("", "")

	if err != nil {
		killTests(err)
	}

	testRoot = tmpRoot

	// Cleanup
	defer os.RemoveAll(testRoot)

	// Create the test webhook server
	cert, err := tls.X509KeyPair(clientCert, clientKey)

	if err != nil {
		killTests(err)
	}

	rootCAs := x509.NewCertPool()

	rootCAs.AppendCertsFromPEM(caCert)

	testServer = httptest.NewUnstartedServer(http.HandlerFunc(webhookHandler))

	testServer.TLS = &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientCAs:    rootCAs,
		ClientAuth:   tls.RequireAndVerifyClientCert,
	}

	// Create the test webhook server
	testServer.StartTLS()

	// Cleanup
	defer testServer.Close()

	// Create an invalid and valid Kubernetes configuration file
	var kubeConfig bytes.Buffer

	err = json.NewEncoder(&kubeConfig).Encode(v1.Config{
		Clusters: []v1.NamedCluster{
			{
				Cluster: v1.Cluster{
					Server: testServer.URL,
					CertificateAuthorityData: caCert,
				},
			},
		},
		AuthInfos: []v1.NamedAuthInfo{
			{
				AuthInfo: v1.AuthInfo{
					ClientCertificateData: clientCert,
					ClientKeyData:         clientKey,
				},
			},
		},
	})

	if err != nil {
		killTests(err)
	}

	// The files needed on disk for the webhook tests
	files := map[string][]byte{
		"ca.pem":         caCert,
		"client.pem":     clientCert,
		"client-key.pem": clientKey,
		"kube-config":    kubeConfig.Bytes(),
	}

	// Write the certificate files to disk or fail
	for fileName, fileData := range files {
		if err := ioutil.WriteFile(filepath.Join(testRoot, fileName), fileData, 0400); err != nil {
			killTests(err)
		}
	}

	// Run the tests
	m.Run()
}

// TestNewGenericAdmissionWebhook tests that NewGenericAdmissionWebhook works as expected
func TestNewGenericAdmissionWebhook(t *testing.T) {
	tests := []genericAdmissionWebhookTest{
		genericAdmissionWebhookTest{
			test:   "Empty webhook config",
			config: &GenericAdmissionWebhookConfig{},
			value:  fmt.Errorf("kubeConfigFile is required"),
		},
		genericAdmissionWebhookTest{
			test: "Broken webhook config",
			config: &GenericAdmissionWebhookConfig{
				KubeConfigFile: filepath.Join(testRoot, "kube-config.missing"),
				Rules: []Rule{
					Rule{
						Type: Skip,
					},
				},
			},
			value: fmt.Errorf("stat .*kube-config.missing.*"),
		},
		genericAdmissionWebhookTest{
			test: "Valid webhook config",
			config: &GenericAdmissionWebhookConfig{
				KubeConfigFile: filepath.Join(testRoot, "kube-config"),
				Rules: []Rule{
					Rule{
						Type: Skip,
					},
				},
			},
			value: nil,
		},
	}

	for _, tt := range tests {
		_, err := NewGenericAdmissionWebhook(tt.config)

		checkForError(t, tt.test, tt.value, err)
	}
}

// TestAdmit tests that GenericAdmissionWebhook#Admit works as expected
func TestAdmit(t *testing.T) {
	configWithFailAction := makeGoodConfig()
	kind := api.Kind("Pod").WithVersion("v1")
	name := "my-pod"
	namespace := "webhook-test"
	object := api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				"pod.name": name,
			},
			Name:      name,
			Namespace: namespace,
		},
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
	}
	oldObject := api.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
	}
	operation := admission.Update
	resource := api.Resource("pods").WithVersion("v1")
	subResource := ""
	userInfo := user.DefaultInfo{
		Name: "webhook-test",
		UID:  "webhook-test",
	}

	configWithFailAction.Rules[0].FailAction = Allow

	tests := []genericAdmissionWebhookTest{
		genericAdmissionWebhookTest{
			test: "No matching rule",
			config: &GenericAdmissionWebhookConfig{
				KubeConfigFile: filepath.Join(testRoot, "kube-config"),
				Rules: []Rule{
					Rule{
						Operations: []admission.Operation{
							admission.Create,
						},
						Type: Send,
					},
				},
			},
			value: nil,
		},
		genericAdmissionWebhookTest{
			test: "Matching rule skips",
			config: &GenericAdmissionWebhookConfig{
				KubeConfigFile: filepath.Join(testRoot, "kube-config"),
				Rules: []Rule{
					Rule{
						Operations: []admission.Operation{
							admission.Update,
						},
						Type: Skip,
					},
				},
			},
			value: nil,
		},
		genericAdmissionWebhookTest{
			test:   "Matching rule sends (webhook internal server error)",
			config: makeGoodConfig(),
			value:  fmt.Errorf(`%san error on the server ("webhook internal server error") has prevented the request from succeeding`, errAdmitPrefix),
		},
		genericAdmissionWebhookTest{
			test:   "Matching rule sends (webhook unsuccessful response)",
			config: makeGoodConfig(),
			value:  fmt.Errorf("%serror contacting webhook: 101", errAdmitPrefix),
		},
		genericAdmissionWebhookTest{
			test:   "Matching rule sends (webhook unmarshallable response)",
			config: makeGoodConfig(),
			value:  fmt.Errorf("%scouldn't get version/kind; json parse error: json: cannot unmarshal string into Go value of type struct.*", errAdmitPrefix),
		},
		genericAdmissionWebhookTest{
			test:   "Matching rule sends (webhook error but allowed via fail action)",
			config: configWithFailAction,
			value:  nil,
		},
		genericAdmissionWebhookTest{
			test:   "Matching rule sends (webhook request denied without reason)",
			config: makeGoodConfig(),
			value:  fmt.Errorf("%swebhook backend denied the request", errAdmitPrefix),
		},
		genericAdmissionWebhookTest{
			test:   "Matching rule sends (webhook request denied with reason)",
			config: makeGoodConfig(),
			value:  fmt.Errorf("%swebhook backend denied the request: you shall not pass", errAdmitPrefix),
		},
		genericAdmissionWebhookTest{
			test:   "Matching rule sends (webhook request allowed)",
			config: makeGoodConfig(),
			value:  nil,
		},
	}

	for _, tt := range tests {
		wh, err := NewGenericAdmissionWebhook(tt.config)

		if err != nil {
			t.Errorf("%s: unexpected error: %v", tt.test, err)
		} else {
			err = wh.Admit(admission.NewAttributesRecord(&object, &oldObject, kind, namespace, name, resource, subResource, operation, &userInfo))

			checkForError(t, tt.test, tt.value, err)
		}
	}
}

func checkForError(t *testing.T, test string, expected, actual interface{}) {
	aErr, _ := actual.(error)
	eErr, _ := expected.(error)

	if eErr != nil {
		if aErr == nil {
			t.Errorf("%s: expected an error", test)
		} else if eErr.Error() != aErr.Error() && !regexp.MustCompile(eErr.Error()).MatchString(aErr.Error()) {
			t.Errorf("%s: unexpected error message to match:\n  Expected: %s\n  Actual:   %s", test, eErr.Error(), aErr.Error())
		}
	} else {
		if aErr != nil {
			t.Errorf("%s: unexpected error: %v", test, aErr)
		}
	}
}

func killTests(err error) {
	panic(fmt.Sprintf("Unable to bootstrap tests: %v", err))
}

func makeGoodConfig() *GenericAdmissionWebhookConfig {
	return &GenericAdmissionWebhookConfig{
		KubeConfigFile: filepath.Join(testRoot, "kube-config"),
		Rules: []Rule{
			Rule{
				Operations: []admission.Operation{
					admission.Update,
				},
				Type: Send,
			},
		},
	}
}

func webhookHandler(w http.ResponseWriter, r *http.Request) {
	requestCount++

	switch requestCount {
	case 1, 4:
		http.Error(w, "webhook internal server error", http.StatusInternalServerError)
		return
	case 2:
		w.WriteHeader(http.StatusSwitchingProtocols)
		w.Write([]byte("webhook invalid request"))
		return
	case 3:
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("webhook invalid response"))
	case 5:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1alpha1.AdmissionReview{
			Status: v1alpha1.AdmissionReviewStatus{
				Allowed: false,
			},
		})
	case 6:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1alpha1.AdmissionReview{
			Status: v1alpha1.AdmissionReviewStatus{
				Allowed: false,
				Result: &metav1.Status{
					Reason: "you shall not pass",
				},
			},
		})
	case 7:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(&v1alpha1.AdmissionReview{
			Status: v1alpha1.AdmissionReviewStatus{
				Allowed: true,
			},
		})
	}
}
