/*
Copyright 2018 The Kubernetes Authors.

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

package podlogs

import (
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestInsecurePodLogs(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)
	clientSet, _ := framework.StartTestServer(t, stopCh, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.MaxRequestBodyBytes = 1024 * 1024
			// I have no idea what this cert is, but it doesn't matter, we just want something that always fails validation
			opts.KubeletConfig.CAData = []byte(`      -----BEGIN CERTIFICATE-----
      MIIDMDCCAhigAwIBAgIIHNPD7sig7YIwDQYJKoZIhvcNAQELBQAwNjESMBAGA1UE
      CxMJb3BlbnNoaWZ0MSAwHgYDVQQDExdhZG1pbi1rdWJlY29uZmlnLXNpZ25lcjAe
      Fw0xOTA1MzAxNTA3MzlaFw0yOTA1MjcxNTA3MzlaMDYxEjAQBgNVBAsTCW9wZW5z
      aGlmdDEgMB4GA1UEAxMXYWRtaW4ta3ViZWNvbmZpZy1zaWduZXIwggEiMA0GCSqG
      SIb3DQEBAQUAA4IBDwAwggEKAoIBAQD0dHk23lHRcuq06FzYDOl9J9+s8pnGxqA3
      IPcARI6ag/98aYe3ENwAB5e1i7AU2F2WiDZgj444w374XLdVgIK8zgQEm9yoqrlc
      +/ayO7ceKklrKHOMwh63LvGLEOqzhol2nFmBhXAZt+HyIoZHXN0IqlA92196+Dml
      0WOn1F4ce6JbAtEceFHPgLeI7KFmVaPz2796pBXh23ii6r7WvV1Rn9MKlMSBJQR4
      0LZzu9/j+GdnFXewdLAAMfgPzwEqv6h3PzvtUCjgdraHEm8Rs7s15S3PUmLK4RQS
      PsThx5BhJEGd/W6EzQ3BKoQfochhu3mnAQtW1J07CullySQ5Gg9fAgMBAAGjQjBA
      MA4GA1UdDwEB/wQEAwICpDAPBgNVHRMBAf8EBTADAQH/MB0GA1UdDgQWBBQkTaaw
      YJSZ5k2Wd+OsM4GFMTGdqzANBgkqhkiG9w0BAQsFAAOCAQEAHK7+zBZPLqK+f9DT
      UEnpwRmZ0aeGS4YgbGIkqpjxJymVOwkRd5A1wslvVfGZ6yOQthF6KlCmqnPyJJMR
      I7FHw8j0h2ci90fEQ6IS90Y/ZJXkcgiK9Ncwa35GFGs8QrBxN4leGhtm84BnnBHN
      cTWpa4zcBwru0CRG7iHc66VX16X8jHB1iFeZ5W/FgY4MsE+G1Vze4mCXSPVI4BZ2
      /qlAgogjBivvSwQ9SFuCszg7IPjvT2ksm+Cf+8eT4YBqW41F85vBGR+FYK14yIla
      Bgqc+dJN9xS9Ah5gLiGQJ6C4niUA11piCpvMsy+j/LQ1Erx47KMar5fuMXYk7iPq
      1vqIwg==
      -----END CERTIFICATE-----
`)
		},
	})

	fakeKubeletServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Write([]byte("fake-log"))
		w.WriteHeader(http.StatusOK)
	}))
	defer fakeKubeletServer.Close()

	fakeKubeletURL, err := url.Parse(fakeKubeletServer.URL)
	if err != nil {
		t.Fatal(err)
	}
	fakeKubeletHost, fakeKubeletPortStr, err := net.SplitHostPort(fakeKubeletURL.Host)
	if err != nil {
		t.Fatal(err)
	}
	fakeKubeletPort, err := strconv.ParseUint(fakeKubeletPortStr, 10, 32)
	if err != nil {
		t.Fatal(err)
	}

	node, err := clientSet.CoreV1().Nodes().Create(&corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "fake"},
	})
	if err != nil {
		t.Fatal(err)
	}
	node.Status = corev1.NodeStatus{
		Addresses: []corev1.NodeAddress{
			{
				Type:    corev1.NodeExternalIP,
				Address: fakeKubeletHost,
			},
		},
		DaemonEndpoints: corev1.NodeDaemonEndpoints{
			KubeletEndpoint: corev1.DaemonEndpoint{
				Port: int32(fakeKubeletPort),
			},
		},
	}
	node, err = clientSet.CoreV1().Nodes().UpdateStatus(node)
	if err != nil {
		t.Fatal(err)
	}

	_, err = clientSet.CoreV1().Namespaces().Create(&corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "ns"},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = clientSet.CoreV1().ServiceAccounts("ns").Create(&corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: "ns"},
	})
	if err != nil {
		t.Fatal(err)
	}

	falseRef := false
	pod, err := clientSet.CoreV1().Pods("ns").Create(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "ns"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "foo",
					Image: "some/image:latest",
				},
			},
			NodeName:                     node.Name,
			AutomountServiceAccountToken: &falseRef,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	insecureResult := clientSet.CoreV1().Pods("ns").GetLogs(pod.Name, &corev1.PodLogOptions{InsecureSkipTLSVerifyBackend: true}).Do()
	if err := insecureResult.Error(); err != nil {
		t.Fatal(err)
	}
	insecureStatusCode := 0
	insecureResult.StatusCode(&insecureStatusCode)
	if insecureStatusCode != http.StatusOK {
		t.Fatal(insecureStatusCode)
	}

	secureResult := clientSet.CoreV1().Pods("ns").GetLogs(pod.Name, &corev1.PodLogOptions{}).Do()
	if err := secureResult.Error(); err == nil || !strings.Contains(err.Error(), "x509: certificate signed by unknown authority") {
		t.Fatal(err)
	}
	secureStatusCode := 0
	secureResult.StatusCode(&secureStatusCode)
	if secureStatusCode == http.StatusOK {
		raw, rawErr := secureResult.Raw()
		if rawErr != nil {
			t.Log(rawErr)
		}
		t.Log(string(raw))
		t.Fatal(secureStatusCode)
	}

}
