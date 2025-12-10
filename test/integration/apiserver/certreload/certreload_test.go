/*
Copyright 2019 The Kubernetes Authors.

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
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"os"
	"path"
	"strings"
	"testing"
	"time"

	authorizationv1 "k8s.io/api/authorization/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/cert"
	"k8s.io/component-base/cli/flag"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type caWithClient struct {
	CACert     []byte
	ClientCert []byte
	ClientKey  []byte
}

func newTestCAWithClient(caSubject pkix.Name, caSerial *big.Int, clientSubject pkix.Name, subjectSerial *big.Int) (*caWithClient, error) {
	ca := &x509.Certificate{
		SerialNumber:          caSerial,
		Subject:               caSubject,
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(24 * time.Hour),
		IsCA:                  true,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
	}

	caPrivateKey, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		return nil, err
	}

	caBytes, err := x509.CreateCertificate(rand.Reader, ca, ca, &caPrivateKey.PublicKey, caPrivateKey)
	if err != nil {
		return nil, err
	}

	caPEM := new(bytes.Buffer)
	err = pem.Encode(caPEM, &pem.Block{
		Type:  "CERTIFICATE",
		Bytes: caBytes,
	})
	if err != nil {
		return nil, err
	}

	clientCert := &x509.Certificate{
		SerialNumber: subjectSerial,
		Subject:      clientSubject,
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(24 * time.Hour),
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		KeyUsage:     x509.KeyUsageDigitalSignature,
	}

	clientCertPrivateKey, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		return nil, err
	}

	clientCertPrivateKeyPEM := new(bytes.Buffer)
	err = pem.Encode(clientCertPrivateKeyPEM, &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(clientCertPrivateKey),
	})
	if err != nil {
		return nil, err
	}

	clientCertBytes, err := x509.CreateCertificate(rand.Reader, clientCert, ca, &clientCertPrivateKey.PublicKey, caPrivateKey)
	if err != nil {
		return nil, err
	}

	clientCertPEM := new(bytes.Buffer)
	err = pem.Encode(clientCertPEM, &pem.Block{
		Type:  "CERTIFICATE",
		Bytes: clientCertBytes,
	})
	if err != nil {
		return nil, err
	}

	return &caWithClient{
		CACert:     caPEM.Bytes(),
		ClientCert: clientCertPEM.Bytes(),
		ClientKey:  clientCertPrivateKeyPEM.Bytes(),
	}, nil
}

func TestClientCAUpdate(t *testing.T) {
	testClientCA(t, false)
}

func TestClientCARecreate(t *testing.T) {
	testClientCA(t, true)
}

func testClientCA(t *testing.T, recreate bool) {
	tCtx := ktesting.Init(t)

	frontProxyCA, err := newTestCAWithClient(
		pkix.Name{
			CommonName: "test-front-proxy-ca",
		},
		big.NewInt(43),
		pkix.Name{
			CommonName:   "test-aggregated-apiserver",
			Organization: []string{"system:masters"},
		},
		big.NewInt(86),
	)
	if err != nil {
		t.Error(err)
		return
	}

	clientCA, err := newTestCAWithClient(
		pkix.Name{
			CommonName: "test-client-ca",
		},
		big.NewInt(42),
		pkix.Name{
			CommonName:   "system:admin",
			Organization: []string{"system:masters"},
		},
		big.NewInt(84),
	)
	if err != nil {
		t.Error(err)
		return
	}

	clientCAFilename := ""
	frontProxyCAFilename := ""

	kubeClient, kubeconfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.MaxRequestBodyBytes = 1024 * 1024
			clientCAFilename = opts.Authentication.ClientCert.ClientCA
			frontProxyCAFilename = opts.Authentication.RequestHeader.ClientCAFile
			opts.Authentication.RequestHeader.AllowedNames = append(opts.Authentication.RequestHeader.AllowedNames, "test-aggregated-apiserver")
		},
	})
	defer tearDownFn()

	// wait for request header info
	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, waitForConfigMapCAContent(t, kubeClient, "requestheader-client-ca-file", "-----BEGIN CERTIFICATE-----", 1))
	if err != nil {
		t.Fatal(err)
	}
	// wait for client cert info
	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, waitForConfigMapCAContent(t, kubeClient, "client-ca-file", "-----BEGIN CERTIFICATE-----", 1))
	if err != nil {
		t.Fatal(err)
	}

	if recreate {
		if err := os.Remove(path.Join(clientCAFilename)); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(path.Join(frontProxyCAFilename)); err != nil {
			t.Fatal(err)
		}
	}

	// when we run this the second time, we know which one we are expecting
	if err := os.WriteFile(clientCAFilename, clientCA.CACert, 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(frontProxyCAFilename, frontProxyCA.CACert, 0644); err != nil {
		t.Fatal(err)
	}

	time.Sleep(4 * time.Second)

	acceptableCAs, err := cert.GetClientCANamesForURL(kubeconfig.Host)
	if err != nil {
		t.Fatal(err)
	}

	expectedCAs := []string{"test-client-ca", "test-front-proxy-ca"}
	if len(expectedCAs) != len(acceptableCAs) {
		t.Fatal(strings.Join(acceptableCAs, ":"))
	}
	for i := range expectedCAs {
		if !strings.Contains(acceptableCAs[i], expectedCAs[i]) {
			t.Errorf("expected %q, got %q", expectedCAs[i], acceptableCAs[i])
		}
	}

	// wait for updated request header info that contains both
	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, waitForConfigMapCAContent(t, kubeClient, "requestheader-client-ca-file", "-----BEGIN CERTIFICATE-----", 2))
	if err != nil {
		t.Error(err)
	}
	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, waitForConfigMapCAContent(t, kubeClient, "requestheader-client-ca-file", string(frontProxyCA.CACert), 1))
	if err != nil {
		t.Error(err)
	}
	// wait for updated client cert info that contains both
	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, waitForConfigMapCAContent(t, kubeClient, "client-ca-file", "-----BEGIN CERTIFICATE-----", 2))
	if err != nil {
		t.Error(err)
	}
	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, waitForConfigMapCAContent(t, kubeClient, "client-ca-file", string(clientCA.CACert), 1))
	if err != nil {
		t.Error(err)
	}

	// Test an aggregated apiserver client (signed by the new front proxy CA) is authorized
	extensionApiserverClient, err := kubernetes.NewForConfig(&rest.Config{
		Host: kubeconfig.Host,
		TLSClientConfig: rest.TLSClientConfig{
			CAData:     kubeconfig.TLSClientConfig.CAData,
			CAFile:     kubeconfig.TLSClientConfig.CAFile,
			ServerName: kubeconfig.TLSClientConfig.ServerName,
			KeyData:    frontProxyCA.ClientKey,
			CertData:   frontProxyCA.ClientCert,
		},
	})
	if err != nil {
		t.Error(err)
		return
	}

	// Call an endpoint to make sure we are authenticated
	err = extensionApiserverClient.AuthorizationV1().RESTClient().
		Post().
		Resource("subjectaccessreviews").
		VersionedParams(&metav1.CreateOptions{}, scheme.ParameterCodec).
		Body(&authorizationv1.SubjectAccessReview{
			Spec: authorizationv1.SubjectAccessReviewSpec{
				ResourceAttributes: &authorizationv1.ResourceAttributes{
					Verb:      "create",
					Resource:  "pods",
					Namespace: "default",
				},
				User: "deads2k",
			},
		}).
		SetHeader("X-Remote-User", "test-aggregated-apiserver").
		SetHeader("X-Remote-Group", "system:masters").
		Do(context.Background()).
		Into(&authorizationv1.SubjectAccessReview{})
	if err != nil {
		t.Error(err)
	}

	// Test a client signed by the new ClientCA is authorized
	testClient, err := kubernetes.NewForConfig(&rest.Config{
		Host: kubeconfig.Host,
		TLSClientConfig: rest.TLSClientConfig{
			CAData:     kubeconfig.TLSClientConfig.CAData,
			CAFile:     kubeconfig.TLSClientConfig.CAFile,
			ServerName: kubeconfig.TLSClientConfig.ServerName,
			KeyData:    clientCA.ClientKey,
			CertData:   clientCA.ClientCert,
		},
	})
	if err != nil {
		t.Error(err)
		return
	}

	// Call an endpoint to make sure we are authenticated
	_, err = testClient.CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	if err != nil {
		t.Error(err)
	}
}

func waitForConfigMapCAContent(t *testing.T, kubeClient kubernetes.Interface, key, content string, count int) func() (bool, error) {
	return func() (bool, error) {
		clusterAuthInfo, err := kubeClient.CoreV1().ConfigMaps("kube-system").Get(context.TODO(), "extension-apiserver-authentication", metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}

		ca := clusterAuthInfo.Data[key]
		if strings.Count(ca, content) == count {
			return true, nil
		}
		t.Log(ca)
		return false, nil
	}
}

var serverKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA13f50PPWuR/InxLIoJjHdNSG+jVUd25CY7ZL2J023X2BAY+1
M6jkLR6C2nSFZnn58ubiB74/d1g/Fg1Twd419iR615A013f+qOoyFx3LFHxU1S6e
v22fgJ6ntK/+4QD5MwNgOwD8k1jN2WxHqNWn16IF4Tidbv8M9A35YHAdtYDYaOJC
kzjVztzRw1y6bKRakpMXxHylQyWmAKDJ2GSbRTbGtjr7Ji54WBfG43k94tO5X8K4
VGbz/uxrKe1IFMHNOlrjR438dbOXusksx9EIqDA9a42J3qjr5NKSqzCIbgBFl6qu
45V3A7cdRI/sJ2G1aqlWIXh2fAQiaFQAEBrPfwIDAQABAoIBAAZbxgWCjJ2d8H+x
QDZtC8XI18redAWqPU9P++ECkrHqmDoBkalanJEwS1BDDATAKL4gTh9IX/sXoZT3
A7e+5PzEitN9r/GD2wIFF0FTYcDTAnXgEFM52vEivXQ5lV3yd2gn+1kCaHG4typp
ZZv34iIc5+uDjjHOWQWCvA86f8XxX5EfYH+GkjfixTtN2xhWWlfi9vzYeESS4Jbt
tqfH0iEaZ1Bm/qvb8vFgKiuSTOoSpaf+ojAdtPtXDjf1bBtQQG+RSQkP59O/taLM
FCVuRrU8EtdB0+9anwmAP+O2UqjL5izA578lQtdIh13jHtGEgOcnfGNUphK11y9r
Mg5V28ECgYEA9fwI6Xy1Rb9b9irp4bU5Ec99QXa4x2bxld5cDdNOZWJQu9OnaIbg
kw/1SyUkZZCGMmibM/BiWGKWoDf8E+rn/ujGOtd70sR9U0A94XMPqEv7iHxhpZmD
rZuSz4/snYbOWCZQYXFoD/nqOwE7Atnz7yh+Jti0qxBQ9bmkb9o0QW8CgYEA4D3d
okzodg5QQ1y9L0J6jIC6YysoDedveYZMd4Un9bKlZEJev4OwiT4xXmSGBYq/7dzo
OJOvN6qgPfibr27mSB8NkAk6jL/VdJf3thWxNYmjF4E3paLJ24X31aSipN1Ta6K3
KKQUQRvixVoI1q+8WHAubBDEqvFnNYRHD+AjKvECgYBkekjhpvEcxme4DBtw+OeQ
4OJXJTmhKemwwB12AERboWc88d3GEqIVMEWQJmHRotFOMfCDrMNfOxYv5+5t7FxL
gaXHT1Hi7CQNJ4afWrKgmjjqrXPtguGIvq2fXzjVt8T9uNjIlNxe+kS1SXFjXsgH
ftDY6VgTMB0B4ozKq6UAvQKBgQDER8K5buJHe+3rmMCMHn+Qfpkndr4ftYXQ9Kn4
MFiy6sV0hdfTgRzEdOjXu9vH/BRVy3iFFVhYvIR42iTEIal2VaAUhM94Je5cmSyd
eE1eFHTqfRPNazmPaqttmSc4cfa0D4CNFVoZR6RupIl6Cect7jvkIaVUD+wMXxWo
osOFsQKBgDLwVhZWoQ13RV/jfQxS3veBUnHJwQJ7gKlL1XZ16mpfEOOVnJF7Es8j
TIIXXYhgSy/XshUbsgXQ+YGliye/rXSCTXHBXvWShOqxEMgeMYMRkcm8ZLp/DH7C
kC2pemkLPUJqgSh1PASGcJbDJIvFGUfP69tUCYpHpk3nHzexuAg3
-----END RSA PRIVATE KEY-----`)

var serverCert = []byte(`-----BEGIN CERTIFICATE-----
MIIDQDCCAiigAwIBAgIJANWw74P5KJk2MA0GCSqGSIb3DQEBCwUAMDQxMjAwBgNV
BAMMKWdlbmVyaWNfd2ViaG9va19hZG1pc3Npb25fcGx1Z2luX3Rlc3RzX2NhMCAX
DTE3MTExNjAwMDUzOVoYDzIyOTEwOTAxMDAwNTM5WjAjMSEwHwYDVQQDExh3ZWJo
b29rLXRlc3QuZGVmYXVsdC5zdmMwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK
AoIBAQDXd/nQ89a5H8ifEsigmMd01Ib6NVR3bkJjtkvYnTbdfYEBj7UzqOQtHoLa
dIVmefny5uIHvj93WD8WDVPB3jX2JHrXkDTXd/6o6jIXHcsUfFTVLp6/bZ+Anqe0
r/7hAPkzA2A7APyTWM3ZbEeo1afXogXhOJ1u/wz0DflgcB21gNho4kKTONXO3NHD
XLpspFqSkxfEfKVDJaYAoMnYZJtFNsa2OvsmLnhYF8bjeT3i07lfwrhUZvP+7Gsp
7UgUwc06WuNHjfx1s5e6ySzH0QioMD1rjYneqOvk0pKrMIhuAEWXqq7jlXcDtx1E
j+wnYbVqqVYheHZ8BCJoVAAQGs9/AgMBAAGjZDBiMAkGA1UdEwQCMAAwCwYDVR0P
BAQDAgXgMB0GA1UdJQQWMBQGCCsGAQUFBwMCBggrBgEFBQcDATApBgNVHREEIjAg
hwR/AAABghh3ZWJob29rLXRlc3QuZGVmYXVsdC5zdmMwDQYJKoZIhvcNAQELBQAD
ggEBAD/GKSPNyQuAOw/jsYZesb+RMedbkzs18sSwlxAJQMUrrXwlVdHrA8q5WhE6
ABLqU1b8lQ8AWun07R8k5tqTmNvCARrAPRUqls/ryER+3Y9YEcxEaTc3jKNZFLbc
T6YtcnkdhxsiO136wtiuatpYL91RgCmuSpR8+7jEHhuFU01iaASu7ypFrUzrKHTF
bKwiLRQi1cMzVcLErq5CDEKiKhUkoDucyARFszrGt9vNIl/YCcBOkcNvM3c05Hn3
M++C29JwS3Hwbubg6WO3wjFjoEhpCwU6qRYUz3MRp4tHO4kxKXx+oQnUiFnR7vW0
YkNtGc1RUDHwecCTFpJtPb7Yu/E=
-----END CERTIFICATE-----`)

var anotherServerKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIJKAIBAAKCAgEAlZJORzCjbzF1SaCXFHyitudlE+q3Z5bGhS2TpXG6d6Laoqtw
No4UC3i+TnMndtrP2pNkV/ZYivsp1fHz92FqFAT+XpYcG8pLm4iL0k0UufOWdLPT
X87HCJjKZ4r7fmzstyjqK4sv9I3ye1jKi0VE1BLrF1KVvEE/1PXCug68EBP/aF06
+uvcr6o8hbMYzgdKSzhRYm9C3kGcawNofqAD/Kk/zn+pMk4Bloy4UgtXFXgj2bEn
mVE+tRWyLv2+TONlmLnXaBW3/MvZtKC3mIs2KG+6aBNuY8PdWzWvMtp30r/ibgnH
zuMKtvXJ5XRhTaST4QYXNbGwb1bIV1ylnX8zdXPEQkuYTQDctaYQCe0RXt1I9Fp3
gVQRxyTM+0IetbsU0k9VvBwQ07mgU8Rik3DxVnfbuJY/wREnERTkgv6ojtRwiszr
GIY5x36peRs30CqRMv3uJtqC/FU6nCQbHxwssQyB/umN6L7bcpsQFDydeK95hvRQ
y6tb2v/vMcw7MMo5kSFUHjoL5Zc4DObwiqs+p7F7S0WIJMBzJOcjmgCMzgZ7Jmc7
bMmrm43GLzOaVLIjuPVVpOp7YgJ/lqRf7K3hZXrMdaXkCm01aL8L59d+3Vfdjp3H
HvmYpCh8bc+Kjs/nR9Rc+2JKK/H13LH3W5Cr8Fnc/FP6TgbvvNwsQV01gG8CAwEA
AQKCAgBLBQn8DPo8YDsqxcBhRy45vQ/mkHiTHX3O+JAwkD1tmiI9Ku3qfxKwukwB
fyKRK6jLQdg3gljgxJ80Ltol/xc8mVCYUoQgsDOB/FfdEEpQBkw1lqhzSnxr5G7I
xl3kCHAmYgAp/PL9n2C620sj1YdzM1X06bgupy+D+gxEU/WhvtYBG5nklv6moSUg
DjdnxyJNXh7710Bbx97Tke8Ma+f0B1P4l/FeSN/lCgm9JPD11L9uhbuN28EvBIXN
qfmUCQ5BLx1KmHIi+n/kaCQN/+0XFQsS/oQEyA2znNaWFBu7egDxHji4nQoXwGoW
i2vujJibafmkNc5/2bA8mTx8JXvCLhU2L9j2ZumpKOda0g+pfMauesL+9rvZdqwW
gjdjndOHZlg3qm40hGCDBVmmV3mdnvXrk1BbuB4Y0N7qGo3PyYtJHGwJILaNQVGR
Sj75uTatxJwFXsqSaJaErV3Q90IiyXX4AOFGnWHOs29GEwtnDbCvT/rzqutTYSXD
Yv0XFDznzJelhZTH7FbaW3FW3YGEG1ER/0MtKpsAH4i7H9q3KKK8yrzUsgUkGwXt
xtoLckh91xilPIGbzARdELTEdHrjlFL+qaz3PIqEQScWz3WBu2JcIzGbp6PQfMZ+
FZXarEb/ADZuX0+WoKFYR5jzwMoQfF/fxe2Ib/37ETNw4BgfSQKCAQEAxOw64XgO
nUVJslzGK/H5fqTVpD1rfRmvVAiSDLAuWpClbpDZXqEPuoPPYsiccuUWu9VkJE1F
6MZEexGx1jFkN08QUHD1Bobzu6ThaBc2PrWHRjFGKM60d0AkhOiL4N04FGwVeCN6
xzIJFk1E4VOOo1+lzeAWRvi1lwuWTgQi+m25nwBJtmYdBLGeS+DXy80Fi6deECei
ipDzJ4rxJsZ61uqBeYC4CfuHW9m5rCzJWPMMMFrPdl3OxEyZzKng4Co5EYc5i/QH
piXD6IJayKcTPRK3tBJZp2YCIIdtQLcjAwmDEDowQtelHkbTihXMGRarf3VcOEoN
ozMRgcLEEynuKwKCAQEAwnF5ZkkJEL/1MCOZ6PZfSKl35ZMIz/4Umk8hOMAQGhCT
cnxlDUfGSBu4OihdBbIuBSBsYDjgcev8uyiIPDVy0FIkBKRGfgrNCLDh19aHljvE
bUc3akvbft0mro86AvSd/Rpc7sj841bru37RDUm6AJOtIvb6DWUpMOZgMm0WMmSI
kNs/UT+7rqg+AZPP8lumnJIFnRK38xOehQAaS1FHWGP//38py8yo8eXpMsoCWMch
c+kZD2jsAYV+SWjjkZjcrv/52+asd4AotRXIShV8E8xItQeq6vLHKOaIe0tC2Y44
ONAKiu4dgABt1voy8I5J63MwgeNmgAUS+KsgUclYzQKCAQEAlt/3bPAzIkQH5uQ1
4U2PvnxEQ4XbaQnYzyWR4K7LlQ/l8ASCxoHYLyr2JdVWKKFk/ZzNERMzUNk3dqNk
AZvuEII/GaKx2MJk04vMN5gxM3KZpinyeymEEynN0RbqtOpJITx+ZoGofB3V4IRr
FciTLJEH0+iwqMe9OXDjQ/rfYcfXw/7QezNZYFNF2RT3wWnfqdQduXrkig3sfotx
oCfJzgf2E0WPu/Y/CxyRqVzXF5N/7zxkX2gYF0YpQCmX5afz+X4FlTju81lT9DyL
mdiIYO6KWSkGD7+UOaAJEOA/rwAGrtQmTdAy7jONt+pjaYV4+DrO4UG7mSJzc1vq
JlSl6QKCAQARqwPv8mT7e6XI2QNMMs7XqGZ3mtOrKpguqVAIexM7exQazAjWmxX+
SV6FElPZh6Y82wRd/e0PDPVrADTY27ZyDXSuY0rwewTEbGYpGZo6YXXoxBbZ9sic
D3ZLWEJaMGYGsJWPMP4hni1PXSebwH5BPSn3Sl/QRcfnZJeLHXRt4cqy9uka9eKU
7T6tIAQ+LmvGQFJ4QlIqqTa3ORoqi9kiw/tn+OMQXKlhSZXWApsR/A4jHSQkzVDc
loeyHfDHsw8ia6oFfEFhnmiUg8UuTiN3HRHiOS8jqCnGoqP2KBGL+StMpkK++wH9
NozEgvmL+DHpTg8zTjlrGortw4btR5FlAoIBABVni+EsGA5K/PM1gIct2pDm+6Kq
UCYScTwIjftuwKLk/KqermG9QJLiJouKO3ZSz7iCelu87Dx1cKeXrc2LQ1pnQzCB
JnI6BCT+zRnQFXjLokJXD2hIS2hXhqV6/9FRXLKKMYePcDxWt/etLNGmpLnhDfb3
sMOH/9pnaGmtk36Ce03Hh7E1C6io/MKfTq+KKUV1UGwO1BdNQCiclkYzAUqn1O+Y
c8BaeGKc2c6as8DKrPTGGQGmzo/ZUxQVfVFl2g7+HXISWBBcui/G5gtnU1afZqbW
mTmDoqs4510vhlkhN9XZ0DyhewDIqNNGEY2vS1x2fJz1XC2Eve4KpSyUsiE=
-----END RSA PRIVATE KEY-----
`)

var anotherServerCert = []byte(`-----BEGIN CERTIFICATE-----
MIIFJjCCAw6gAwIBAgIJAOcEAbv8NslfMA0GCSqGSIb3DQEBCwUAMEAxCzAJBgNV
BAYTAlVTMQswCQYDVQQIDAJDQTETMBEGA1UECgwKQWNtZSwgSW5jLjEPMA0GA1UE
AwwGc29tZUNBMCAXDTE4MDYwODEzMzkyNFoYDzIyMTgwNDIxMTMzOTI0WjBDMQsw
CQYDVQQGEwJVUzELMAkGA1UECAwCQ0ExEzARBgNVBAoMCkFjbWUsIEluYy4xEjAQ
BgNVBAMMCWxvY2FsaG9zdDCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIB
AJWSTkcwo28xdUmglxR8orbnZRPqt2eWxoUtk6Vxunei2qKrcDaOFAt4vk5zJ3ba
z9qTZFf2WIr7KdXx8/dhahQE/l6WHBvKS5uIi9JNFLnzlnSz01/OxwiYymeK+35s
7Lco6iuLL/SN8ntYyotFRNQS6xdSlbxBP9T1wroOvBAT/2hdOvrr3K+qPIWzGM4H
Sks4UWJvQt5BnGsDaH6gA/ypP85/qTJOAZaMuFILVxV4I9mxJ5lRPrUVsi79vkzj
ZZi512gVt/zL2bSgt5iLNihvumgTbmPD3Vs1rzLad9K/4m4Jx87jCrb1yeV0YU2k
k+EGFzWxsG9WyFdcpZ1/M3VzxEJLmE0A3LWmEAntEV7dSPRad4FUEcckzPtCHrW7
FNJPVbwcENO5oFPEYpNw8VZ327iWP8ERJxEU5IL+qI7UcIrM6xiGOcd+qXkbN9Aq
kTL97ibagvxVOpwkGx8cLLEMgf7pjei+23KbEBQ8nXiveYb0UMurW9r/7zHMOzDK
OZEhVB46C+WXOAzm8IqrPqexe0tFiCTAcyTnI5oAjM4GeyZnO2zJq5uNxi8zmlSy
I7j1VaTqe2ICf5akX+yt4WV6zHWl5AptNWi/C+fXft1X3Y6dxx75mKQofG3Pio7P
50fUXPtiSivx9dyx91uQq/BZ3PxT+k4G77zcLEFdNYBvAgMBAAGjHjAcMBoGA1Ud
EQQTMBGCCWxvY2FsaG9zdIcEfwAAATANBgkqhkiG9w0BAQsFAAOCAgEABL8kffi7
48qSD+/l/UwCYdmqta1vAbOkvLnPtfXe1XlDpJipNuPxUBc8nNTemtrbg0erNJnC
jQHodqmdKBJJOdaEKTwAGp5pYvvjlU3WasmhfJy+QwOWgeqjJcTUo3+DEaHRls16
AZXlsp3hB6z0gzR/qzUuZwpMbL477JpuZtAcwLYeVvLG8bQRyWyEy8JgGDoYSn8s
Z16s+r6AX+cnL/2GHkZ+oc3iuXJbnac4xfWTKDiYnyzK6RWRnoyro7X0jiPz6XX3
wyoWzB1uMSCXscrW6ZcKyKqz75lySLuwGxOMhX4nGOoYHY0ZtrYn5WK2ZAJxsQnn
8QcjPB0nq37U7ifk1uebmuXe99iqyKnWaLvlcpe+HnO5pVxFkSQEf7Zh+hEnRDkN
IBzLFnqwDS1ug/oQ1aSvc8oBh2ylKDJuGtPNqGKibNJyb2diXO/aEUOKRUKPAxKa
dbKsc4Y1bhZNN3/MICMoyghwAOiuwUQMR5uhxTkQmZUwNrPFa+eW6GvyoYLFUsZs
hZfWLNGD5mLADElxs0HF7F9Zk6pSocTDXba4d4lfxsq88SyZZ7PbjJYFRfLQPzd1
CfvpRPqolEmZo1Y5Q644PELYiJRKpBxmX5GtC5j5eaUD9XdGKvXsGhb0m0gW75rq
iUnnLkZt2ya1cDJDiCnJjo7r5KxMo0XXFDc=
-----END CERTIFICATE-----
`)

func TestServingCertUpdate(t *testing.T) {
	testServingCert(t, false)
}

func TestServingCertRecreate(t *testing.T) {
	testServingCert(t, true)
}

func testServingCert(t *testing.T, recreate bool) {
	tCtx := ktesting.Init(t)

	var servingCertPath string

	_, kubeconfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.MaxRequestBodyBytes = 1024 * 1024
			servingCertPath = opts.SecureServing.ServerCert.CertDirectory
		},
	})
	defer tearDownFn()

	if recreate {
		if err := os.Remove(path.Join(servingCertPath, "apiserver.key")); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(path.Join(servingCertPath, "apiserver.crt")); err != nil {
			t.Fatal(err)
		}
	}

	if err := os.WriteFile(path.Join(servingCertPath, "apiserver.key"), serverKey, 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path.Join(servingCertPath, "apiserver.crt"), serverCert, 0644); err != nil {
		t.Fatal(err)
	}

	time.Sleep(4 * time.Second)

	// get the certs we're actually serving with
	_, actualCerts, err := cert.GetServingCertificatesForURL(kubeconfig.Host, "")
	if err != nil {
		t.Fatal(err)
	}
	if err := checkServingCerts(serverCert, actualCerts); err != nil {
		t.Fatal(err)
	}
}

func TestSNICert(t *testing.T) {
	var servingCertPath string

	tCtx := ktesting.Init(t)

	_, kubeconfig, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.MaxRequestBodyBytes = 1024 * 1024
			servingCertPath = opts.SecureServing.ServerCert.CertDirectory

			if err := os.WriteFile(path.Join(servingCertPath, "foo.key"), anotherServerKey, 0644); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path.Join(servingCertPath, "foo.crt"), anotherServerCert, 0644); err != nil {
				t.Fatal(err)
			}

			opts.SecureServing.SNICertKeys = []flag.NamedCertKey{{
				Names:    []string{"foo"},
				CertFile: path.Join(servingCertPath, "foo.crt"),
				KeyFile:  path.Join(servingCertPath, "foo.key"),
			}}
		},
	})
	defer tearDownFn()

	// When we run this the second time, we know which one we are expecting.
	_, actualCerts, err := cert.GetServingCertificatesForURL(kubeconfig.Host, "foo")
	if err != nil {
		t.Fatal(err)
	}
	if err := checkServingCerts(anotherServerCert, actualCerts); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(path.Join(servingCertPath, "foo.key"), serverKey, 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path.Join(servingCertPath, "foo.crt"), serverCert, 0644); err != nil {
		t.Fatal(err)
	}

	time.Sleep(4 * time.Second)

	_, actualCerts, err = cert.GetServingCertificatesForURL(kubeconfig.Host, "foo")
	if err != nil {
		t.Fatal(err)
	}
	if err := checkServingCerts(serverCert, actualCerts); err != nil {
		t.Fatal(err)
	}
}

func checkServingCerts(expectedBytes []byte, actual [][]byte) error {
	expectedCerts, err := cert.ParseCertsPEM(expectedBytes)
	if err != nil {
		return err
	}
	expected := [][]byte{}
	for _, curr := range expectedCerts {
		currBytes, err := cert.EncodeCertificates(curr)
		if err != nil {
			return err
		}
		expected = append(expected, []byte(strings.TrimSpace(string(currBytes))))
	}

	if len(expected) != len(actual) {
		var certs []string
		for _, a := range actual {
			certs = append(certs, string(a))
		}
		return fmt.Errorf("unexpected number of certs %d vs %d: %v", len(expected), len(actual), strings.Join(certs, "\n"))
	}
	for i := range expected {
		if !bytes.Equal(actual[i], expected[i]) {
			return fmt.Errorf("expected %q, got %q", string(expected[i]), string(actual[i]))
		}
	}
	return nil
}
