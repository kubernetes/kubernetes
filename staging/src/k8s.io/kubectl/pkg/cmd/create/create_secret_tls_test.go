/*
Copyright 2021 The Kubernetes Authors.

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

package create

import (
	"fmt"
	"os"
	"path"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utiltesting "k8s.io/client-go/util/testing"
)

var rsaSelfSignedCertPEM = `
-----BEGIN CERTIFICATE-----
MIIDZTCCAk2gAwIBAgIUEARJE682DlpLrwXr32hkJo2OHOowDQYJKoZIhvcNAQEL
BQAwQjELMAkGA1UEBhMCVVMxFTATBgNVBAcMDERlZmF1bHQgQ2l0eTEcMBoGA1UE
CgwTRGVmYXVsdCBDb21wYW55IEx0ZDAeFw0yMTA0MTIxMTEyMDlaFw0yMTA1MTIx
MTEyMDlaMEIxCzAJBgNVBAYTAlVTMRUwEwYDVQQHDAxEZWZhdWx0IENpdHkxHDAa
BgNVBAoME0RlZmF1bHQgQ29tcGFueSBMdGQwggEiMA0GCSqGSIb3DQEBAQUAA4IB
DwAwggEKAoIBAQDQOIOlz+GhLxwigsBBj6ZXOB6DNK9DACmmw0pz3M+U0o4+PI85
8ae3q2eizvjMwCHgvQmh82w9kaI2NehnXCygG4qi7MTRNj+UnsrP5haTc5FyucYl
GUADD9MUuyR9qZwkAt+PY4QmRotWnBlKLD/I+rXBVVv1KveJUkxoBLGk42kpMdS7
RT06vmpGVHjq9HikrRvicdFbUfm4YODvFMNNStnoInZJmmGxumnGxhNkO+n6mswk
3/Je5QEuZ8S2yIGkMXVOCUzAeScbI+NGiursYx5OPjN0doR4xNEHYIC53ATDBaK3
z3Hxhp2tYNPDbZvGnFPsjcFAiXspYjViQDVJAgMBAAGjUzBRMB0GA1UdDgQWBBTR
a1tRtnbp9ZQruY5RSvmzP/duiTAfBgNVHSMEGDAWgBTRa1tRtnbp9ZQruY5RSvmz
P/duiTAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQCYtfHoz7Y8
evmSGUqfOay7DCyL2tFLdWJdki4YG0NDquA9QKZ8jfl0epUnlX+UaaxT1fswAd0H
YO0V48MCADlx2xC/pDmSzIS8chPbipqpGZuHMl+LseRaltZbnyMf7VSrK0EbW0xh
bKbZDPR3lSkAQaCmYqlzauY0ZJa3bHZrhzem0wMUdwTFH03AJASqq3PzlGEOcuqa
6Z0me1WVcR9oHwfbfOEF3DiQinFhRyG/DtCD2oYbbaO9e9VP0+1Hy05YkTN572Gw
9jF5Z4wn5rFZJNglVoDiTEPwUyt+iXdGRPvQ7ftaTmK+jfbwxNbMMjehOi2y1nCW
GIvEgkp0W7eG
-----END CERTIFICATE-----
`

var rsaCertPEM = `-----BEGIN CERTIFICATE-----
MIIDCzCCAfMCFAIsO+psszOfir2J0i6dqWN6RnBQMA0GCSqGSIb3DQEBCwUAMEIx
CzAJBgNVBAYTAlVTMRUwEwYDVQQHDAxEZWZhdWx0IENpdHkxHDAaBgNVBAoME0Rl
ZmF1bHQgQ29tcGFueSBMdGQwHhcNMjEwNDEyMTAzOTU5WhcNMjEwNTEyMTAzOTU5
WjBCMQswCQYDVQQGEwJVUzEVMBMGA1UEBwwMRGVmYXVsdCBDaXR5MRwwGgYDVQQK
DBNEZWZhdWx0IENvbXBhbnkgTHRkMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB
CgKCAQEA0DiDpc/hoS8cIoLAQY+mVzgegzSvQwAppsNKc9zPlNKOPjyPOfGnt6tn
os74zMAh4L0JofNsPZGiNjXoZ1wsoBuKouzE0TY/lJ7Kz+YWk3ORcrnGJRlAAw/T
FLskfamcJALfj2OEJkaLVpwZSiw/yPq1wVVb9Sr3iVJMaASxpONpKTHUu0U9Or5q
RlR46vR4pK0b4nHRW1H5uGDg7xTDTUrZ6CJ2SZphsbppxsYTZDvp+prMJN/yXuUB
LmfEtsiBpDF1TglMwHknGyPjRorq7GMeTj4zdHaEeMTRB2CAudwEwwWit89x8Yad
rWDTw22bxpxT7I3BQIl7KWI1YkA1SQIDAQABMA0GCSqGSIb3DQEBCwUAA4IBAQAc
+FBQUbp1RE2ysq7vZaLmZtEvwm0D55J7QQByjLh6opiPwS1hj+rYAOMpMhC97oQF
8saheoYocMhS6jL100tuA8n7MFVa4oFQSKn/S5bP9gecyTTikSbu8sRenV/959Xd
l4Eo75Qq9eNv8AKfHzcMEBM8rhrQEwRsVlDN3c0jVsq/J3kPy+JEs3tcsVPS3ra3
5ZDs53CSKQ6RHr6eagr/uQeoYn+NGx7bf88puZQhpg/S85H2am3e0vmesZ4xIeeD
uF+xcXBtg6i8/Gm+kBzYYycYhlo8u5W26KByIM/qMj/cFYNuq4aiQ6SiTWAmutqz
aHa9ghdQ169u3ScCJw6M
-----END CERTIFICATE-----
`

var rsaKeyPEM = `-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0DiDpc/hoS8cIoLAQY+mVzgegzSvQwAppsNKc9zPlNKOPjyP
OfGnt6tnos74zMAh4L0JofNsPZGiNjXoZ1wsoBuKouzE0TY/lJ7Kz+YWk3ORcrnG
JRlAAw/TFLskfamcJALfj2OEJkaLVpwZSiw/yPq1wVVb9Sr3iVJMaASxpONpKTHU
u0U9Or5qRlR46vR4pK0b4nHRW1H5uGDg7xTDTUrZ6CJ2SZphsbppxsYTZDvp+prM
JN/yXuUBLmfEtsiBpDF1TglMwHknGyPjRorq7GMeTj4zdHaEeMTRB2CAudwEwwWi
t89x8YadrWDTw22bxpxT7I3BQIl7KWI1YkA1SQIDAQABAoIBAQC6eSJNKLLkaxBD
R91t2XwauEN7NX+P/WFP262dva6ZlUeWLR4Hwod1UafqgnkGxTqRzjoGM75IFVi0
O+r6Re4hJQkvh+Nria2/J8ZyEZk+TE2B9SWiq85L76wV8NNpRrBy/6++9lyu+pZm
5j0v1Bj7oKuNjShhlC1DadTLgwikpPWt6mu+1sWIrrN2wLvIppZKX31T5oBolQDL
wk/iVbKD85RphxzkPlfiKYuPHfTSHmYNlMkFcJi90F3NQQUmaWWf0ZoEoK58mYNM
jcEomhsovtGipuB7+cTOPwjGR4YtH5av5IjEaGXd8WZDL/Sg4zyluDf1v56EddhL
F8AavIMRAoGBAPw54NF32nvxRvJK+kqLv/nHaFG3suqKqF/url07oAMPEzv8n4iE
d2liwXOAe+AYdJFHbPPI5mHRJQzcQPlNzW+2HcCC8XQZOma5xvCrxmLojzTvJEN5
uwfGgZNzmIAE4CA0ouXNBUoroG1QWSy9ZazBeqFmEWGL4pGG4XKN2QrDAoGBANNW
FDPh9ue+GVLvfMjyReUVCf6dh8SJJBuOMIA56cybFBxRmJHqE0RgEHkbWJykB3Iw
qb4TCg9dkuPy04JuhI1lrfttMvIEAslbRPxHAf+IMIt7kpZXeIoc7zPagpi4TwLd
KY+sRZ4xchzxO4poItkD450TCnGWqLTWBhg400cDAoGAewJXLJFBUtUW/q+mZZjG
ZbDkpYXrkgtRloe3Le0YWqWNgeHwhAnmmtT497WftGj44Klu7235PZdcdGsunOde
26570BmMXEy5eMP9y/5aYH5+6RgAHZBOsLoVE656n2TBUbOaBmz4uXWRZf6bnwA3
iAtMHU7EB0jLlKGtbcrUITUCgYARh4ZNd2S/fCklk+/Jyy64/bHCiNaGGsn/7x9e
w279Ja/ZWXtKPxwyA7XaFcaX15M2iYrK1VF0TNKuTan1m60q/VAdFsWvBV4lzYg/
VLR5uZYtO6bBCahZ7GR67JkAiekj16xm2mc74+YPOIMzy8d4MLZkhPvMyC5eMZJ3
197OeQKBgQCaEqwxBbtNdAjKZwNR9K7M1ubh0sUA4DU1xOEFgX7LMGKIUNFruB56
nTmq79qCAMyoqjD+kKZTBK0G9s9hB8j7WIzUBCV132uGhBdvAAPzcksTV1UId4Bt
n/luzREtgnfV/zjZHiO1brOc1LcPLUyAnbQFFIz5rvU1CjRNEumC3w==
-----END RSA PRIVATE KEY-----
`

var rsaCAPEM = `-----BEGIN CERTIFICATE-----
MIIDZTCCAk2gAwIBAgIUA8ZO+ysA12hPno7jbTMKg6Kvog0wDQYJKoZIhvcNAQEL
BQAwQjELMAkGA1UEBhMCVVMxFTATBgNVBAcMDERlZmF1bHQgQ2l0eTEcMBoGA1UE
CgwTRGVmYXVsdCBDb21wYW55IEx0ZDAeFw0yMTA0MTIxMDM5NDdaFw0yMTA1MTIx
MDM5NDdaMEIxCzAJBgNVBAYTAlVTMRUwEwYDVQQHDAxEZWZhdWx0IENpdHkxHDAa
BgNVBAoME0RlZmF1bHQgQ29tcGFueSBMdGQwggEiMA0GCSqGSIb3DQEBAQUAA4IB
DwAwggEKAoIBAQDpAvbQ5YJ7Cy+WTS0+B6KKea1ENM7+1yDTSojMO/8KXqByQJMi
BDIzHfCp2gxzM69A3xMy0p9dIAmk6xOFoh9jN/z/K8dsD8I4gDpa5QrAf3pgVaoL
3YIdP3ZZmLlsl6MbYsGKBVm50JibY5hOE+kAeP72oSAiBP2nNEYshXlHqV3cicd9
tHz+bY1jwwNIwHtAV+sNxb7Gyck4jQGijc/4aKZysojpeboTrXfFP0MP269Alzkq
UfsK0ep7bwEN4Ym+bsQ9toQ9t7ADckveblWWQ1xLRwA5AWq63ro/ttkTVgn7Ppiw
tx+hPHTb6tXQ0QVriri3VF8Q6si4/oNOLfzxAgMBAAGjUzBRMB0GA1UdDgQWBBQq
SCV2WXyha5UmBTJJx0rOMtf3SzAfBgNVHSMEGDAWgBQqSCV2WXyha5UmBTJJx0rO
Mtf3SzAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQCk1R/VV62t
a3sNFgKErNtTxxyi/VaWHP5mSf/ggKVFnVjQawK7RuLSD2jYygU9VFMqJ/BQRbZM
eh0anTmY06RiDtdvLr/s56hXVwtuHIoo2mTaFgggBkL7HJo68i11riB9yXhlEyKg
avPAfDRmAOmADVLzeNug8CcYTtEgXjhEKnBw7cBcFWxZFUtWIGCyHzRReD2yjzrj
DF1KyI8emof6Cx/Tc4SSP1hrrkb8fVPRdFe4PWQqd/muzYZ4ol5PXFrIu3S8q9Sq
aP+477RvbC9DU5XyFFD2kYmTHoJcy0wMaEX3cXDUr9EMLYlz0stYLNGD7g++Y38Y
ikoCPiJSMDzz
-----END CERTIFICATE-----
`

const mismatchRSAKeyPEM = `-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQC/665h55hWD4V2
kiQ+B/G9NNfBw69eBibEhI9vWkPUyn36GO2r3HPtRE63wBfFpV486ns9DoZnnAYE
JaGjVNCCqS5tQyMBWp843o66KBrEgBpuddChigvyul33FhD1ImFnN+Vy0ajOJ+1/
Zai28zBXWbxCWEbqz7s8e2UsPlBd0Caj4gcd32yD2BwiHqzB8odToWRUT7l+pS8R
qA1BruQvtjEIrcoWVlE170ZYe7+Apm96A+WvtVRkozPynxHF8SuEiw4hAh0lXR6b
4zZz4tZVV8ev2HpffveV/68GiCyeFDbglqd4sZ/Iga/rwu7bVY/BzFApHwu2hmmV
XLnaa3uVAgMBAAECggEAG+kvnCdtPR7Wvw6z3J2VJ3oW4qQNzfPBEZVhssUC1mB4
f7W+Yt8VsOzdMdXq3yCUmvFS6OdC3rCPI21Bm5pLFKV8DgHUhm7idwfO4/3PHsKu
lV/m7odAA5Xc8oEwCCZu2e8EHHWnQgwGex+SsMCfSCTRvyhNb/qz9TDQ3uVVFL9e
9a4OKqZl/GlRspJSuXhy+RSVulw9NjeX1VRjIbhqpdXAmQNXgShA+gZSQh8T/tgv
XQYsMtg+FUDvcunJQf4OW5BY7IenYBV/GvsnJU8L7oD0wjNSAwe/iLKqV/NpYhre
QR4DsGnmoRYlUlHdHFTTJpReDjWm+vH3T756yDdFAQKBgQD2/sP5dM/aEW7Z1TgS
TG4ts1t8Rhe9escHxKZQR81dfOxBeCJMBDm6ySfR8rvyUM4VsogxBL/RhRQXsjJM
7wN08MhdiXG0J5yy/oNo8W6euD8m8Mk1UmqcZjSgV4vA7zQkvkr6DRJdybKsT9mE
jouEwev8sceS6iBpPw/+Ws8z1QKBgQDG6uYHMfMcS844xKQQWhargdN2XBzeG6TV
YXfNFstNpD84d9zIbpG/AKJF8fKrseUhXkJhkDjFGJTriD3QQsntOFaDOrHMnveV
zGzvC4OTFUUFHe0SVJ0HuLf8YCHoZ+DXEeCKCN6zBXnUue+bt3NvLOf2yN5o9kYx
SIa8O1vIwQKBgEdONXWG65qg/ceVbqKZvhUjen3eHmxtTZhIhVsX34nlzq73567a
aXArMnvB/9Bs05IgAIFmRZpPOQW+RBdByVWxTabzTwgbh3mFUJqzWKQpvNGZIf1q
1axhNUA1BfulEwCojyyxKWQ6HoLwanOCU3T4JxDEokEfpku8EPn1bWwhAoGAAN8A
eOGYHfSbB5ac3VF3rfKYmXkXy0U1uJV/r888vq9Mc5PazKnnS33WOBYyKNxTk4zV
H5ZBGWPdKxbipmnUdox7nIGCS9IaZXaKt5VGUzuRnM8fvafPNDxz2dAV9e2Wh3qV
kCUvzHrmqK7TxMvN3pvEvEju6GjDr+2QYXylD0ECgYAGK5r+y+EhtKkYFLeYReUt
znvSsWq+JCQH/cmtZLaVOldCaMRL625hSl3XPPcMIHE14xi3d4njoXWzvzPcg8L6
vNXk3GiNldACS+vwk4CwEqe5YlZRm5doD07wIdsg2zRlnKsnXNM152OwgmcchDul
rLTt0TTazzwBCgCD0Jkoqg==
-----END PRIVATE KEY-----`

func TestCreateSecretTLS(t *testing.T) {

	validCertTmpDir := utiltesting.MkTmpdirOrDie("tls-valid-cert-test")
	validKeyPath, validCertPath, validCAPath := writeCertData(validCertTmpDir, rsaKeyPEM, rsaCertPEM, rsaCAPEM, t)
	defer tearDown(validCertTmpDir)

	selfSignedCertTmpDir := utiltesting.MkTmpdirOrDie("tls-selfSigned-cert-test")
	selfSignedKeyPath, selfSignedCertPath, _ := writeCertData(selfSignedCertTmpDir, rsaKeyPEM, rsaSelfSignedCertPEM, "", t)
	defer tearDown(selfSignedCertTmpDir)

	invalidCertTmpDir := utiltesting.MkTmpdirOrDie("tls-invalid-cert-test")
	invalidKeyPath, invalidCertPath, invalidCAPath := writeCertData(invalidCertTmpDir, "test", "test", "test", t)
	defer tearDown(invalidCertTmpDir)

	mismatchCertTmpDir := utiltesting.MkTmpdirOrDie("tls-mismatch-test")
	mismatchKeyPath, mismatchCertPath, mismatchCAPath := writeCertData(mismatchCertTmpDir, rsaKeyPEM, mismatchRSAKeyPEM, "", t)
	defer tearDown(mismatchCertTmpDir)

	tests := map[string]struct {
		tlsSecretName    string
		tlsKey           string
		tlsCert          string
		tlsCertAuthority string
		appendHash       bool
		expected         *corev1.Secret
		expectErr        bool
	}{
		"create_secret_tls": {
			tlsSecretName:    "foo",
			tlsKey:           validKeyPath,
			tlsCert:          validCertPath,
			tlsCertAuthority: validCAPath,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Type: corev1.SecretTypeTLS,
				Data: map[string][]byte{
					corev1.TLSPrivateKeyKey: []byte(rsaKeyPEM),
					corev1.TLSCertKey:       []byte(rsaCertPEM),
					"ca.crt":                []byte(rsaCAPEM),
				},
			},
			expectErr: false,
		},
		"create_secret_tls_hash": {
			tlsSecretName:    "foo",
			tlsKey:           validKeyPath,
			tlsCert:          validCertPath,
			tlsCertAuthority: validCAPath,
			appendHash:       true,
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo-th4dtb9f52",
				},
				Type: corev1.SecretTypeTLS,
				Data: map[string][]byte{
					corev1.TLSPrivateKeyKey: []byte(rsaKeyPEM),
					corev1.TLSCertKey:       []byte(rsaCertPEM),
					"ca.crt":                []byte(rsaCAPEM),
				},
			},
			expectErr: false,
		},
		"create_secret_selfsigned_tls": {
			tlsSecretName:    "foo",
			tlsKey:           selfSignedKeyPath,
			tlsCert:          selfSignedCertPath,
			tlsCertAuthority: "",
			expected: &corev1.Secret{
				TypeMeta: metav1.TypeMeta{
					APIVersion: corev1.SchemeGroupVersion.String(),
					Kind:       "Secret",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Type: corev1.SecretTypeTLS,
				Data: map[string][]byte{
					corev1.TLSPrivateKeyKey: []byte(rsaKeyPEM),
					corev1.TLSCertKey:       []byte(rsaSelfSignedCertPEM),
				},
			},
			expectErr: false,
		},
		"create_secret_invalid_tls": {
			tlsSecretName:    "foo",
			tlsKey:           invalidKeyPath,
			tlsCert:          invalidCertPath,
			tlsCertAuthority: invalidCAPath,
			expectErr:        true,
		},
		"create_secret_mismatch_tls": {
			tlsSecretName:    "foo",
			tlsKey:           mismatchKeyPath,
			tlsCert:          mismatchCertPath,
			tlsCertAuthority: mismatchCAPath,
			expectErr:        true,
		},
		"create_invalid_filepath_and_certpath_secret_tls": {
			tlsSecretName:    "foo",
			tlsKey:           "testKeyPath",
			tlsCert:          "testCertPath",
			tlsCertAuthority: "",
			expectErr:        true,
		},
	}

	// Run all the tests
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			secretTLSOptions := CreateSecretTLSOptions{
				Name:          test.tlsSecretName,
				Key:           test.tlsKey,
				Cert:          test.tlsCert,
				CertAuthority: test.tlsCertAuthority,
				AppendHash:    test.appendHash,
			}
			secretTLS, err := secretTLSOptions.createSecretTLS()

			if !test.expectErr && err != nil {
				t.Errorf("test %s, unexpected error: %v", name, err)
			}
			if test.expectErr && err == nil {
				t.Errorf("test %s was expecting an error but no error occurred", name)
			}
			if !apiequality.Semantic.DeepEqual(secretTLS, test.expected) {
				t.Errorf("test %s\n expected:\n%#v\ngot:\n%#v", name, test.expected, secretTLS)
			}
		})
	}
}

func tearDown(tmpDir string) {
	err := os.RemoveAll(tmpDir)
	if err != nil {
		fmt.Printf("Error in cleaning up test: %v", err)
	}
}

func write(path, contents string, t *testing.T) {
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("Failed to create %v.", path)
	}
	defer f.Close()
	_, err = f.WriteString(contents)
	if err != nil {
		t.Fatalf("Failed to write to %v.", path)
	}
}

func writeCertData(tmpDirPath, key, cert, ca string, t *testing.T) (keyPath, certPath, caPath string) {
	keyPath = path.Join(tmpDirPath, "tls.key")
	certPath = path.Join(tmpDirPath, "tls.cert")
	write(keyPath, key, t)
	write(certPath, cert, t)
	if ca != "" {
		caPath = path.Join(tmpDirPath, "ca.cert")
		write(caPath, ca, t)
	} else {
		caPath = ""
	}
	return
}
