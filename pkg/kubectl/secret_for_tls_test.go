/*
Copyright 2015 The Kubernetes Authors.

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

package kubectl

import (
	"fmt"
	"os"
	"path"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/api"
)

var rsaCertPEM = `-----BEGIN CERTIFICATE-----
MIIB0zCCAX2gAwIBAgIJAI/M7BYjwB+uMA0GCSqGSIb3DQEBBQUAMEUxCzAJBgNV
BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
aWRnaXRzIFB0eSBMdGQwHhcNMTIwOTEyMjE1MjAyWhcNMTUwOTEyMjE1MjAyWjBF
MQswCQYDVQQGEwJBVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50
ZXJuZXQgV2lkZ2l0cyBQdHkgTHRkMFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBANLJ
hPHhITqQbPklG3ibCVxwGMRfp/v4XqhfdQHdcVfHap6NQ5Wok/4xIA+ui35/MmNa
rtNuC+BdZ1tMuVCPFZcCAwEAAaNQME4wHQYDVR0OBBYEFJvKs8RfJaXTH08W+SGv
zQyKn0H8MB8GA1UdIwQYMBaAFJvKs8RfJaXTH08W+SGvzQyKn0H8MAwGA1UdEwQF
MAMBAf8wDQYJKoZIhvcNAQEFBQADQQBJlffJHybjDGxRMqaRmDhX0+6v02TUKZsW
r5QuVbpQhH6u+0UgcW0jp9QwpxoPTLTWGXEWBBBurxFwiCBhkQ+V
-----END CERTIFICATE-----
`

var rsaKeyPEM = `-----BEGIN RSA PRIVATE KEY-----
MIIBOwIBAAJBANLJhPHhITqQbPklG3ibCVxwGMRfp/v4XqhfdQHdcVfHap6NQ5Wo
k/4xIA+ui35/MmNartNuC+BdZ1tMuVCPFZcCAwEAAQJAEJ2N+zsR0Xn8/Q6twa4G
6OB1M1WO+k+ztnX/1SvNeWu8D6GImtupLTYgjZcHufykj09jiHmjHx8u8ZZB/o1N
MQIhAPW+eyZo7ay3lMz1V01WVjNKK9QSn1MJlb06h/LuYv9FAiEA25WPedKgVyCW
SmUwbPw8fnTcpqDWE3yTO3vKcebqMSsCIBF3UmVue8YU3jybC3NxuXq3wNm34R8T
xVLHwDXh/6NJAiEAl2oHGGLz64BuAfjKrqwz7qMYr9HCLIe/YsoWq/olzScCIQDi
D2lWusoe2/nEqfDVVWGWlyJ7yOmqaVm/iNUN9B2N2g==
-----END RSA PRIVATE KEY-----
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

func writeKeyPair(tmpDirPath, key, cert string, t *testing.T) (keyPath, certPath string) {
	keyPath = path.Join(tmpDirPath, "tls.key")
	certPath = path.Join(tmpDirPath, "tls.cert")
	write(keyPath, key, t)
	write(certPath, cert, t)
	return
}

func TestSecretForTLSGenerate(t *testing.T) {
	invalidCertTmpDir := utiltesting.MkTmpdirOrDie("tls-test")
	defer tearDown(invalidCertTmpDir)
	invalidKeyPath, invalidCertPath := writeKeyPair(invalidCertTmpDir, "test", "test", t)

	validCertTmpDir := utiltesting.MkTmpdirOrDie("tls-test")
	defer tearDown(validCertTmpDir)
	validKeyPath, validCertPath := writeKeyPair(validCertTmpDir, rsaKeyPEM, rsaCertPEM, t)

	mismatchCertTmpDir := utiltesting.MkTmpdirOrDie("tls-mismatch-test")
	defer tearDown(mismatchCertTmpDir)
	mismatchKeyPath, mismatchCertPath := writeKeyPair(mismatchCertTmpDir, mismatchRSAKeyPEM, rsaCertPEM, t)

	tests := map[string]struct {
		params    map[string]interface{}
		expected  *api.Secret
		expectErr bool
	}{
		"test-valid-tls-secret": {
			params: map[string]interface{}{
				"name": "foo",
				"key":  validKeyPath,
				"cert": validCertPath,
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					api.TLSCertKey:       []byte(rsaCertPEM),
					api.TLSPrivateKeyKey: []byte(rsaKeyPEM),
				},
				Type: api.SecretTypeTLS,
			},
			expectErr: false,
		},
		"test-invalid-key-pair": {
			params: map[string]interface{}{
				"name": "foo",
				"key":  invalidKeyPath,
				"cert": invalidCertPath,
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					api.TLSCertKey:       []byte("test"),
					api.TLSPrivateKeyKey: []byte("test"),
				},
				Type: api.SecretTypeTLS,
			},
			expectErr: true,
		},
		"test-mismatched-key-pair": {
			params: map[string]interface{}{
				"name": "foo",
				"key":  mismatchKeyPath,
				"cert": mismatchCertPath,
			},
			expected: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string][]byte{
					api.TLSCertKey:       []byte(rsaCertPEM),
					api.TLSPrivateKeyKey: []byte(mismatchRSAKeyPEM),
				},
				Type: api.SecretTypeTLS,
			},
			expectErr: true,
		},
		"test-missing-required-param": {
			params: map[string]interface{}{
				"name": "foo",
				"key":  "/tmp/foo.key",
			},
			expectErr: true,
		},
	}

	generator := SecretForTLSGeneratorV1{}
	for _, test := range tests {
		obj, err := generator.Generate(test.params)
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if test.expectErr && err != nil {
			continue
		}
		if !reflect.DeepEqual(obj.(*api.Secret), test.expected) {
			t.Errorf("\nexpected:\n%#v\nsaw:\n%#v", test.expected, obj.(*api.Secret))
		}
	}
}
