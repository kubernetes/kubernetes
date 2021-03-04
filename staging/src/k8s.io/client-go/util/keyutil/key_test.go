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

package keyutil

import (
	"io/ioutil"
	"os"
	"testing"
)

const (
	// rsaPrivateKey is a RSA Private Key in PKCS#1 format
	// openssl genrsa -out rsa2048.pem 2048
	rsaPrivateKey = `-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA92mVjhBKOFsdxFzb/Pjq+7b5TJlODAdY5hK+WxLZTIrfhDPq
FWrGKdjSNiHbXrdEtwJh9V+RqPZVSN3aWy1224RgkyNdMJsXhJKuCC24ZKY8SXtW
xuTYmMRaMnCsv6QBGRTIbZ2EFbAObVM7lDyv1VqY3amZIWFQMlZ9CNpxDSPa5yi4
3gopbXkne0oGNmey9X0qtpk7NMZIgAL6Zz4rZ30bcfC2ag6RLOFI2E/c4n8c38R8
9MfXfLkj8/Cxo4JfI9NvRCpPOpFO8d/ZtWVUuIrBQN+Y7tkN2T60Qq/TkKXUrhDe
fwlTlktZVJ/GztLYU41b2GcWsh/XO+PH831rmwIDAQABAoIBAQCC9c6GDjVbM0/E
WurPMusfJjE7zII1d8YkspM0HfwLug6qKdikUYpnKC/NG4rEzfl/bbFwco/lgc6O
7W/hh2U8uQttlvCDA/Uk5YddKOZL0Hpk4vaB/SxxYK3luSKXpjY2knutGg2KdVCN
qdsFkkH4iyYTXuyBcMNEgedZQldI/kEujIH/L7FE+DF5TMzT4lHhozDoG+fy564q
qVGUZXJn0ubc3GaPn2QOLNNM44sfYA4UJCpKBXPu85bvNObjxVQO4WqwwxU1vRnL
UUsaGaelhSVJCo0dVPRvrfPPKZ09HTwpy40EkgQo6VriFc1EBoQDjENLbAJv9OfQ
aCc9wiZhAoGBAP/8oEy48Zbb0P8Vdy4djf5tfBW8yXFLWzXewJ4l3itKS1r42nbX
9q3cJsgRTQm8uRcMIpWxsc3n6zG+lREvTkoTB3ViI7+uQPiqA+BtWyNy7jzufFke
ONKZfg7QxxmYRWZBRnoNGNbMpNeERuLmhvQuom9D1WbhzAYJbfs/O4WTAoGBAPds
2FNDU0gaesFDdkIUGq1nIJqRQDW485LXZm4pFqBFxdOpbdWRuYT2XZjd3fD0XY98
Nhkpb7NTMCuK3BdKcqIptt+cK+quQgYid0hhhgZbpCQ5AL6c6KgyjgpYlh2enzU9
Zo3yg8ej1zbbA11sBlhX+5iO2P1u5DG+JHLwUUbZAoGAUwaU102EzfEtsA4+QW7E
hyjrfgFlNKHES4yb3K9bh57pIfBkqvcQwwMMcQdrfSUAw0DkVrjzel0mI1Q09QXq
1ould6UFAz55RC2gZEITtUOpkYmoOx9aPrQZ9qQwb1S77ZZuTVfCHqjxLhVxCFbM
npYhiQTvShciHTMhwMOZgpECgYAVV5EtVXBYltgh1YTc3EkUzgF087R7LdHsx6Gx
POATwRD4WfP8aQ58lpeqOPEM+LcdSlSMRRO6fyF3kAm+BJDwxfJdRWZQXumZB94M
I0VhRQRaj4Qt7PDwmTPBVrTUJzuKZxpyggm17b8Bn1Ch/VBqzGQKW8AB1E/grosM
UwhfuQKBgQC2JO/iqTQScHClf0qlItCJsBuVukFmSAVCkpOD8YdbdlPdOOwSk1wQ
C0eAlsC3BCMvkpidKQmra6IqIrvTGI6EFgkrb3aknWdup2w8j2udYCNqyE3W+fVe
p8FdYQ1FkACQ+daO5VlClL/9l0sGjKXlNKbpmJ2H4ngZmXj5uGmxuQ==
-----END RSA PRIVATE KEY-----`

	// rsaPublicKey is a RSA Public Key in PEM encoded format
	// openssl rsa -in rsa2048.pem -pubout -out rsa2048pub.pem
	rsaPublicKey = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA92mVjhBKOFsdxFzb/Pjq
+7b5TJlODAdY5hK+WxLZTIrfhDPqFWrGKdjSNiHbXrdEtwJh9V+RqPZVSN3aWy12
24RgkyNdMJsXhJKuCC24ZKY8SXtWxuTYmMRaMnCsv6QBGRTIbZ2EFbAObVM7lDyv
1VqY3amZIWFQMlZ9CNpxDSPa5yi43gopbXkne0oGNmey9X0qtpk7NMZIgAL6Zz4r
Z30bcfC2ag6RLOFI2E/c4n8c38R89MfXfLkj8/Cxo4JfI9NvRCpPOpFO8d/ZtWVU
uIrBQN+Y7tkN2T60Qq/TkKXUrhDefwlTlktZVJ/GztLYU41b2GcWsh/XO+PH831r
mwIDAQAB
-----END PUBLIC KEY-----`

	// certificate is an x509 certificate in PEM encoded format
	// openssl req -new -key rsa2048.pem -sha256 -nodes -x509 -days 1826 -out x509certificate.pem -subj "/C=US/CN=not-valid"
	certificate = `-----BEGIN CERTIFICATE-----
MIIDFTCCAf2gAwIBAgIJAN8B8NOwtiUCMA0GCSqGSIb3DQEBCwUAMCExCzAJBgNV
BAYTAlVTMRIwEAYDVQQDDAlub3QtdmFsaWQwHhcNMTcwMzIyMDI1NjM2WhcNMjIw
MzIyMDI1NjM2WjAhMQswCQYDVQQGEwJVUzESMBAGA1UEAwwJbm90LXZhbGlkMIIB
IjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA92mVjhBKOFsdxFzb/Pjq+7b5
TJlODAdY5hK+WxLZTIrfhDPqFWrGKdjSNiHbXrdEtwJh9V+RqPZVSN3aWy1224Rg
kyNdMJsXhJKuCC24ZKY8SXtWxuTYmMRaMnCsv6QBGRTIbZ2EFbAObVM7lDyv1VqY
3amZIWFQMlZ9CNpxDSPa5yi43gopbXkne0oGNmey9X0qtpk7NMZIgAL6Zz4rZ30b
cfC2ag6RLOFI2E/c4n8c38R89MfXfLkj8/Cxo4JfI9NvRCpPOpFO8d/ZtWVUuIrB
QN+Y7tkN2T60Qq/TkKXUrhDefwlTlktZVJ/GztLYU41b2GcWsh/XO+PH831rmwID
AQABo1AwTjAdBgNVHQ4EFgQU1I5GfinLF7ta+dBJ6UWcrYaexLswHwYDVR0jBBgw
FoAU1I5GfinLF7ta+dBJ6UWcrYaexLswDAYDVR0TBAUwAwEB/zANBgkqhkiG9w0B
AQsFAAOCAQEAUl0wUD4y41juHFOVMYiziPYr1ShSpQXdwp8FfaHrzI5hsr8UMe8D
dzb9QzZ4bx3yZhiG3ahrSBh956thMTHrKTEwAfJIEXI4cuSVWQAaOJ4Em5SDFxQe
d0E6Ui2nGh1SFGF7oyuEXyzqgRMWFNDFw9HLUNgXaO18Zfouw8+K0BgbfEWEcSi1
JLQbyhCjz088gltrliQGPWDFAg9cHBKtJhuTzZkvuqK1CLEmBhtzP1zFiGBfOJc8
v+aKjAwrPUNX11cXOCPxBv2qXMetxaovBem6AI2hvypCInXaVQfP+yOLubzlTDjS
Y708SlY38hmS1uTwDpyLOn8AKkZ8jtx75g==
-----END CERTIFICATE-----`

	// ecdsaPrivateKeyWithParams is a ECDSA Private Key with included EC Parameters block
	// openssl ecparam -name prime256v1 -genkey -out ecdsa256params.pem
	ecdsaPrivateKeyWithParams = `-----BEGIN EC PARAMETERS-----
BggqhkjOPQMBBw==
-----END EC PARAMETERS-----
-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIAwSOWQqlMTZNqNF7tgua812Jxib1DVOgb2pHHyIEyNNoAoGCCqGSM49
AwEHoUQDQgAEyxYNrs6a6tsNCFNYn+l+JDUZ0PnUZbcsDgJn2O62D1se8M5iQ5rY
iIv6RpxE3VHvlHEIvYgCZkG0jHszTUopBg==
-----END EC PRIVATE KEY-----`

	// ecdsaPrivateKey is a ECDSA Private Key in ASN.1 format
	// openssl ecparam -name prime256v1 -genkey -noout -out ecdsa256.pem
	ecdsaPrivateKey = `-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIP6Qw6dHDiLsSnLXUhQVTPE0fTQQrj3XSbiQAZPXnk5+oAoGCCqGSM49
AwEHoUQDQgAEZZzi1u5f2/AEGFI/HYUhU+u6cTK1q2bbtE7r1JMK+/sQA5sNAp+7
Vdc3psr1OaNzyTyuhTECyRdFKXm63cMnGg==
-----END EC PRIVATE KEY-----`

	// ecdsaPublicKey is a ECDSA Public Key in PEM encoded format
	// openssl ec -in ecdsa256.pem -pubout -out ecdsa256pub.pem
	ecdsaPublicKey = `-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEZZzi1u5f2/AEGFI/HYUhU+u6cTK1
q2bbtE7r1JMK+/sQA5sNAp+7Vdc3psr1OaNzyTyuhTECyRdFKXm63cMnGg==
-----END PUBLIC KEY-----`
)

func TestReadPrivateKey(t *testing.T) {
	f, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatalf("error creating tmpfile: %v", err)
	}
	defer os.Remove(f.Name())

	if _, err := PrivateKeyFromFile(f.Name()); err == nil {
		t.Fatalf("Expected error reading key from empty file, got none")
	}

	if err := ioutil.WriteFile(f.Name(), []byte(rsaPrivateKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing private key to tmpfile: %v", err)
	}
	if _, err := PrivateKeyFromFile(f.Name()); err != nil {
		t.Fatalf("error reading private RSA key: %v", err)
	}

	if err := ioutil.WriteFile(f.Name(), []byte(ecdsaPrivateKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing private key to tmpfile: %v", err)
	}
	if _, err := PrivateKeyFromFile(f.Name()); err != nil {
		t.Fatalf("error reading private ECDSA key: %v", err)
	}

	if err := ioutil.WriteFile(f.Name(), []byte(ecdsaPrivateKeyWithParams), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing private key to tmpfile: %v", err)
	}
	if _, err := PrivateKeyFromFile(f.Name()); err != nil {
		t.Fatalf("error reading private ECDSA key with params: %v", err)
	}
}

func TestReadPublicKeys(t *testing.T) {
	f, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatalf("error creating tmpfile: %v", err)
	}
	defer os.Remove(f.Name())

	if _, err := PublicKeysFromFile(f.Name()); err == nil {
		t.Fatalf("Expected error reading keys from empty file, got none")
	}

	if err := ioutil.WriteFile(f.Name(), []byte(rsaPublicKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing public key to tmpfile: %v", err)
	}
	if keys, err := PublicKeysFromFile(f.Name()); err != nil {
		t.Fatalf("error reading RSA public key: %v", err)
	} else if len(keys) != 1 {
		t.Fatalf("expected 1 key, got %d", len(keys))
	}

	if err := ioutil.WriteFile(f.Name(), []byte(ecdsaPublicKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing public key to tmpfile: %v", err)
	}
	if keys, err := PublicKeysFromFile(f.Name()); err != nil {
		t.Fatalf("error reading ECDSA public key: %v", err)
	} else if len(keys) != 1 {
		t.Fatalf("expected 1 key, got %d", len(keys))
	}

	if err := ioutil.WriteFile(f.Name(), []byte(rsaPublicKey+"\n"+ecdsaPublicKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing public key to tmpfile: %v", err)
	}
	if keys, err := PublicKeysFromFile(f.Name()); err != nil {
		t.Fatalf("error reading combined RSA/ECDSA public key file: %v", err)
	} else if len(keys) != 2 {
		t.Fatalf("expected 2 keys, got %d", len(keys))
	}

	if err := ioutil.WriteFile(f.Name(), []byte(certificate), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing certificate to tmpfile: %v", err)
	}
	if keys, err := PublicKeysFromFile(f.Name()); err != nil {
		t.Fatalf("error reading public key from certificate file: %v", err)
	} else if len(keys) != 1 {
		t.Fatalf("expected 1 keys, got %d", len(keys))
	}

}
