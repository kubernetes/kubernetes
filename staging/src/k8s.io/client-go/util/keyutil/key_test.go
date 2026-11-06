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

	// mldsaPrivateKey is a ML-DSA-65 Private Key in unencrypted PKCS#8 format
	// openssl genpkey -algorithm ML-DSA-65 -out mldsa65.pem
	mldsaPrivateKey = `-----BEGIN PRIVATE KEY-----
MDQCAQAwCwYJYIZIAWUDBAMSBCKAINgd6EB5JBi28ztvyFxNM3YCNcldR8MmSO04
zfKOIXsK
-----END PRIVATE KEY-----`

	// mldsaPublicKey is a ML-DSA-65 Public Key in PEM encoded format
	// openssl pkey -in mldsa65.pem -pubout -out mldsa65pub.pem
	mldsaPublicKey = `-----BEGIN PUBLIC KEY-----
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
O1AXlDlT
-----END PUBLIC KEY-----`
)

func TestReadPrivateKey(t *testing.T) {
	f, err := os.CreateTemp("", "")
	if err != nil {
		t.Fatalf("error creating tmpfile: %v", err)
	}
	defer os.Remove(f.Name())

	if _, err := PrivateKeyFromFile(f.Name()); err == nil {
		t.Fatalf("Expected error reading key from empty file, got none")
	}

	if err := os.WriteFile(f.Name(), []byte(rsaPrivateKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing private key to tmpfile: %v", err)
	}
	if _, err := PrivateKeyFromFile(f.Name()); err != nil {
		t.Fatalf("error reading private RSA key: %v", err)
	}

	if err := os.WriteFile(f.Name(), []byte(ecdsaPrivateKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing private key to tmpfile: %v", err)
	}
	if _, err := PrivateKeyFromFile(f.Name()); err != nil {
		t.Fatalf("error reading private ECDSA key: %v", err)
	}

	if err := os.WriteFile(f.Name(), []byte(ecdsaPrivateKeyWithParams), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing private key to tmpfile: %v", err)
	}
	if _, err := PrivateKeyFromFile(f.Name()); err != nil {
		t.Fatalf("error reading private ECDSA key with params: %v", err)
	}

	if err := os.WriteFile(f.Name(), []byte(mldsaPrivateKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing private key to tmpfile: %v", err)
	}
	if _, err := PrivateKeyFromFile(f.Name()); err != nil {
		t.Fatalf("error reading private ML-DSA key: %v", err)
	}
}

func TestReadPublicKeys(t *testing.T) {
	f, err := os.CreateTemp("", "")
	if err != nil {
		t.Fatalf("error creating tmpfile: %v", err)
	}
	defer os.Remove(f.Name())

	if _, err := PublicKeysFromFile(f.Name()); err == nil {
		t.Fatalf("Expected error reading keys from empty file, got none")
	}

	if err := os.WriteFile(f.Name(), []byte(rsaPublicKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing public key to tmpfile: %v", err)
	}
	if keys, err := PublicKeysFromFile(f.Name()); err != nil {
		t.Fatalf("error reading RSA public key: %v", err)
	} else if len(keys) != 1 {
		t.Fatalf("expected 1 key, got %d", len(keys))
	}

	if err := os.WriteFile(f.Name(), []byte(ecdsaPublicKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing public key to tmpfile: %v", err)
	}
	if keys, err := PublicKeysFromFile(f.Name()); err != nil {
		t.Fatalf("error reading ECDSA public key: %v", err)
	} else if len(keys) != 1 {
		t.Fatalf("expected 1 key, got %d", len(keys))
	}

	if err := os.WriteFile(f.Name(), []byte(mldsaPublicKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing public key to tmpfile: %v", err)
	}
	if keys, err := PublicKeysFromFile(f.Name()); err != nil {
		t.Fatalf("error reading ML-DSA public key: %v", err)
	} else if len(keys) != 1 {
		t.Fatalf("expected 1 key, got %d", len(keys))
	}

	if err := os.WriteFile(f.Name(), []byte(rsaPublicKey+"\n"+ecdsaPublicKey+"\n"+mldsaPublicKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing public key to tmpfile: %v", err)
	}
	if keys, err := PublicKeysFromFile(f.Name()); err != nil {
		t.Fatalf("error reading combined RSA/ECDSA/ML-DSA public key file: %v", err)
	} else if len(keys) != 3 {
		t.Fatalf("expected 3 keys, got %d", len(keys))
	}

	if err := os.WriteFile(f.Name(), []byte(certificate), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing certificate to tmpfile: %v", err)
	}
	if keys, err := PublicKeysFromFile(f.Name()); err != nil {
		t.Fatalf("error reading public key from certificate file: %v", err)
	} else if len(keys) != 1 {
		t.Fatalf("expected 1 keys, got %d", len(keys))
	}

}
