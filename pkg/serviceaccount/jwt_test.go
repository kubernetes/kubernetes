/*
Copyright 2014 The Kubernetes Authors.

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

package serviceaccount_test

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	jose "gopkg.in/square/go-jose.v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/keyutil"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

const otherPublicKey = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArXz0QkIG1B5Bj2/W69GH
rsm5e+RC3kE+VTgocge0atqlLBek35tRqLgUi3AcIrBZ/0YctMSWDVcRt5fkhWwe
Lqjj6qvAyNyOkrkBi1NFDpJBjYJtuKHgRhNxXbOzTSNpdSKXTfOkzqv56MwHOP25
yP/NNAODUtr92D5ySI5QX8RbXW+uDn+ixul286PBW/BCrE4tuS88dA0tYJPf8LCu
sqQOwlXYH/rNUg4Pyl9xxhR5DIJR0OzNNfChjw60zieRIt2LfM83fXhwk8IxRGkc
gPZm7ZsipmfbZK2Tkhnpsa4QxDg7zHJPMsB5kxRXW0cQipXcC3baDyN9KBApNXa0
PwIDAQAB
-----END PUBLIC KEY-----`

const rsaPublicKey = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA249XwEo9k4tM8fMxV7zx
OhcrP+WvXn917koM5Qr2ZXs4vo26e4ytdlrV0bQ9SlcLpQVSYjIxNfhTZdDt+ecI
zshKuv1gKIxbbLQMOuK1eA/4HALyEkFgmS/tleLJrhc65tKPMGD+pKQ/xhmzRuCG
51RoiMgbQxaCyYxGfNLpLAZK9L0Tctv9a0mJmGIYnIOQM4kC1A1I1n3EsXMWmeJU
j7OTh/AjjCnMnkgvKT2tpKxYQ59PgDgU8Ssc7RDSmSkLxnrv+OrN80j6xrw0OjEi
B4Ycr0PqfzZcvy8efTtFQ/Jnc4Bp1zUtFXt7+QeevePtQ2EcyELXE0i63T1CujRM
WwIDAQAB
-----END PUBLIC KEY-----
`

// Obtained by:
//
//   1. Serializing rsaPublicKey as DER
//   2. Taking the SHA256 of the DER bytes
//   3. URLSafe Base64-encoding the sha bytes
const rsaKeyID = "JHJehTTTZlsspKHT-GaJxK7Kd1NQgZJu3fyK6K_QDYU"

// Fake value for testing.
const rsaPrivateKey = `-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA249XwEo9k4tM8fMxV7zxOhcrP+WvXn917koM5Qr2ZXs4vo26
e4ytdlrV0bQ9SlcLpQVSYjIxNfhTZdDt+ecIzshKuv1gKIxbbLQMOuK1eA/4HALy
EkFgmS/tleLJrhc65tKPMGD+pKQ/xhmzRuCG51RoiMgbQxaCyYxGfNLpLAZK9L0T
ctv9a0mJmGIYnIOQM4kC1A1I1n3EsXMWmeJUj7OTh/AjjCnMnkgvKT2tpKxYQ59P
gDgU8Ssc7RDSmSkLxnrv+OrN80j6xrw0OjEiB4Ycr0PqfzZcvy8efTtFQ/Jnc4Bp
1zUtFXt7+QeevePtQ2EcyELXE0i63T1CujRMWwIDAQABAoIBAHJx8GqyCBDNbqk7
e7/hI9iE1S10Wwol5GH2RWxqX28cYMKq+8aE2LI1vPiXO89xOgelk4DN6urX6xjK
ZBF8RRIMQy/e/O2F4+3wl+Nl4vOXV1u6iVXMsD6JRg137mqJf1Fr9elg1bsaRofL
Q7CxPoB8dhS+Qb+hj0DhlqhgA9zG345CQCAds0ZYAZe8fP7bkwrLqZpMn7Dz9WVm
++YgYYKjuE95kPuup/LtWfA9rJyE/Fws8/jGvRSpVn1XglMLSMKhLd27sE8ZUSV0
2KUzbfRGE0+AnRULRrjpYaPu0XQ2JjdNvtkjBnv27RB89W9Gklxq821eH1Y8got8
FZodjxECgYEA93pz7AQZ2xDs67d1XLCzpX84GxKzttirmyj3OIlxgzVHjEMsvw8v
sjFiBU5xEEQDosrBdSknnlJqyiq1YwWG/WDckr13d8G2RQWoySN7JVmTQfXcLoTu
YGRiiTuoEi3ab3ZqrgGrFgX7T/cHuasbYvzCvhM2b4VIR3aSxU2DTUMCgYEA4x7J
T/ErP6GkU5nKstu/mIXwNzayEO1BJvPYsy7i7EsxTm3xe/b8/6cYOz5fvJLGH5mT
Q8YvuLqBcMwZardrYcwokD55UvNLOyfADDFZ6l3WntIqbA640Ok2g1X4U8J09xIq
ZLIWK1yWbbvi4QCeN5hvWq47e8sIj5QHjIIjRwkCgYEAyNqjltxFN9zmzPDa2d24
EAvOt3pYTYBQ1t9KtqImdL0bUqV6fZ6PsWoPCgt+DBuHb+prVPGP7Bkr/uTmznU/
+AlTO+12NsYLbr2HHagkXE31DEXE7CSLa8RNjN/UKtz4Ohq7vnowJvG35FCz/mb3
FUHbtHTXa2+bGBUOTf/5Hw0CgYBxw0r9EwUhw1qnUYJ5op7OzFAtp+T7m4ul8kCa
SCL8TxGsgl+SQ34opE775dtYfoBk9a0RJqVit3D8yg71KFjOTNAIqHJm/Vyyjc+h
i9rJDSXiuczsAVfLtPVMRfS0J9QkqeG4PIfkQmVLI/CZ2ZBmsqEcX+eFs4ZfPLun
Qsxe2QKBgGuPilIbLeIBDIaPiUI0FwU8v2j8CEQBYvoQn34c95hVQsig/o5z7zlo
UsO0wlTngXKlWdOcCs1kqEhTLrstf48djDxAYAxkw40nzeJOt7q52ib/fvf4/UBy
X024wzbiw1q07jFCyfQmODzURAx1VNT7QVUMdz/N8vy47/H40AZJ
-----END RSA PRIVATE KEY-----
`

// openssl ecparam -name prime256v1 -genkey -noout -out ecdsa256.pem
// Fake value for testing.
const ecdsaPrivateKey = `-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIEZmTmUhuanLjPA2CLquXivuwBDHTt5XYwgIr/kA1LtRoAoGCCqGSM49
AwEHoUQDQgAEH6cuzP8XuD5wal6wf9M6xDljTOPLX2i8uIp/C/ASqiIGUeeKQtX0
/IR3qCXyThP/dbCiHrF3v1cuhBOHY8CLVg==
-----END EC PRIVATE KEY-----`

// openssl ec -in ecdsa256.pem -pubout -out ecdsa256pub.pem
const ecdsaPublicKey = `-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEH6cuzP8XuD5wal6wf9M6xDljTOPL
X2i8uIp/C/ASqiIGUeeKQtX0/IR3qCXyThP/dbCiHrF3v1cuhBOHY8CLVg==
-----END PUBLIC KEY-----`

// Obtained by:
//
//   1. Serializing ecdsaPublicKey as DER
//   2. Taking the SHA256 of the DER bytes
//   3. URLSafe Base64-encoding the sha bytes
const ecdsaKeyID = "SoABiieYuNx4UdqYvZRVeuC6SihxgLrhLy9peHMHpTc"

func getPrivateKey(data string) interface{} {
	key, err := keyutil.ParsePrivateKeyPEM([]byte(data))
	if err != nil {
		panic(fmt.Errorf("unexpected error parsing private key: %v", err))
	}
	return key
}

func getPublicKey(data string) interface{} {
	keys, err := keyutil.ParsePublicKeysPEM([]byte(data))
	if err != nil {
		panic(fmt.Errorf("unexpected error parsing public key: %v", err))
	}
	return keys[0]
}

func TestTokenGenerateAndValidate(t *testing.T) {
	expectedUserName := "system:serviceaccount:test:my-service-account"
	expectedUserUID := "12345"

	// Related API objects
	serviceAccount := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-service-account",
			UID:       "12345",
			Namespace: "test",
		},
	}
	rsaSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-rsa-secret",
			Namespace: "test",
		},
	}
	ecdsaSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-ecdsa-secret",
			Namespace: "test",
		},
	}

	// Generate the RSA token
	rsaGenerator, err := serviceaccount.JWTTokenGenerator(serviceaccount.LegacyIssuer, getPrivateKey(rsaPrivateKey))
	if err != nil {
		t.Fatalf("error making generator: %v", err)
	}
	rsaToken, err := rsaGenerator.GenerateToken(serviceaccount.LegacyClaims(*serviceAccount, *rsaSecret))
	if err != nil {
		t.Fatalf("error generating token: %v", err)
	}
	if len(rsaToken) == 0 {
		t.Fatalf("no token generated")
	}
	rsaSecret.Data = map[string][]byte{
		"token": []byte(rsaToken),
	}

	checkJSONWebSignatureHasKeyID(t, rsaToken, rsaKeyID)

	// Generate the ECDSA token
	ecdsaGenerator, err := serviceaccount.JWTTokenGenerator(serviceaccount.LegacyIssuer, getPrivateKey(ecdsaPrivateKey))
	if err != nil {
		t.Fatalf("error making generator: %v", err)
	}
	ecdsaToken, err := ecdsaGenerator.GenerateToken(serviceaccount.LegacyClaims(*serviceAccount, *ecdsaSecret))
	if err != nil {
		t.Fatalf("error generating token: %v", err)
	}
	if len(ecdsaToken) == 0 {
		t.Fatalf("no token generated")
	}
	ecdsaSecret.Data = map[string][]byte{
		"token": []byte(ecdsaToken),
	}

	checkJSONWebSignatureHasKeyID(t, ecdsaToken, ecdsaKeyID)

	// Generate signer with same keys as RSA signer but different issuer
	badIssuerGenerator, err := serviceaccount.JWTTokenGenerator("foo", getPrivateKey(rsaPrivateKey))
	if err != nil {
		t.Fatalf("error making generator: %v", err)
	}
	badIssuerToken, err := badIssuerGenerator.GenerateToken(serviceaccount.LegacyClaims(*serviceAccount, *rsaSecret))
	if err != nil {
		t.Fatalf("error generating token: %v", err)
	}

	testCases := map[string]struct {
		Client clientset.Interface
		Keys   []interface{}
		Token  string

		ExpectedErr      bool
		ExpectedOK       bool
		ExpectedUserName string
		ExpectedUserUID  string
		ExpectedGroups   []string
	}{
		"no keys": {
			Token:       rsaToken,
			Client:      nil,
			Keys:        []interface{}{},
			ExpectedErr: false,
			ExpectedOK:  false,
		},
		"invalid keys (rsa)": {
			Token:       rsaToken,
			Client:      nil,
			Keys:        []interface{}{getPublicKey(otherPublicKey), getPublicKey(ecdsaPublicKey)},
			ExpectedErr: true,
			ExpectedOK:  false,
		},
		"invalid keys (ecdsa)": {
			Token:       ecdsaToken,
			Client:      nil,
			Keys:        []interface{}{getPublicKey(otherPublicKey), getPublicKey(rsaPublicKey)},
			ExpectedErr: true,
			ExpectedOK:  false,
		},
		"valid key (rsa)": {
			Token:            rsaToken,
			Client:           nil,
			Keys:             []interface{}{getPublicKey(rsaPublicKey)},
			ExpectedErr:      false,
			ExpectedOK:       true,
			ExpectedUserName: expectedUserName,
			ExpectedUserUID:  expectedUserUID,
			ExpectedGroups:   []string{"system:serviceaccounts", "system:serviceaccounts:test"},
		},
		"valid key, invalid issuer (rsa)": {
			Token:       badIssuerToken,
			Client:      nil,
			Keys:        []interface{}{getPublicKey(rsaPublicKey)},
			ExpectedErr: false,
			ExpectedOK:  false,
		},
		"valid key (ecdsa)": {
			Token:            ecdsaToken,
			Client:           nil,
			Keys:             []interface{}{getPublicKey(ecdsaPublicKey)},
			ExpectedErr:      false,
			ExpectedOK:       true,
			ExpectedUserName: expectedUserName,
			ExpectedUserUID:  expectedUserUID,
			ExpectedGroups:   []string{"system:serviceaccounts", "system:serviceaccounts:test"},
		},
		"rotated keys (rsa)": {
			Token:            rsaToken,
			Client:           nil,
			Keys:             []interface{}{getPublicKey(otherPublicKey), getPublicKey(ecdsaPublicKey), getPublicKey(rsaPublicKey)},
			ExpectedErr:      false,
			ExpectedOK:       true,
			ExpectedUserName: expectedUserName,
			ExpectedUserUID:  expectedUserUID,
			ExpectedGroups:   []string{"system:serviceaccounts", "system:serviceaccounts:test"},
		},
		"rotated keys (ecdsa)": {
			Token:            ecdsaToken,
			Client:           nil,
			Keys:             []interface{}{getPublicKey(otherPublicKey), getPublicKey(rsaPublicKey), getPublicKey(ecdsaPublicKey)},
			ExpectedErr:      false,
			ExpectedOK:       true,
			ExpectedUserName: expectedUserName,
			ExpectedUserUID:  expectedUserUID,
			ExpectedGroups:   []string{"system:serviceaccounts", "system:serviceaccounts:test"},
		},
		"valid lookup": {
			Token:            rsaToken,
			Client:           fake.NewSimpleClientset(serviceAccount, rsaSecret, ecdsaSecret),
			Keys:             []interface{}{getPublicKey(rsaPublicKey)},
			ExpectedErr:      false,
			ExpectedOK:       true,
			ExpectedUserName: expectedUserName,
			ExpectedUserUID:  expectedUserUID,
			ExpectedGroups:   []string{"system:serviceaccounts", "system:serviceaccounts:test"},
		},
		"invalid secret lookup": {
			Token:       rsaToken,
			Client:      fake.NewSimpleClientset(serviceAccount),
			Keys:        []interface{}{getPublicKey(rsaPublicKey)},
			ExpectedErr: true,
			ExpectedOK:  false,
		},
		"invalid serviceaccount lookup": {
			Token:       rsaToken,
			Client:      fake.NewSimpleClientset(rsaSecret, ecdsaSecret),
			Keys:        []interface{}{getPublicKey(rsaPublicKey)},
			ExpectedErr: true,
			ExpectedOK:  false,
		},
	}

	for k, tc := range testCases {
		auds := authenticator.Audiences{"api"}
		getter := serviceaccountcontroller.NewGetterFromClient(
			tc.Client,
			v1listers.NewSecretLister(newIndexer(func(namespace, name string) (interface{}, error) {
				return tc.Client.CoreV1().Secrets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
			})),
			v1listers.NewServiceAccountLister(newIndexer(func(namespace, name string) (interface{}, error) {
				return tc.Client.CoreV1().ServiceAccounts(namespace).Get(context.TODO(), name, metav1.GetOptions{})
			})),
			v1listers.NewPodLister(newIndexer(func(namespace, name string) (interface{}, error) {
				return tc.Client.CoreV1().Pods(namespace).Get(context.TODO(), name, metav1.GetOptions{})
			})),
		)
		authn := serviceaccount.JWTTokenAuthenticator(serviceaccount.LegacyIssuer, tc.Keys, auds, serviceaccount.NewLegacyValidator(tc.Client != nil, getter))

		// An invalid, non-JWT token should always fail
		ctx := authenticator.WithAudiences(context.Background(), auds)
		if _, ok, err := authn.AuthenticateToken(ctx, "invalid token"); err != nil || ok {
			t.Errorf("%s: Expected err=nil, ok=false for non-JWT token", k)
			continue
		}

		resp, ok, err := authn.AuthenticateToken(ctx, tc.Token)
		if (err != nil) != tc.ExpectedErr {
			t.Errorf("%s: Expected error=%v, got %v", k, tc.ExpectedErr, err)
			continue
		}

		if ok != tc.ExpectedOK {
			t.Errorf("%s: Expected ok=%v, got %v", k, tc.ExpectedOK, ok)
			continue
		}

		if err != nil || !ok {
			continue
		}

		if resp.User.GetName() != tc.ExpectedUserName {
			t.Errorf("%s: Expected username=%v, got %v", k, tc.ExpectedUserName, resp.User.GetName())
			continue
		}
		if resp.User.GetUID() != tc.ExpectedUserUID {
			t.Errorf("%s: Expected userUID=%v, got %v", k, tc.ExpectedUserUID, resp.User.GetUID())
			continue
		}
		if !reflect.DeepEqual(resp.User.GetGroups(), tc.ExpectedGroups) {
			t.Errorf("%s: Expected groups=%v, got %v", k, tc.ExpectedGroups, resp.User.GetGroups())
			continue
		}
	}
}

func checkJSONWebSignatureHasKeyID(t *testing.T, jwsString string, expectedKeyID string) {
	jws, err := jose.ParseSigned(jwsString)
	if err != nil {
		t.Fatalf("Error checking for key ID: couldn't parse token: %v", err)
	}

	if jws.Signatures[0].Header.KeyID != expectedKeyID {
		t.Errorf("Token %q has the wrong KeyID (got %q, want %q)", jwsString, jws.Signatures[0].Header.KeyID, expectedKeyID)
	}
}

func newIndexer(get func(namespace, name string) (interface{}, error)) cache.Indexer {
	return &fakeIndexer{get: get}
}

type fakeIndexer struct {
	cache.Indexer
	get func(namespace, name string) (interface{}, error)
}

func (f *fakeIndexer) GetByKey(key string) (interface{}, bool, error) {
	parts := strings.SplitN(key, "/", 2)
	namespace := parts[0]
	name := ""
	if len(parts) == 2 {
		name = parts[1]
	}
	obj, err := f.get(namespace, name)
	return obj, err == nil, err
}
