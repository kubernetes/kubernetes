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
	"crypto/rsa"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"github.com/dgrijalva/jwt-go"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
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

const publicKey = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA249XwEo9k4tM8fMxV7zx
OhcrP+WvXn917koM5Qr2ZXs4vo26e4ytdlrV0bQ9SlcLpQVSYjIxNfhTZdDt+ecI
zshKuv1gKIxbbLQMOuK1eA/4HALyEkFgmS/tleLJrhc65tKPMGD+pKQ/xhmzRuCG
51RoiMgbQxaCyYxGfNLpLAZK9L0Tctv9a0mJmGIYnIOQM4kC1A1I1n3EsXMWmeJU
j7OTh/AjjCnMnkgvKT2tpKxYQ59PgDgU8Ssc7RDSmSkLxnrv+OrN80j6xrw0OjEi
B4Ycr0PqfzZcvy8efTtFQ/Jnc4Bp1zUtFXt7+QeevePtQ2EcyELXE0i63T1CujRM
WwIDAQAB
-----END PUBLIC KEY-----
`

const privateKey = `-----BEGIN RSA PRIVATE KEY-----
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

func getPrivateKey(data string) *rsa.PrivateKey {
	key, _ := jwt.ParseRSAPrivateKeyFromPEM([]byte(data))
	return key
}

func getPublicKey(data string) *rsa.PublicKey {
	key, _ := jwt.ParseRSAPublicKeyFromPEM([]byte(data))
	return key
}

func TestReadPrivateKey(t *testing.T) {
	f, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatalf("error creating tmpfile: %v", err)
	}
	defer os.Remove(f.Name())

	if err := ioutil.WriteFile(f.Name(), []byte(privateKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing private key to tmpfile: %v", err)
	}

	if _, err := serviceaccount.ReadPrivateKey(f.Name()); err != nil {
		t.Fatalf("error reading private key: %v", err)
	}
}

func TestReadPublicKey(t *testing.T) {
	f, err := ioutil.TempFile("", "")
	if err != nil {
		t.Fatalf("error creating tmpfile: %v", err)
	}
	defer os.Remove(f.Name())

	if err := ioutil.WriteFile(f.Name(), []byte(publicKey), os.FileMode(0600)); err != nil {
		t.Fatalf("error writing public key to tmpfile: %v", err)
	}

	if _, err := serviceaccount.ReadPublicKey(f.Name()); err != nil {
		t.Fatalf("error reading public key: %v", err)
	}
}

func TestTokenGenerateAndValidate(t *testing.T) {
	expectedUserName := "system:serviceaccount:test:my-service-account"
	expectedUserUID := "12345"

	// Related API objects
	serviceAccount := &api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:      "my-service-account",
			UID:       "12345",
			Namespace: "test",
		},
	}
	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:      "my-secret",
			Namespace: "test",
		},
	}

	// Generate the token
	generator := serviceaccount.JWTTokenGenerator(getPrivateKey(privateKey))
	token, err := generator.GenerateToken(*serviceAccount, *secret)
	if err != nil {
		t.Fatalf("error generating token: %v", err)
	}
	if len(token) == 0 {
		t.Fatalf("no token generated")
	}

	// "Save" the token
	secret.Data = map[string][]byte{
		"token": []byte(token),
	}

	testCases := map[string]struct {
		Client clientset.Interface
		Keys   []*rsa.PublicKey

		ExpectedErr      bool
		ExpectedOK       bool
		ExpectedUserName string
		ExpectedUserUID  string
		ExpectedGroups   []string
	}{
		"no keys": {
			Client:      nil,
			Keys:        []*rsa.PublicKey{},
			ExpectedErr: false,
			ExpectedOK:  false,
		},
		"invalid keys": {
			Client:      nil,
			Keys:        []*rsa.PublicKey{getPublicKey(otherPublicKey)},
			ExpectedErr: true,
			ExpectedOK:  false,
		},
		"valid key": {
			Client:           nil,
			Keys:             []*rsa.PublicKey{getPublicKey(publicKey)},
			ExpectedErr:      false,
			ExpectedOK:       true,
			ExpectedUserName: expectedUserName,
			ExpectedUserUID:  expectedUserUID,
			ExpectedGroups:   []string{"system:serviceaccounts", "system:serviceaccounts:test"},
		},
		"rotated keys": {
			Client:           nil,
			Keys:             []*rsa.PublicKey{getPublicKey(otherPublicKey), getPublicKey(publicKey)},
			ExpectedErr:      false,
			ExpectedOK:       true,
			ExpectedUserName: expectedUserName,
			ExpectedUserUID:  expectedUserUID,
			ExpectedGroups:   []string{"system:serviceaccounts", "system:serviceaccounts:test"},
		},
		"valid lookup": {
			Client:           fake.NewSimpleClientset(serviceAccount, secret),
			Keys:             []*rsa.PublicKey{getPublicKey(publicKey)},
			ExpectedErr:      false,
			ExpectedOK:       true,
			ExpectedUserName: expectedUserName,
			ExpectedUserUID:  expectedUserUID,
			ExpectedGroups:   []string{"system:serviceaccounts", "system:serviceaccounts:test"},
		},
		"invalid secret lookup": {
			Client:      fake.NewSimpleClientset(serviceAccount),
			Keys:        []*rsa.PublicKey{getPublicKey(publicKey)},
			ExpectedErr: true,
			ExpectedOK:  false,
		},
		"invalid serviceaccount lookup": {
			Client:      fake.NewSimpleClientset(secret),
			Keys:        []*rsa.PublicKey{getPublicKey(publicKey)},
			ExpectedErr: true,
			ExpectedOK:  false,
		},
	}

	for k, tc := range testCases {
		getter := serviceaccountcontroller.NewGetterFromClient(tc.Client)
		authenticator := serviceaccount.JWTTokenAuthenticator(tc.Keys, tc.Client != nil, getter)

		// An invalid, non-JWT token should always fail
		if _, ok, err := authenticator.AuthenticateToken("invalid token"); err != nil || ok {
			t.Errorf("%s: Expected err=nil, ok=false for non-JWT token", k)
			continue
		}

		user, ok, err := authenticator.AuthenticateToken(token)
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

		if user.GetName() != tc.ExpectedUserName {
			t.Errorf("%s: Expected username=%v, got %v", k, tc.ExpectedUserName, user.GetName())
			continue
		}
		if user.GetUID() != tc.ExpectedUserUID {
			t.Errorf("%s: Expected userUID=%v, got %v", k, tc.ExpectedUserUID, user.GetUID())
			continue
		}
		if !reflect.DeepEqual(user.GetGroups(), tc.ExpectedGroups) {
			t.Errorf("%s: Expected groups=%v, got %v", k, tc.ExpectedGroups, user.GetGroups())
			continue
		}
	}
}

func TestMakeSplitUsername(t *testing.T) {
	username := serviceaccount.MakeUsername("ns", "name")
	ns, name, err := serviceaccount.SplitUsername(username)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if ns != "ns" || name != "name" {
		t.Errorf("Expected ns/name, got %s/%s", ns, name)
	}

	invalid := []string{"test", "system:serviceaccount", "system:serviceaccount:", "system:serviceaccount:ns", "system:serviceaccount:ns:name:extra"}
	for _, n := range invalid {
		_, _, err := serviceaccount.SplitUsername("test")
		if err == nil {
			t.Errorf("Expected error for %s", n)
		}
	}
}
