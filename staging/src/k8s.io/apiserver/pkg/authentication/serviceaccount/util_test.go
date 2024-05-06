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

package serviceaccount

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
)

func TestUserInfo(t *testing.T) {
	tests := map[string]struct {
		info             ServiceAccountInfo
		expectedUserInfo *user.DefaultInfo
	}{
		"extracts pod name/uid": {
			info: ServiceAccountInfo{Name: "name", Namespace: "ns", PodName: "test", PodUID: "uid"},
			expectedUserInfo: &user.DefaultInfo{
				Name:   "system:serviceaccount:ns:name",
				Groups: []string{"system:serviceaccounts", "system:serviceaccounts:ns"},
				Extra: map[string][]string{
					"authentication.kubernetes.io/pod-name": {"test"},
					"authentication.kubernetes.io/pod-uid":  {"uid"},
				},
			},
		},
		"extracts node name/uid": {
			info: ServiceAccountInfo{Name: "name", Namespace: "ns", NodeName: "test", NodeUID: "uid"},
			expectedUserInfo: &user.DefaultInfo{
				Name:   "system:serviceaccount:ns:name",
				Groups: []string{"system:serviceaccounts", "system:serviceaccounts:ns"},
				Extra: map[string][]string{
					"authentication.kubernetes.io/node-name": {"test"},
					"authentication.kubernetes.io/node-uid":  {"uid"},
				},
			},
		},
		"extracts node name only": {
			info: ServiceAccountInfo{Name: "name", Namespace: "ns", NodeName: "test"},
			expectedUserInfo: &user.DefaultInfo{
				Name:   "system:serviceaccount:ns:name",
				Groups: []string{"system:serviceaccounts", "system:serviceaccounts:ns"},
				Extra: map[string][]string{
					"authentication.kubernetes.io/node-name": {"test"},
				},
			},
		},
		"does not extract node UID if name is not set": {
			info: ServiceAccountInfo{Name: "name", Namespace: "ns", NodeUID: "test"},
			expectedUserInfo: &user.DefaultInfo{
				Name:   "system:serviceaccount:ns:name",
				Groups: []string{"system:serviceaccounts", "system:serviceaccounts:ns"},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			userInfo := test.info.UserInfo()
			if !reflect.DeepEqual(userInfo, test.expectedUserInfo) {
				t.Errorf("expected %#v but got %#v", test.expectedUserInfo, userInfo)
			}
		})
	}
}

func TestMakeUsername(t *testing.T) {

	testCases := map[string]struct {
		Namespace   string
		Name        string
		ExpectedErr bool
	}{
		"valid": {
			Namespace:   "foo",
			Name:        "bar",
			ExpectedErr: false,
		},
		"empty": {
			ExpectedErr: true,
		},
		"empty namespace": {
			Namespace:   "",
			Name:        "foo",
			ExpectedErr: true,
		},
		"empty name": {
			Namespace:   "foo",
			Name:        "",
			ExpectedErr: true,
		},
		"extra segments": {
			Namespace:   "foo",
			Name:        "bar:baz",
			ExpectedErr: true,
		},
		"invalid chars in namespace": {
			Namespace:   "foo ",
			Name:        "bar",
			ExpectedErr: true,
		},
		"invalid chars in name": {
			Namespace:   "foo",
			Name:        "bar ",
			ExpectedErr: true,
		},
	}

	for k, tc := range testCases {
		username := MakeUsername(tc.Namespace, tc.Name)
		if !MatchesUsername(tc.Namespace, tc.Name, username) {
			t.Errorf("%s: Expected to match username", k)
		}
		namespace, name, err := SplitUsername(username)
		if (err != nil) != tc.ExpectedErr {
			t.Errorf("%s: Expected error=%v, got %v", k, tc.ExpectedErr, err)
			continue
		}
		if err != nil {
			continue
		}

		if namespace != tc.Namespace {
			t.Errorf("%s: Expected namespace %q, got %q", k, tc.Namespace, namespace)
		}
		if name != tc.Name {
			t.Errorf("%s: Expected name %q, got %q", k, tc.Name, name)
		}
	}
}

func TestMatchUsername(t *testing.T) {

	testCases := []struct {
		TestName  string
		Namespace string
		Name      string
		Username  string
		Expect    bool
	}{
		{Namespace: "foo", Name: "bar", Username: "foo", Expect: false},
		{Namespace: "foo", Name: "bar", Username: "system:serviceaccount:", Expect: false},
		{Namespace: "foo", Name: "bar", Username: "system:serviceaccount:foo", Expect: false},
		{Namespace: "foo", Name: "bar", Username: "system:serviceaccount:foo:", Expect: false},
		{Namespace: "foo", Name: "bar", Username: "system:serviceaccount:foo:bar", Expect: true},
		{Namespace: "foo", Name: "bar", Username: "system:serviceaccount::bar", Expect: false},
		{Namespace: "foo", Name: "bar", Username: "system:serviceaccount:bar", Expect: false},
		{Namespace: "foo", Name: "bar", Username: ":bar", Expect: false},
		{Namespace: "foo", Name: "bar", Username: "foo:bar", Expect: false},
		{Namespace: "foo", Name: "bar", Username: "", Expect: false},

		{Namespace: "foo2", Name: "bar", Username: "system:serviceaccount:foo:bar", Expect: false},
		{Namespace: "foo", Name: "bar2", Username: "system:serviceaccount:foo:bar", Expect: false},
		{Namespace: "foo:", Name: "bar", Username: "system:serviceaccount:foo:bar", Expect: false},
		{Namespace: "foo", Name: ":bar", Username: "system:serviceaccount:foo:bar", Expect: false},
	}

	for _, tc := range testCases {
		t.Run(tc.TestName, func(t *testing.T) {
			actual := MatchesUsername(tc.Namespace, tc.Name, tc.Username)
			if actual != tc.Expect {
				t.Fatalf("unexpected match")
			}
		})
	}
}

func TestIsServiceAccountToken(t *testing.T) {

	secretIns := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "token-secret-1",
			Namespace:       "default",
			UID:             "23456",
			ResourceVersion: "1",
			Annotations: map[string]string{
				v1.ServiceAccountNameKey: "default",
				v1.ServiceAccountUIDKey:  "12345",
			},
		},
		Type: v1.SecretTypeServiceAccountToken,
		Data: map[string][]byte{
			"token":     []byte("ABC"),
			"ca.crt":    []byte("CA Data"),
			"namespace": []byte("default"),
		},
	}

	secretTypeMistmatch := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "token-secret-2",
			Namespace:       "default",
			UID:             "23456",
			ResourceVersion: "1",
			Annotations: map[string]string{
				v1.ServiceAccountNameKey: "default",
				v1.ServiceAccountUIDKey:  "12345",
			},
		},
		Type: v1.SecretTypeOpaque,
	}

	saIns := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "default",
			UID:             "12345",
			Namespace:       "default",
			ResourceVersion: "1",
		},
	}

	saInsNameNotEqual := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "non-default",
			UID:             "12345",
			Namespace:       "default",
			ResourceVersion: "1",
		},
	}

	saInsUIDNotEqual := &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "default",
			UID:             "67890",
			Namespace:       "default",
			ResourceVersion: "1",
		},
	}

	tests := map[string]struct {
		secret *v1.Secret
		sa     *v1.ServiceAccount
		expect bool
	}{
		"correct service account": {
			secret: secretIns,
			sa:     saIns,
			expect: true,
		},
		"service account name not equal": {
			secret: secretIns,
			sa:     saInsNameNotEqual,
			expect: false,
		},
		"service account uid not equal": {
			secret: secretIns,
			sa:     saInsUIDNotEqual,
			expect: false,
		},
		"service account type not equal": {
			secret: secretTypeMistmatch,
			sa:     saIns,
			expect: false,
		},
	}

	for k, v := range tests {
		actual := IsServiceAccountToken(v.secret, v.sa)
		if actual != v.expect {
			t.Errorf("%s failed, expected %t but received %t", k, v.expect, actual)
		}
	}

}
