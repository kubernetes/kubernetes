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

package serviceaccount

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"gopkg.in/square/go-jose.v2/jwt"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/apis/core"
)

func init() {
	now = func() time.Time {
		// epoch time: 1514764800
		return time.Date(2018, time.January, 1, 0, 0, 0, 0, time.UTC)
	}
}

func TestClaims(t *testing.T) {
	sa := core.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "myns",
			Name:      "mysvcacct",
			UID:       "mysvcacct-uid",
		},
	}
	pod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "myns",
			Name:      "mypod",
			UID:       "mypod-uid",
		},
	}
	sec := &core.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "myns",
			Name:      "mysecret",
			UID:       "mysecret-uid",
		},
	}
	cs := []struct {
		// input
		sa        core.ServiceAccount
		pod       *core.Pod
		sec       *core.Secret
		exp       int64
		warnafter int64
		aud       []string
		// desired
		sc *jwt.Claims
		pc *privateClaims
	}{
		{
			// pod and secret
			sa:  sa,
			pod: pod,
			sec: sec,
			// really fast
			exp: 0,
			// nil audience
			aud: nil,

			sc: &jwt.Claims{
				Subject:   "system:serviceaccount:myns:mysvcacct",
				IssuedAt:  jwt.NumericDate(1514764800),
				NotBefore: jwt.NumericDate(1514764800),
				Expiry:    jwt.NumericDate(1514764800),
			},
			pc: &privateClaims{
				Kubernetes: kubernetes{
					Namespace: "myns",
					Svcacct:   ref{Name: "mysvcacct", UID: "mysvcacct-uid"},
					Pod:       &ref{Name: "mypod", UID: "mypod-uid"},
				},
			},
		},
		{
			// pod
			sa:  sa,
			pod: pod,
			// empty audience
			aud: []string{},
			exp: 100,

			sc: &jwt.Claims{
				Subject:   "system:serviceaccount:myns:mysvcacct",
				IssuedAt:  jwt.NumericDate(1514764800),
				NotBefore: jwt.NumericDate(1514764800),
				Expiry:    jwt.NumericDate(1514764800 + 100),
			},
			pc: &privateClaims{
				Kubernetes: kubernetes{
					Namespace: "myns",
					Svcacct:   ref{Name: "mysvcacct", UID: "mysvcacct-uid"},
					Pod:       &ref{Name: "mypod", UID: "mypod-uid"},
				},
			},
		},
		{
			// secret
			sa:  sa,
			sec: sec,
			exp: 100,
			// single member audience
			aud: []string{"1"},

			sc: &jwt.Claims{
				Subject:   "system:serviceaccount:myns:mysvcacct",
				Audience:  []string{"1"},
				IssuedAt:  jwt.NumericDate(1514764800),
				NotBefore: jwt.NumericDate(1514764800),
				Expiry:    jwt.NumericDate(1514764800 + 100),
			},
			pc: &privateClaims{
				Kubernetes: kubernetes{
					Namespace: "myns",
					Svcacct:   ref{Name: "mysvcacct", UID: "mysvcacct-uid"},
					Secret:    &ref{Name: "mysecret", UID: "mysecret-uid"},
				},
			},
		},
		{
			// no obj binding
			sa:  sa,
			exp: 100,
			// multimember audience
			aud: []string{"1", "2"},

			sc: &jwt.Claims{
				Subject:   "system:serviceaccount:myns:mysvcacct",
				Audience:  []string{"1", "2"},
				IssuedAt:  jwt.NumericDate(1514764800),
				NotBefore: jwt.NumericDate(1514764800),
				Expiry:    jwt.NumericDate(1514764800 + 100),
			},
			pc: &privateClaims{
				Kubernetes: kubernetes{
					Namespace: "myns",
					Svcacct:   ref{Name: "mysvcacct", UID: "mysvcacct-uid"},
				},
			},
		},
		{
			// warn after provided
			sa:        sa,
			pod:       pod,
			sec:       sec,
			exp:       60 * 60 * 24,
			warnafter: 60 * 60,
			// nil audience
			aud: nil,

			sc: &jwt.Claims{
				Subject:   "system:serviceaccount:myns:mysvcacct",
				IssuedAt:  jwt.NumericDate(1514764800),
				NotBefore: jwt.NumericDate(1514764800),
				Expiry:    jwt.NumericDate(1514764800 + 60*60*24),
			},
			pc: &privateClaims{
				Kubernetes: kubernetes{
					Namespace: "myns",
					Svcacct:   ref{Name: "mysvcacct", UID: "mysvcacct-uid"},
					Pod:       &ref{Name: "mypod", UID: "mypod-uid"},
					WarnAfter: jwt.NumericDate(1514764800 + 60*60),
				},
			},
		},
	}
	for i, c := range cs {
		t.Run(fmt.Sprintf("case %d", i), func(t *testing.T) {
			// comparing json spews has the benefit over
			// reflect.DeepEqual that we are also asserting that
			// claims structs are json serializable
			spew := func(obj interface{}) string {
				b, err := json.Marshal(obj)
				if err != nil {
					t.Fatalf("err, couldn't marshal claims: %v", err)
				}
				return string(b)
			}

			sc, pc := Claims(c.sa, c.pod, c.sec, c.exp, c.warnafter, c.aud)
			if spew(sc) != spew(c.sc) {
				t.Errorf("standard claims differed\n\tsaw:\t%s\n\twant:\t%s", spew(sc), spew(c.sc))
			}
			if spew(pc) != spew(c.pc) {
				t.Errorf("private claims differed\n\tsaw: %s\n\twant: %s", spew(pc), spew(c.pc))
			}
		})
	}
}

type deletionTestCase struct {
	name      string
	time      *metav1.Time
	expectErr bool
}

type claimTestCase struct {
	name      string
	getter    ServiceAccountTokenGetter
	private   *privateClaims
	expectErr bool
}

func TestValidatePrivateClaims(t *testing.T) {
	var (
		nowUnix = int64(1514764800)

		serviceAccount = &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "saname", Namespace: "ns", UID: "sauid"}}
		secret         = &v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "secretname", Namespace: "ns", UID: "secretuid"}}
		pod            = &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "podname", Namespace: "ns", UID: "poduid"}}
	)

	deletionTestCases := []deletionTestCase{
		{
			name: "valid",
			time: nil,
		},
		{
			name: "deleted now",
			time: &metav1.Time{Time: time.Unix(nowUnix, 0)},
		},
		{
			name: "deleted near past",
			time: &metav1.Time{Time: time.Unix(nowUnix-1, 0)},
		},
		{
			name: "deleted near future",
			time: &metav1.Time{Time: time.Unix(nowUnix+1, 0)},
		},
		{
			name: "deleted now-leeway",
			time: &metav1.Time{Time: time.Unix(nowUnix-60, 0)},
		},
		{
			name:      "deleted now-leeway-1",
			time:      &metav1.Time{Time: time.Unix(nowUnix-61, 0)},
			expectErr: true,
		},
	}

	testcases := []claimTestCase{
		{
			name:      "missing serviceaccount",
			getter:    fakeGetter{nil, nil, nil},
			private:   &privateClaims{Kubernetes: kubernetes{Svcacct: ref{Name: "saname", UID: "sauid"}, Namespace: "ns"}},
			expectErr: true,
		},
		{
			name:      "missing secret",
			getter:    fakeGetter{serviceAccount, nil, nil},
			private:   &privateClaims{Kubernetes: kubernetes{Svcacct: ref{Name: "saname", UID: "sauid"}, Secret: &ref{Name: "secretname", UID: "secretuid"}, Namespace: "ns"}},
			expectErr: true,
		},
		{
			name:      "missing pod",
			getter:    fakeGetter{serviceAccount, nil, nil},
			private:   &privateClaims{Kubernetes: kubernetes{Svcacct: ref{Name: "saname", UID: "sauid"}, Pod: &ref{Name: "podname", UID: "poduid"}, Namespace: "ns"}},
			expectErr: true,
		},
		{
			name:      "different uid serviceaccount",
			getter:    fakeGetter{serviceAccount, nil, nil},
			private:   &privateClaims{Kubernetes: kubernetes{Svcacct: ref{Name: "saname", UID: "sauidold"}, Namespace: "ns"}},
			expectErr: true,
		},
		{
			name:      "different uid secret",
			getter:    fakeGetter{serviceAccount, secret, nil},
			private:   &privateClaims{Kubernetes: kubernetes{Svcacct: ref{Name: "saname", UID: "sauid"}, Secret: &ref{Name: "secretname", UID: "secretuidold"}, Namespace: "ns"}},
			expectErr: true,
		},
		{
			name:      "different uid pod",
			getter:    fakeGetter{serviceAccount, nil, pod},
			private:   &privateClaims{Kubernetes: kubernetes{Svcacct: ref{Name: "saname", UID: "sauid"}, Pod: &ref{Name: "podname", UID: "poduidold"}, Namespace: "ns"}},
			expectErr: true,
		},
	}

	for _, deletionTestCase := range deletionTestCases {
		var (
			deletedServiceAccount = serviceAccount.DeepCopy()
			deletedPod            = pod.DeepCopy()
			deletedSecret         = secret.DeepCopy()
		)
		deletedServiceAccount.DeletionTimestamp = deletionTestCase.time
		deletedPod.DeletionTimestamp = deletionTestCase.time
		deletedSecret.DeletionTimestamp = deletionTestCase.time

		testcases = append(testcases,
			claimTestCase{
				name:      deletionTestCase.name + " serviceaccount",
				getter:    fakeGetter{deletedServiceAccount, nil, nil},
				private:   &privateClaims{Kubernetes: kubernetes{Svcacct: ref{Name: "saname", UID: "sauid"}, Namespace: "ns"}},
				expectErr: deletionTestCase.expectErr,
			},
			claimTestCase{
				name:      deletionTestCase.name + " secret",
				getter:    fakeGetter{serviceAccount, deletedSecret, nil},
				private:   &privateClaims{Kubernetes: kubernetes{Svcacct: ref{Name: "saname", UID: "sauid"}, Secret: &ref{Name: "secretname", UID: "secretuid"}, Namespace: "ns"}},
				expectErr: deletionTestCase.expectErr,
			},
			claimTestCase{
				name:      deletionTestCase.name + " pod",
				getter:    fakeGetter{serviceAccount, nil, deletedPod},
				private:   &privateClaims{Kubernetes: kubernetes{Svcacct: ref{Name: "saname", UID: "sauid"}, Pod: &ref{Name: "podname", UID: "poduid"}, Namespace: "ns"}},
				expectErr: deletionTestCase.expectErr,
			},
		)
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			v := &validator{tc.getter}
			_, err := v.Validate(context.Background(), "", &jwt.Claims{Expiry: jwt.NumericDate(nowUnix)}, tc.private)
			if err != nil && !tc.expectErr {
				t.Fatal(err)
			}
			if err == nil && tc.expectErr {
				t.Fatal("expected error, got none")
			}
			if err != nil {
				return
			}
		})
	}
}

type fakeGetter struct {
	serviceAccount *v1.ServiceAccount
	secret         *v1.Secret
	pod            *v1.Pod
}

func (f fakeGetter) GetServiceAccount(namespace, name string) (*v1.ServiceAccount, error) {
	if f.serviceAccount == nil {
		return nil, apierrors.NewNotFound(schema.GroupResource{Group: "", Resource: "serviceaccounts"}, name)
	}
	return f.serviceAccount, nil
}
func (f fakeGetter) GetPod(namespace, name string) (*v1.Pod, error) {
	if f.pod == nil {
		return nil, apierrors.NewNotFound(schema.GroupResource{Group: "", Resource: "pods"}, name)
	}
	return f.pod, nil
}
func (f fakeGetter) GetSecret(namespace, name string) (*v1.Secret, error) {
	if f.secret == nil {
		return nil, apierrors.NewNotFound(schema.GroupResource{Group: "", Resource: "secrets"}, name)
	}
	return f.secret, nil
}
