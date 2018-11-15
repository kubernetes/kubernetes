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
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"gopkg.in/square/go-jose.v2/jwt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
		sa  core.ServiceAccount
		pod *core.Pod
		sec *core.Secret
		exp int64
		aud []string
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

			sc, pc := Claims(c.sa, c.pod, c.sec, c.exp, c.aud)
			if spew(sc) != spew(c.sc) {
				t.Errorf("standard claims differed\n\tsaw:\t%s\n\twant:\t%s", spew(sc), spew(c.sc))
			}
			if spew(pc) != spew(c.pc) {
				t.Errorf("private claims differed\n\tsaw: %s\n\twant: %s", spew(pc), spew(c.pc))
			}
		})
	}
}
