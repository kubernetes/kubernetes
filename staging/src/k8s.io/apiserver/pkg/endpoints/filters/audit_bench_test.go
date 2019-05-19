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

package filters

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit/policy"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
)

type benchAuditSink struct {
	b *testing.B
}

func (s *benchAuditSink) ProcessEvents(evs ...*auditinternal.Event) bool {
	for _, ev := range evs {
		require.NotNil(s.b, ev)
	}
	return true
}

func BenchmarkAudit(b *testing.B) {
	shortRunningPath := "/api/v1/namespaces/default/pods/foo"

	for _, test := range []struct {
		desc           string
		path           string
		verb           string
		dynamicEnabled bool
	}{
		{
			"dynamic disabled",
			shortRunningPath,
			"GET",
			false,
		},
		{
			"dynamic enabled",
			shortRunningPath,
			"GET",
			true,
		},
	} {
		b.Run(test.desc, func(b *testing.B) {
			sink := &benchAuditSink{b}
			checker := policy.FakeChecker(auditinternal.LevelRequestResponse, []auditinternal.Stage{})
			if test.dynamicEnabled {
				checker = policy.NewDynamicChecker()
			}
			h := func(w http.ResponseWriter, req *http.Request) {
				w.Write([]byte("foo"))
			}
			handler := WithAudit(http.HandlerFunc(h), sink, checker, func(r *http.Request, ri *request.RequestInfo) bool {
				// simplified long-running check
				return ri.Verb == "watch"
			})
			req, _ := http.NewRequest(test.verb, test.path, nil)
			req = withTestContext(req, &user.DefaultInfo{Name: "admin"}, nil)
			req.RemoteAddr = "127.0.0.1"

			func() {
				defer func() {
					recover()
				}()
				for i := 0; i < b.N; i++ {
					handler.ServeHTTP(httptest.NewRecorder(), req)
				}
			}()
		})
	}
}
