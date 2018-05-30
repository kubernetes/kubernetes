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

package audit

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestHandle(t *testing.T) {
	var registry audit.Registry = audit.WithRegistry(nil)
	var alwaysAllowAuthz authorizer.Authorizer = authorizerfactory.NewAlwaysAllowAuthorizer()
	var wg sync.WaitGroup
	var audits = AuditHandler{
		registry: registry,
		authz:    alwaysAllowAuthz,
	}
	events := []*auditinternal.Event{
		{AuditID: types.UID("my-fake-audit-id")},
	}

	req := httptest.NewRequest("GET", "http://127.0.0.1/audits", nil)
	req.RemoteAddr = "localhost:1234"
	ctx := req.Context()
	ctx = request.WithUser(ctx, &user.DefaultInfo{Name: "tom"})
	req = req.WithContext(ctx)
	w := newCloseNotifyingRecorder()
	wg.Add(1)
	go func() {
		audits.handle(w, req)
		wg.Done()
	}()

	// wait until the backend registered
	for !strings.Contains(fmt.Sprint(registry), req.RemoteAddr) {
		time.Sleep(time.Second)
	}
	registry.ProcessEvents(events...)
	registry.Shutdown()
	wg.Wait()

	resp := w.Result()
	body, _ := ioutil.ReadAll(resp.Body)
	assert.Equal(t, http.StatusOK, resp.StatusCode, "http request should success")
	assert.Contains(t, string(body), "my-fake-audit-id")
}

type closeNotifyingRecorder struct {
	*httptest.ResponseRecorder
	closed chan bool
}

func newCloseNotifyingRecorder() *closeNotifyingRecorder {
	return &closeNotifyingRecorder{
		httptest.NewRecorder(),
		make(chan bool, 1),
	}
}

func (c *closeNotifyingRecorder) close() {
	c.closed <- true
}

func (c *closeNotifyingRecorder) CloseNotify() <-chan bool {
	return c.closed
}
