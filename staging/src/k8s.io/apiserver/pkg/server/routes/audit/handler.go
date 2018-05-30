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
	"context"
	"fmt"
	"net/http"
	"net/url"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/apis/audit/install"
	auditv1beta1 "k8s.io/apiserver/pkg/apis/audit/v1beta1"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/mux"
)

func init() {
	install.Install(audit.Scheme)
}

// AuditHandler provides a pull based audit event stream
type AuditHandler struct {
	registry audit.Registry
	authz    authorizer.Authorizer
}

// Install installs the /audits http handers
func (a AuditHandler) Install(c *mux.PathRecorderMux, b audit.Backend, authz authorizer.Authorizer) {
	if registry, ok := b.(audit.Registry); ok {
		a.registry = registry
	} else {
		return
	}
	a.authz = authz
	c.HandleFunc("/audits", a.handle)
}

func (a AuditHandler) handle(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		w.WriteHeader(http.StatusNotAcceptable)
		fmt.Fprint(w, "unsupported http method")
		return
	}
	qm := newQueryParm(req.URL.Query())
	ctx := req.Context()
	decision, reason, err := a.authorize(ctx, qm)
	if decision != authorizer.DecisionAllow {
		if err != nil {
			responsewriters.InternalError(w, req, err)
			return
		}
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprint(w, fmt.Sprintf("Forbidden to read audit events. %s", reason))
		return
	}

	backend := newBackend(req.RemoteAddr)
	a.registry.Register(backend)
	glog.V(5).Infof("Registered backend %v for audit pull stream", backend)

	defer func() {
		defer utilruntime.HandleCrash()
		a.registry.UnRegister(backend)
		glog.V(5).Infof("UnRegistered backend %v.", backend)
		close(backend.conClosed)
	}()

	var notify <-chan bool
	if cn, ok := w.(http.CloseNotifier); ok {
		notify = cn.CloseNotify()
	}

	for {
		select {
		case <-backend.shutdown:
			return
		case <-notify:
			return
		case event := <-backend.buffer:
			if !qm.match(event) {
				continue
			}
			bs, err := runtime.Encode(audit.Codecs.LegacyCodec(auditv1beta1.SchemeGroupVersion), event)
			if err != nil {
				fmt.Fprint(w, err.Error())
			} else {
				fmt.Fprint(w, string(bs[:]))
			}
			if flush, ok := w.(http.Flusher); ok {
				flush.Flush()
			}
		}
	}
}

func (a AuditHandler) authorize(ctx context.Context, qp queryParm) (decision authorizer.Decision, reason string, err error) {
	attribs := authorizer.AttributesRecord{}

	user, ok := request.UserFrom(ctx)
	if ok {
		attribs.User = user
	} else {
		return authorizer.DecisionNoOpinion, "", fmt.Errorf("Failed to found user info from context")
	}

	if qp.username == user.GetName() {
		return authorizer.DecisionAllow, "", nil
	}
	// Start with common attributes that apply to resource and non-resource requests
	attribs.ResourceRequest = true
	attribs.Verb = "list"

	attribs.APIGroup = "audit.k8s.io"
	attribs.APIVersion = "v1beta1"
	attribs.Resource = "events"
	attribs.Subresource = ""
	if len(qp.namespace) > 0 && qp.namespace != "<none>" {
		attribs.Namespace = qp.namespace
	}

	decision, reason, err = a.authz.Authorize(attribs)
	if decision != authorizer.DecisionAllow && len(attribs.Namespace) > 0 {
		// if one can get events from cluster scope, he can get events from any namespace
		attribs.Namespace = ""
		if decision, reason, err := a.authz.Authorize(attribs); decision == authorizer.DecisionAllow {
			return decision, reason, err
		}
	}
	return decision, reason, err
}

type queryParm struct {
	// username of the audit event
	username string
	// group of the audit event
	group string
	// namespace of the audit event, <none> for cluster scoped request, empty string for all namespaces
	namespace string
	// apiGroup of the audit event, <core> for the core api group.
	apiGroup string
	// resource of the audit event, for example: pods.
	resource string
}

func newQueryParm(values url.Values) queryParm {
	return queryParm{
		username:  values.Get("username"),
		group:     values.Get("group"),
		namespace: values.Get("namespace"),
		apiGroup:  values.Get("apigroup"),
		resource:  values.Get("resource"),
	}
}

func (q queryParm) match(ev *auditinternal.Event) bool {
	if len(q.username) > 0 {
		if ev.User.Username != q.username {
			return false
		}
	}
	if len(q.group) > 0 {
		found := false
		for _, group := range ev.User.Groups {
			if group == q.group {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	if len(q.namespace) > 0 {
		if q.namespace == "<none>" {
			if ev.ObjectRef != nil && len(ev.ObjectRef.Namespace) > 0 {
				return false
			}
		} else {
			if ev.ObjectRef == nil || q.namespace != ev.ObjectRef.Namespace {
				return false
			}
		}
	}
	if len(q.apiGroup) > 0 {
		if q.apiGroup == "<core>" {
			if ev.ObjectRef == nil || len(ev.ObjectRef.APIGroup) > 0 {
				return false
			}
		} else {
			if ev.ObjectRef == nil || q.apiGroup != ev.ObjectRef.APIGroup {
				return false
			}
		}
	}
	if len(q.resource) > 0 {
		if ev.ObjectRef == nil || q.resource != ev.ObjectRef.Resource {
			return false
		}
	}
	return true
}
