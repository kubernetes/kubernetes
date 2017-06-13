/*
Copyright 2017 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/pborman/uuid"

	"reflect"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/apis/audit"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	authenticationv1 "k8s.io/client-go/pkg/apis/authentication/v1"
)

func NewEventFromRequest(req *http.Request, level auditinternal.Level, attribs authorizer.Attributes) (*auditinternal.Event, error) {
	ev := &auditinternal.Event{
		Timestamp:  metav1.NewTime(time.Now()),
		Verb:       attribs.GetVerb(),
		RequestURI: req.URL.RequestURI(),
	}

	ev.Level = level

	// prefer the id from the headers. If not available, create a new one.
	// TODO(audit): do we want to forbid the header for non-front-proxy users?
	ids := req.Header[auditinternal.HeaderAuditID]
	if len(ids) > 0 {
		ev.AuditID = types.UID(ids[0])
	} else {
		ev.AuditID = types.UID(uuid.NewRandom().String())
	}

	ips := utilnet.SourceIPs(req)
	ev.SourceIPs = make([]string, len(ips))
	for i := range ips {
		ev.SourceIPs[i] = ips[i].String()
	}

	if user := attribs.GetUser(); user != nil {
		ev.User.Username = user.GetName()
		ev.User.Extra = map[string]auditinternal.ExtraValue{}
		for k, v := range user.GetExtra() {
			ev.User.Extra[k] = auditinternal.ExtraValue(v)
		}
		ev.User.Groups = user.GetGroups()
		ev.User.UID = user.GetUID()
	}

	if asuser := req.Header.Get(authenticationv1.ImpersonateUserHeader); len(asuser) > 0 {
		ev.ImpersonatedUser = &auditinternal.UserInfo{
			Username: asuser,
		}
		if requestedGroups := req.Header[authenticationv1.ImpersonateGroupHeader]; len(requestedGroups) > 0 {
			ev.ImpersonatedUser.Groups = requestedGroups
		}

		ev.ImpersonatedUser.Extra = map[string]auditinternal.ExtraValue{}
		for k, v := range req.Header {
			if !strings.HasPrefix(k, authenticationv1.ImpersonateUserExtraHeaderPrefix) {
				continue
			}
			k = k[len(authenticationv1.ImpersonateUserExtraHeaderPrefix):]
			ev.ImpersonatedUser.Extra[k] = auditinternal.ExtraValue(v)
		}
	}

	if attribs.IsResourceRequest() {
		ev.ObjectRef = &auditinternal.ObjectReference{
			Namespace:   attribs.GetNamespace(),
			Name:        attribs.GetName(),
			Resource:    attribs.GetResource(),
			Subresource: attribs.GetSubresource(),
			APIVersion:  attribs.GetAPIGroup() + "/" + attribs.GetAPIVersion(),
		}
	}

	return ev, nil
}

// LogRequestObject fills in the request object into an audit event. The passed runtime.Object
// will be converted to the given gv.
func LogRequestObject(ae *audit.Event, obj runtime.Object, gvr schema.GroupVersionResource, subresource string, s runtime.NegotiatedSerializer) {
	if ae == nil || ae.Level.Less(audit.LevelMetadata) {
		return
	}

	// complete ObjectRef
	if ae.ObjectRef == nil {
		ae.ObjectRef = &audit.ObjectReference{}
	}
	if acc, ok := obj.(metav1.ObjectMetaAccessor); ok {
		meta := acc.GetObjectMeta()
		if len(ae.ObjectRef.Namespace) == 0 {
			ae.ObjectRef.Namespace = meta.GetNamespace()
		}
		if len(ae.ObjectRef.Name) == 0 {
			ae.ObjectRef.Name = meta.GetName()
		}
		if len(ae.ObjectRef.UID) == 0 {
			ae.ObjectRef.UID = meta.GetUID()
		}
		if len(ae.ObjectRef.ResourceVersion) == 0 {
			ae.ObjectRef.ResourceVersion = meta.GetResourceVersion()
		}
	}
	// TODO: ObjectRef should include the API group.
	if len(ae.ObjectRef.APIVersion) == 0 {
		ae.ObjectRef.APIVersion = gvr.Version
	}
	if len(ae.ObjectRef.Resource) == 0 {
		ae.ObjectRef.Resource = gvr.Resource
	}
	if len(ae.ObjectRef.Subresource) == 0 {
		ae.ObjectRef.Subresource = subresource
	}

	if ae.Level.Less(audit.LevelRequest) {
		return
	}

	// TODO(audit): hook into the serializer to avoid double conversion
	var err error
	ae.RequestObject, err = encodeObject(obj, gvr.GroupVersion(), s)
	if err != nil {
		// TODO(audit): add error slice to audit event struct
		glog.Warningf("Auditing failed of %v request: %v", reflect.TypeOf(obj).Name(), err)
		return
	}
}

// LogRquestPatch fills in the given patch as the request object into an audit event.
func LogRequestPatch(ae *audit.Event, patch []byte) {
	if ae == nil || ae.Level.Less(audit.LevelRequest) {
		return
	}

	ae.RequestObject = &runtime.Unknown{
		Raw:         patch,
		ContentType: runtime.ContentTypeJSON,
	}
}

// LogResponseObject fills in the response object into an audit event. The passed runtime.Object
// will be converted to the given gv.
func LogResponseObject(ae *audit.Event, obj runtime.Object, gv schema.GroupVersion, s runtime.NegotiatedSerializer) {
	if ae == nil || ae.Level.Less(audit.LevelRequestResponse) {
		return
	}

	if status, ok := obj.(*metav1.Status); ok {
		ae.ResponseStatus = status
	}

	// TODO(audit): hook into the serializer to avoid double conversion
	var err error
	ae.ResponseObject, err = encodeObject(obj, gv, s)
	if err != nil {
		glog.Warningf("Audit failed for %q response: %v", reflect.TypeOf(obj).Name(), err)
	}
}

func encodeObject(obj runtime.Object, gv schema.GroupVersion, serializer runtime.NegotiatedSerializer) (*runtime.Unknown, error) {
	supported := serializer.SupportedMediaTypes()
	for i := range supported {
		if supported[i].MediaType == "application/json" {
			enc := serializer.EncoderForVersion(supported[i].Serializer, gv)
			var buf bytes.Buffer
			if err := enc.Encode(obj, &buf); err != nil {
				return nil, fmt.Errorf("encoding failed: %v", err)
			}

			return &runtime.Unknown{
				Raw:         buf.Bytes(),
				ContentType: runtime.ContentTypeJSON,
			}, nil
		}
	}
	return nil, fmt.Errorf("no json encoder found")
}
