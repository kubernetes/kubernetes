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
	"time"

	"github.com/pborman/uuid"
	"k8s.io/klog"

	"reflect"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

const (
	maxUserAgentLength      = 1024
	userAgentTruncateSuffix = "...TRUNCATED"
)

func NewEventFromRequest(req *http.Request, level auditinternal.Level, attribs authorizer.Attributes) (*auditinternal.Event, error) {
	ev := &auditinternal.Event{
		RequestReceivedTimestamp: metav1.NewMicroTime(time.Now()),
		Verb:                     attribs.GetVerb(),
		RequestURI:               req.URL.RequestURI(),
		UserAgent:                maybeTruncateUserAgent(req),
		Level:                    level,
	}

	// prefer the id from the headers. If not available, create a new one.
	// TODO(audit): do we want to forbid the header for non-front-proxy users?
	ids := req.Header.Get(auditinternal.HeaderAuditID)
	if ids != "" {
		ev.AuditID = types.UID(ids)
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

	if attribs.IsResourceRequest() {
		ev.ObjectRef = &auditinternal.ObjectReference{
			Namespace:   attribs.GetNamespace(),
			Name:        attribs.GetName(),
			Resource:    attribs.GetResource(),
			Subresource: attribs.GetSubresource(),
			APIGroup:    attribs.GetAPIGroup(),
			APIVersion:  attribs.GetAPIVersion(),
		}
	}

	return ev, nil
}

// LogImpersonatedUser fills in the impersonated user attributes into an audit event.
func LogImpersonatedUser(ae *auditinternal.Event, user user.Info) {
	if ae == nil || ae.Level.Less(auditinternal.LevelMetadata) {
		return
	}
	ae.ImpersonatedUser = &auditinternal.UserInfo{
		Username: user.GetName(),
	}
	ae.ImpersonatedUser.Groups = user.GetGroups()
	ae.ImpersonatedUser.UID = user.GetUID()
	ae.ImpersonatedUser.Extra = map[string]auditinternal.ExtraValue{}
	for k, v := range user.GetExtra() {
		ae.ImpersonatedUser.Extra[k] = auditinternal.ExtraValue(v)
	}
}

// LogRequestObject fills in the request object into an audit event. The passed runtime.Object
// will be converted to the given gv.
func LogRequestObject(ae *auditinternal.Event, obj runtime.Object, gvr schema.GroupVersionResource, subresource string, s runtime.NegotiatedSerializer) {
	if ae == nil || ae.Level.Less(auditinternal.LevelMetadata) {
		return
	}

	// complete ObjectRef
	if ae.ObjectRef == nil {
		ae.ObjectRef = &auditinternal.ObjectReference{}
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
	if len(ae.ObjectRef.APIVersion) == 0 {
		ae.ObjectRef.APIGroup = gvr.Group
		ae.ObjectRef.APIVersion = gvr.Version
	}
	if len(ae.ObjectRef.Resource) == 0 {
		ae.ObjectRef.Resource = gvr.Resource
	}
	if len(ae.ObjectRef.Subresource) == 0 {
		ae.ObjectRef.Subresource = subresource
	}

	if ae.Level.Less(auditinternal.LevelRequest) {
		return
	}

	// TODO(audit): hook into the serializer to avoid double conversion
	var err error
	ae.RequestObject, err = encodeObject(obj, gvr.GroupVersion(), s)
	if err != nil {
		// TODO(audit): add error slice to audit event struct
		klog.Warningf("Auditing failed of %v request: %v", reflect.TypeOf(obj).Name(), err)
		return
	}
}

// LogRequestPatch fills in the given patch as the request object into an audit event.
func LogRequestPatch(ae *auditinternal.Event, patch []byte) {
	if ae == nil || ae.Level.Less(auditinternal.LevelRequest) {
		return
	}

	ae.RequestObject = &runtime.Unknown{
		Raw:         patch,
		ContentType: runtime.ContentTypeJSON,
	}
}

// LogResponseObject fills in the response object into an audit event. The passed runtime.Object
// will be converted to the given gv.
func LogResponseObject(ae *auditinternal.Event, obj runtime.Object, gv schema.GroupVersion, s runtime.NegotiatedSerializer) {
	if ae == nil || ae.Level.Less(auditinternal.LevelMetadata) {
		return
	}
	if status, ok := obj.(*metav1.Status); ok {
		// selectively copy the bounded fields.
		ae.ResponseStatus = &metav1.Status{
			Status: status.Status,
			Reason: status.Reason,
			Code:   status.Code,
		}
	}

	if ae.Level.Less(auditinternal.LevelRequestResponse) {
		return
	}
	// TODO(audit): hook into the serializer to avoid double conversion
	var err error
	ae.ResponseObject, err = encodeObject(obj, gv, s)
	if err != nil {
		klog.Warningf("Audit failed for %q response: %v", reflect.TypeOf(obj).Name(), err)
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

// LogAnnotation fills in the Annotations according to the key value pair.
func LogAnnotation(ae *auditinternal.Event, key, value string) {
	if ae == nil || ae.Level.Less(auditinternal.LevelMetadata) {
		return
	}
	if ae.Annotations == nil {
		ae.Annotations = make(map[string]string)
	}
	if v, ok := ae.Annotations[key]; ok && v != value {
		klog.Warningf("Failed to set annotations[%q] to %q for audit:%q, it has already been set to %q", key, value, ae.AuditID, ae.Annotations[key])
		return
	}
	ae.Annotations[key] = value
}

// LogAnnotations fills in the Annotations according to the annotations map.
func LogAnnotations(ae *auditinternal.Event, annotations map[string]string) {
	if ae == nil || ae.Level.Less(auditinternal.LevelMetadata) {
		return
	}
	for key, value := range annotations {
		LogAnnotation(ae, key, value)
	}
}

// truncate User-Agent if too long, otherwise return it directly.
func maybeTruncateUserAgent(req *http.Request) string {
	ua := req.UserAgent()
	if len(ua) > maxUserAgentLength {
		ua = ua[:maxUserAgentLength] + userAgentTruncateSuffix
	}

	return ua
}
