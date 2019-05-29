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

package responsewriters

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// Avoid emitting errors that look like valid HTML. Quotes are okay.
var sanitizer = strings.NewReplacer(`&`, "&amp;", `<`, "&lt;", `>`, "&gt;")

// Forbidden renders a simple forbidden error
func Forbidden(ctx context.Context, attributes authorizer.Attributes, w http.ResponseWriter, req *http.Request, reason string, s runtime.NegotiatedSerializer) {
	msg := sanitizer.Replace(forbiddenMessage(attributes))
	w.Header().Set("X-Content-Type-Options", "nosniff")

	var errMsg string
	if len(reason) == 0 {
		errMsg = fmt.Sprintf("%s", msg)
	} else {
		errMsg = fmt.Sprintf("%s: %s", msg, reason)
	}
	gv := schema.GroupVersion{Group: attributes.GetAPIGroup(), Version: attributes.GetAPIVersion()}
	gr := schema.GroupResource{Group: attributes.GetAPIGroup(), Resource: attributes.GetResource()}
	ErrorNegotiated(apierrors.NewForbidden(gr, attributes.GetName(), fmt.Errorf(errMsg)), s, gv, w, req)
}

func forbiddenMessage(attributes authorizer.Attributes) string {
	username := ""
	if user := attributes.GetUser(); user != nil {
		username = user.GetName()
	}

	if !attributes.IsResourceRequest() {
		return fmt.Sprintf("User %q cannot %s path %q", username, attributes.GetVerb(), attributes.GetPath())
	}

	resource := attributes.GetResource()
	if subresource := attributes.GetSubresource(); len(subresource) > 0 {
		resource = resource + "/" + subresource
	}

	if ns := attributes.GetNamespace(); len(ns) > 0 {
		return fmt.Sprintf("User %q cannot %s resource %q in API group %q in the namespace %q", username, attributes.GetVerb(), resource, attributes.GetAPIGroup(), ns)
	}

	return fmt.Sprintf("User %q cannot %s resource %q in API group %q at the cluster scope", username, attributes.GetVerb(), resource, attributes.GetAPIGroup())
}

// InternalError renders a simple internal error
func InternalError(w http.ResponseWriter, req *http.Request, err error) {
	http.Error(w, sanitizer.Replace(fmt.Sprintf("Internal Server Error: %q: %v", req.RequestURI, err)),
		http.StatusInternalServerError)
	utilruntime.HandleError(err)
}
