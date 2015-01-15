// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package appengine

import (
	"fmt"
	"regexp"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine/internal"
	basepb "google.golang.org/appengine/internal/base"
)

// Namespace returns a replacement context that operates within the given namespace.
func Namespace(c Context, namespace string) (Context, error) {
	if !validNamespace.MatchString(namespace) {
		return nil, fmt.Errorf("appengine: namespace %q does not match /%s/", namespace, validNamespace)
	}
	return &namespacedContext{
		Context:   c,
		namespace: namespace,
	}, nil
}

// validNamespace matches valid namespace names.
var validNamespace = regexp.MustCompile(`^[0-9A-Za-z._-]{0,100}$`)

// namespacedContext wraps a Context to support namespaces.
type namespacedContext struct {
	Context
	namespace string
}

func (n *namespacedContext) Call(service, method string, in, out proto.Message, opts *internal.CallOptions) error {
	// Apply any namespace mods.
	if mod, ok := internal.NamespaceMods[service]; ok {
		mod(in, n.namespace)
	}
	if service == "__go__" && method == "GetNamespace" {
		out.(*basepb.StringProto).Value = proto.String(n.namespace)
		return nil
	}

	return n.Context.Call(service, method, in, out, opts)
}
