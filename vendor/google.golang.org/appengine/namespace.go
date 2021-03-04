// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package appengine

import (
	"fmt"
	"regexp"

	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
)

// Namespace returns a replacement context that operates within the given namespace.
func Namespace(c context.Context, namespace string) (context.Context, error) {
	if !validNamespace.MatchString(namespace) {
		return nil, fmt.Errorf("appengine: namespace %q does not match /%s/", namespace, validNamespace)
	}
	return internal.NamespacedContext(c, namespace), nil
}

// validNamespace matches valid namespace names.
var validNamespace = regexp.MustCompile(`^[0-9A-Za-z._-]{0,100}$`)
