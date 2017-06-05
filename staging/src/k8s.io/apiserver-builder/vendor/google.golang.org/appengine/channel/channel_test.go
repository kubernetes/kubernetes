// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package channel

import (
	"testing"

	"google.golang.org/appengine/internal"
)

func TestRemapError(t *testing.T) {
	err := &internal.APIError{
		Service: "xmpp",
	}
	err = remapError(err).(*internal.APIError)
	if err.Service != "channel" {
		t.Errorf("err.Service = %q, want %q", err.Service, "channel")
	}
}
