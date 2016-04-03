// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cloud

import (
	"net/http"
	"testing"

	"google.golang.org/cloud/internal"
)

func TestClientTransportMutate(t *testing.T) {
	c := &http.Client{Transport: http.DefaultTransport}
	NewContext("project-id", c)
	NewContext("project-id", c)

	tr, ok := c.Transport.(*internal.Transport)
	if !ok {
		t.Errorf("Transport is expected to be an internal.Transport, found to be a %T", c.Transport)
	}
	if _, ok := tr.Base.(*internal.Transport); ok {
		t.Errorf("Transport's Base shouldn't have been an internal.Transport, found to be a %T", tr.Base)
	}
}
