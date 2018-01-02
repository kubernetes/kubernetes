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

package storage

import (
	"fmt"
	"net/http"
	"testing"

	"golang.org/x/net/context"

	"google.golang.org/api/option"
)

type fakeTransport struct{}

func (t *fakeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("error handling request")
}

func TestErrorOnObjectsInsertCall(t *testing.T) {
	ctx := context.Background()
	hc := &http.Client{Transport: &fakeTransport{}}
	client, err := NewClient(ctx, option.WithHTTPClient(hc))
	if err != nil {
		t.Errorf("error when creating client: %v", err)
	}
	wc := client.Bucket("bucketname").Object("filename1").NewWriter(ctx)
	wc.ContentType = "text/plain"

	// We can't check that the Write fails, since it depends on the write to the
	// underling fakeTransport failing which is racy.
	wc.Write([]byte("hello world"))

	// Close must always return an error though since it waits for the transport to
	// have closed.
	if err := wc.Close(); err == nil {
		t.Errorf("expected error on close, got nil")
	}
}
