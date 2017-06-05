// Copyright 2016 Google Inc. All Rights Reserved.
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

package logging

import (
	"testing"

	"golang.org/x/net/context"
	"google.golang.org/api/iterator"
)

func TestMonitoredResourceDescriptors(t *testing.T) {
	// We can't create MonitoredResourceDescriptors, and there is no guarantee
	// about what the service will return. So we just check that the result is
	// non-empty.
	it := client.ResourceDescriptors(context.Background())
	n := 0
loop:
	for {
		_, err := it.Next()
		switch err {
		case nil:
			n++
		case iterator.Done:
			break loop
		default:
			t.Fatal(err)
		}
	}
	if n == 0 {
		t.Fatal("Next: got no MetricResourceDescriptors, expected at least one")
	}
	// TODO(jba) test pagination.
}
