//go:build unix

/*
Copyright The Kubernetes Authors.

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

package remote

import (
	"context"
	"testing"
	"time"
)

// A killed ssh can leave a descendant holding its pipes; the deadline must still return.
func TestRunSSHCommandContextReturnsWhenADescendantHoldsThePipes(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 200*time.Millisecond)
	defer cancel()
	start := time.Now()
	// sh is the child the deadline kills; the backgrounded sleep inherits stdout and stderr.
	_, err := runSSHCommandContext(ctx, "test-host", "sh", "-c", "sleep 8 & wait")
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected an error after the deadline")
	}
	if elapsed > 4*time.Second {
		t.Fatalf("runSSHCommandContext returned after %v, want well under the sleep of the descendant", elapsed)
	}
}

// Without a deadline the call keeps waiting for the pipes, so a background child does not turn success into an error.
func TestRunSSHCommandContextWithoutDeadlineWaitsForThePipes(t *testing.T) {
	out, err := runSSHCommandContext(context.Background(), "test-host", "sh", "-c", "sleep 2 & echo ok")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "ok\n" {
		t.Fatalf("output = %q, want %q", out, "ok\n")
	}
}
