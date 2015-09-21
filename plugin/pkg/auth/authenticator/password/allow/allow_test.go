/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package allow

import "testing"

func TestAllowEmpty(t *testing.T) {
	allow := NewAllow()
	user, ok, err := allow.AuthenticatePassword("", "")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if ok {
		t.Fatalf("Unexpected success")
	}
	if user != nil {
		t.Fatalf("Unexpected user: %v", user)
	}
}

func TestAllowPresent(t *testing.T) {
	allow := NewAllow()
	user, ok, err := allow.AuthenticatePassword("myuser", "")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if !ok {
		t.Fatalf("Unexpected failure")
	}
	if user.GetName() != "myuser" || user.GetUID() != "myuser" {
		t.Fatalf("Unexpected user name or uid: %v", user)
	}
}
