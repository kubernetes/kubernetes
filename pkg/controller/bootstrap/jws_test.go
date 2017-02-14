/*
Copyright 2016 The Kubernetes Authors.

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

package bootstrap

import "testing"

const (
	content = "Hello from the other side. I must have called a thousand times."
	secret  = "my voice is my passcode"
	id      = "joshua"
)

func TestComputeDetachedSig(t *testing.T) {
	sig, err := computeDetachedSig(content, id, secret)
	if err != nil {
		t.Errorf("Error when computing signature: %v", err)
	}
	if sig != "eyJhbGciOiJIUzI1NiIsImtpZCI6Impvc2h1YSJ9..VShe2taLd-YTrmWuRkcL_8QTNDHYxQIEBsAYYiIj1_8" {
		t.Errorf("Wrong signature. Got: %v", sig)
	}

	// Try with null content
	sig, err = computeDetachedSig("", id, secret)
	if err != nil {
		t.Errorf("Error when computing signature: %v", err)
	}
	if sig != "eyJhbGciOiJIUzI1NiIsImtpZCI6Impvc2h1YSJ9..7Ui1ALizW4jXphVUB7xUqC9vLYLL9RZeOFfVLoB7Tgk" {
		t.Errorf("Wrong signature. Got: %v", sig)
	}

	// Try with no secret
	sig, err = computeDetachedSig(content, id, "")
	if err != nil {
		t.Errorf("Error when computing signature: %v", err)
	}
	if sig != "eyJhbGciOiJIUzI1NiIsImtpZCI6Impvc2h1YSJ9..UfkqvDGiIFxrMnFseDj9LYJOLNrvjW8aHhF71mvvAs8" {
		t.Errorf("Wrong signature. Got: %v", sig)
	}
}

func TestDetachedTokenIsValid(t *testing.T) {
	// Valid detached JWS token and valid inputs should succeed
	sig := "eyJhbGciOiJIUzI1NiIsImtpZCI6Impvc2h1YSJ9..VShe2taLd-YTrmWuRkcL_8QTNDHYxQIEBsAYYiIj1_8"
	if !DetachedTokenIsValid(sig, content, id, secret) {
		t.Errorf("Content %q and token \"%s:%s\" should equal signature: %q", content, id, secret, sig)
	}

	// Invalid detached JWS token and valid inputs should fail
	sig2 := sig + "foo"
	if DetachedTokenIsValid(sig2, content, id, secret) {
		t.Errorf("Content %q and token \"%s:%s\" should not equal signature: %q", content, id, secret, sig)
	}
}
