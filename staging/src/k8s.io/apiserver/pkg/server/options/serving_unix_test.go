// +build !windows

/*
Copyright 2020 The Kubernetes Authors.

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

package options

import (
	"net"
	"testing"
)

func TestCreateListenerSharePort(t *testing.T) {
	addr := "127.0.0.1:12345"
	c := net.ListenConfig{Control: permitPortReuse}

	if _, _, err := CreateListener("tcp", addr, c); err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}

	if _, _, err := CreateListener("tcp", addr, c); err != nil {
		t.Fatalf("failed to create 2nd listener: %v", err)
	}
}

func TestCreateListenerPreventUpgrades(t *testing.T) {
	addr := "127.0.0.1:12346"

	if _, _, err := CreateListener("tcp", addr, net.ListenConfig{}); err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}

	if _, _, err := CreateListener("tcp", addr, net.ListenConfig{Control: permitPortReuse}); err == nil {
		t.Fatalf("creating second listener without port sharing should fail")
	}
}
