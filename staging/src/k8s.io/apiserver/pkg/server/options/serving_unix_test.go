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
	"fmt"
	"net"
	"testing"
)

func TestCreateListenerSharePort(t *testing.T) {
	c := net.ListenConfig{Control: permitPortReuse}

	l, port, err := CreateListener("tcp", "127.0.0.1:0", c)
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}
	defer l.Close()

	l2, _, err := CreateListener("tcp", fmt.Sprintf("127.0.0.1:%d", port), c)
	if err != nil {
		t.Fatalf("failed to create 2nd listener: %v", err)
	}
	defer l2.Close()
}

func TestCreateListenerPreventUpgrades(t *testing.T) {
	l, port, err := CreateListener("tcp", "127.0.0.1:0", net.ListenConfig{})
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}
	defer l.Close()

	l2, _, err := CreateListener("tcp", fmt.Sprintf("127.0.0.1:%d", port), net.ListenConfig{Control: permitPortReuse})
	if err == nil {
		l2.Close()
		t.Fatalf("creating second listener without port sharing should fail")
	}
}
