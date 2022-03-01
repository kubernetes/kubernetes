/*
Copyright 2021 The Kubernetes Authors.

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

	"k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/rest"
	netutils "k8s.io/utils/net"
)

func TestEmptyMainCert(t *testing.T) {
	secureServingInfo := &server.SecureServingInfo{}
	var loopbackClientConfig *rest.Config

	s := (&SecureServingOptions{
		BindAddress: netutils.ParseIPSloppy("127.0.0.1"),
	}).WithLoopback()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen on 127.0.0.1:0")
	}
	defer ln.Close()
	s.Listener = ln
	s.BindPort = ln.Addr().(*net.TCPAddr).Port

	if err := s.ApplyTo(&secureServingInfo, &loopbackClientConfig); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if loopbackClientConfig == nil {
		t.Errorf("unexpected empty loopbackClientConfig")
	}
	if e, a := 1, len(secureServingInfo.SNICerts); e != a {
		t.Errorf("expected %d SNICert, got %d", e, a)
	}
}
