/*
Copyright 2014 The Kubernetes Authors.

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

package client

import (
	"testing"

	restclient "k8s.io/client-go/rest"
)

func TestMakeTransportInvalid(t *testing.T) {
	config := &KubeletClientConfig{
		EnableHttps: true,
		//Invalid certificate and key path
		TLSClientConfig: restclient.TLSClientConfig{
			CertFile: "../../client/testdata/mycertinvalid.cer",
			KeyFile:  "../../client/testdata/mycertinvalid.key",
			CAFile:   "../../client/testdata/myCA.cer",
		},
	}

	rt, err := MakeTransport(config)
	if err == nil {
		t.Errorf("Expected an error")
	}
	if rt != nil {
		t.Error("rt should be nil as we provided invalid cert file")
	}
}

func TestMakeTransportValid(t *testing.T) {
	config := &KubeletClientConfig{
		Port:        1234,
		EnableHttps: true,
		TLSClientConfig: restclient.TLSClientConfig{
			CertFile: "../../client/testdata/mycertvalid.cer",
			// TLS Configuration, only applies if EnableHttps is true.
			KeyFile: "../../client/testdata/mycertvalid.key",
			// TLS Configuration, only applies if EnableHttps is true.
			CAFile: "../../client/testdata/myCA.cer",
		},
	}

	rt, err := MakeTransport(config)
	if err != nil {
		t.Errorf("Not expecting an error #%v", err)
	}
	if rt == nil {
		t.Error("rt should not be nil")
	}
}
