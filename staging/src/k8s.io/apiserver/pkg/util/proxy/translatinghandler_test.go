/*
Copyright 2023 The Kubernetes Authors.

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

package proxy

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/httpstream/wsstream"
)

// fakeHandler implements http.Handler interface
type fakeHandler struct {
	served bool
}

// ServeHTTP stores the fact that this fake handler was called.
func (fh *fakeHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	fh.served = true
}

func TestTranslatingHandler(t *testing.T) {
	tests := map[string]struct {
		upgrade          string
		version          string
		expectTranslator bool
	}{
		"websocket/v5 upgrade, serves translator": {
			upgrade:          "websocket",
			version:          "v5.channel.k8s.io",
			expectTranslator: true,
		},
		"websocket/v5 upgrade with multiple other versions, serves translator": {
			upgrade:          "websocket",
			version:          "v5.channel.k8s.io, v4.channel.k8s.io, v3.channel.k8s.io",
			expectTranslator: true,
		},
		"websocket/v5 upgrade with multiple other versions out of order, serves translator": {
			upgrade:          "websocket",
			version:          "v4.channel.k8s.io, v3.channel.k8s.io, v5.channel.k8s.io",
			expectTranslator: true,
		},
		"no upgrade, serves delegate": {
			upgrade:          "",
			version:          "",
			expectTranslator: false,
		},
		"no upgrade with v5, serves delegate": {
			upgrade:          "",
			version:          "v5.channel.k8s.io",
			expectTranslator: false,
		},
		"websocket/v5 wrong case upgrade, serves delegage": {
			upgrade:          "websocket",
			version:          "v5.CHANNEL.k8s.io",
			expectTranslator: false,
		},
		"spdy/v5 upgrade, serves delegate": {
			upgrade:          "spdy",
			version:          "v5.channel.k8s.io",
			expectTranslator: false,
		},
		"spdy/v4 upgrade, serves delegate": {
			upgrade:          "spdy",
			version:          "v4.channel.k8s.io",
			expectTranslator: false,
		},
		"websocket/v4 upgrade, serves delegate": {
			upgrade:          "websocket",
			version:          "v4.channel.k8s.io",
			expectTranslator: false,
		},
		"websocket without version upgrade, serves delegate": {
			upgrade:          "websocket",
			version:          "",
			expectTranslator: false,
		},
	}
	for name, test := range tests {
		req, err := http.NewRequest("GET", "http://www.example.com/", nil)
		require.NoError(t, err)
		if test.upgrade != "" {
			req.Header.Add("Connection", "Upgrade")
			req.Header.Add("Upgrade", test.upgrade)
		}
		if len(test.version) > 0 {
			req.Header.Add(wsstream.WebSocketProtocolHeader, test.version)
		}
		delegate := fakeHandler{}
		translator := fakeHandler{}
		translatingHandler := NewTranslatingHandler(&delegate, &translator,
			wsstream.IsWebSocketRequestWithStreamCloseProtocol)
		translatingHandler.ServeHTTP(nil, req)
		if !delegate.served && !translator.served {
			t.Errorf("unexpected neither translator nor delegate served")
			continue
		}
		if test.expectTranslator {
			if !translator.served {
				t.Errorf("%s: expected translator served, got delegate served", name)
			}
		} else if !delegate.served {
			t.Errorf("%s: expected delegate served, got translator served", name)
		}
	}
}
