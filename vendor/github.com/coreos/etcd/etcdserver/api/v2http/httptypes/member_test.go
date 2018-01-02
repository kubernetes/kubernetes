// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package httptypes

import (
	"encoding/json"
	"net/url"
	"reflect"
	"testing"

	"github.com/coreos/etcd/pkg/types"
)

func TestMemberUnmarshal(t *testing.T) {
	tests := []struct {
		body       []byte
		wantMember Member
		wantError  bool
	}{
		// no URLs, just check ID & Name
		{
			body:       []byte(`{"id": "c", "name": "dungarees"}`),
			wantMember: Member{ID: "c", Name: "dungarees", PeerURLs: nil, ClientURLs: nil},
		},

		// both client and peer URLs
		{
			body: []byte(`{"peerURLs": ["http://127.0.0.1:2379"], "clientURLs": ["http://127.0.0.1:2379"]}`),
			wantMember: Member{
				PeerURLs: []string{
					"http://127.0.0.1:2379",
				},
				ClientURLs: []string{
					"http://127.0.0.1:2379",
				},
			},
		},

		// multiple peer URLs
		{
			body: []byte(`{"peerURLs": ["http://127.0.0.1:2379", "https://example.com"]}`),
			wantMember: Member{
				PeerURLs: []string{
					"http://127.0.0.1:2379",
					"https://example.com",
				},
				ClientURLs: nil,
			},
		},

		// multiple client URLs
		{
			body: []byte(`{"clientURLs": ["http://127.0.0.1:2379", "https://example.com"]}`),
			wantMember: Member{
				PeerURLs: nil,
				ClientURLs: []string{
					"http://127.0.0.1:2379",
					"https://example.com",
				},
			},
		},

		// invalid JSON
		{
			body:      []byte(`{"peerU`),
			wantError: true,
		},
	}

	for i, tt := range tests {
		got := Member{}
		err := json.Unmarshal(tt.body, &got)
		if tt.wantError != (err != nil) {
			t.Errorf("#%d: want error %t, got %v", i, tt.wantError, err)
			continue
		}

		if !reflect.DeepEqual(tt.wantMember, got) {
			t.Errorf("#%d: incorrect output: want=%#v, got=%#v", i, tt.wantMember, got)
		}
	}
}

func TestMemberCreateRequestUnmarshal(t *testing.T) {
	body := []byte(`{"peerURLs": ["http://127.0.0.1:8081", "https://127.0.0.1:8080"]}`)
	want := MemberCreateRequest{
		PeerURLs: types.URLs([]url.URL{
			{Scheme: "http", Host: "127.0.0.1:8081"},
			{Scheme: "https", Host: "127.0.0.1:8080"},
		}),
	}

	var req MemberCreateRequest
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("Unmarshal returned unexpected err=%v", err)
	}

	if !reflect.DeepEqual(want, req) {
		t.Fatalf("Failed to unmarshal MemberCreateRequest: want=%#v, got=%#v", want, req)
	}
}

func TestMemberCreateRequestUnmarshalFail(t *testing.T) {
	tests := [][]byte{
		// invalid JSON
		[]byte(``),
		[]byte(`{`),

		// spot-check validation done in types.NewURLs
		[]byte(`{"peerURLs": "foo"}`),
		[]byte(`{"peerURLs": ["."]}`),
		[]byte(`{"peerURLs": []}`),
		[]byte(`{"peerURLs": ["http://127.0.0.1:2379/foo"]}`),
		[]byte(`{"peerURLs": ["http://127.0.0.1"]}`),
	}

	for i, tt := range tests {
		var req MemberCreateRequest
		if err := json.Unmarshal(tt, &req); err == nil {
			t.Errorf("#%d: expected err, got nil", i)
		}
	}
}
