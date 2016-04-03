// Copyright 2015 CoreOS, Inc.
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

package client

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"reflect"
	"testing"

	"golang.org/x/net/context"

	"github.com/coreos/etcd/pkg/types"
)

func TestMembersAPIActionList(t *testing.T) {
	ep := url.URL{Scheme: "http", Host: "example.com"}
	act := &membersAPIActionList{}

	wantURL := &url.URL{
		Scheme: "http",
		Host:   "example.com",
		Path:   "/v2/members",
	}

	got := *act.HTTPRequest(ep)
	err := assertRequest(got, "GET", wantURL, http.Header{}, nil)
	if err != nil {
		t.Error(err.Error())
	}
}

func TestMembersAPIActionAdd(t *testing.T) {
	ep := url.URL{Scheme: "http", Host: "example.com"}
	act := &membersAPIActionAdd{
		peerURLs: types.URLs([]url.URL{
			{Scheme: "https", Host: "127.0.0.1:8081"},
			{Scheme: "http", Host: "127.0.0.1:8080"},
		}),
	}

	wantURL := &url.URL{
		Scheme: "http",
		Host:   "example.com",
		Path:   "/v2/members",
	}
	wantHeader := http.Header{
		"Content-Type": []string{"application/json"},
	}
	wantBody := []byte(`{"peerURLs":["https://127.0.0.1:8081","http://127.0.0.1:8080"]}`)

	got := *act.HTTPRequest(ep)
	err := assertRequest(got, "POST", wantURL, wantHeader, wantBody)
	if err != nil {
		t.Error(err.Error())
	}
}

func TestMembersAPIActionUpdate(t *testing.T) {
	ep := url.URL{Scheme: "http", Host: "example.com"}
	act := &membersAPIActionUpdate{
		memberID: "0xabcd",
		peerURLs: types.URLs([]url.URL{
			{Scheme: "https", Host: "127.0.0.1:8081"},
			{Scheme: "http", Host: "127.0.0.1:8080"},
		}),
	}

	wantURL := &url.URL{
		Scheme: "http",
		Host:   "example.com",
		Path:   "/v2/members/0xabcd",
	}
	wantHeader := http.Header{
		"Content-Type": []string{"application/json"},
	}
	wantBody := []byte(`{"peerURLs":["https://127.0.0.1:8081","http://127.0.0.1:8080"]}`)

	got := *act.HTTPRequest(ep)
	err := assertRequest(got, "PUT", wantURL, wantHeader, wantBody)
	if err != nil {
		t.Error(err.Error())
	}
}

func TestMembersAPIActionRemove(t *testing.T) {
	ep := url.URL{Scheme: "http", Host: "example.com"}
	act := &membersAPIActionRemove{memberID: "XXX"}

	wantURL := &url.URL{
		Scheme: "http",
		Host:   "example.com",
		Path:   "/v2/members/XXX",
	}

	got := *act.HTTPRequest(ep)
	err := assertRequest(got, "DELETE", wantURL, http.Header{}, nil)
	if err != nil {
		t.Error(err.Error())
	}
}

func TestMembersAPIActionLeader(t *testing.T) {
	ep := url.URL{Scheme: "http", Host: "example.com"}
	act := &membersAPIActionLeader{}

	wantURL := &url.URL{
		Scheme: "http",
		Host:   "example.com",
		Path:   "/v2/members/leader",
	}

	got := *act.HTTPRequest(ep)
	err := assertRequest(got, "GET", wantURL, http.Header{}, nil)
	if err != nil {
		t.Error(err.Error())
	}
}

func TestAssertStatusCode(t *testing.T) {
	if err := assertStatusCode(404, 400); err == nil {
		t.Errorf("assertStatusCode failed to detect conflict in 400 vs 404")
	}

	if err := assertStatusCode(404, 400, 404); err != nil {
		t.Errorf("assertStatusCode found conflict in (404,400) vs 400: %v", err)
	}
}

func TestV2MembersURL(t *testing.T) {
	got := v2MembersURL(url.URL{
		Scheme: "http",
		Host:   "foo.example.com:4002",
		Path:   "/pants",
	})
	want := &url.URL{
		Scheme: "http",
		Host:   "foo.example.com:4002",
		Path:   "/pants/v2/members",
	}

	if !reflect.DeepEqual(want, got) {
		t.Fatalf("v2MembersURL got %#v, want %#v", got, want)
	}
}

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

func TestMemberCollectionUnmarshalFail(t *testing.T) {
	mc := &memberCollection{}
	if err := mc.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Errorf("got nil error")
	}
}

func TestMemberCollectionUnmarshal(t *testing.T) {
	tests := []struct {
		body []byte
		want memberCollection
	}{
		{
			body: []byte(`{}`),
			want: memberCollection([]Member{}),
		},
		{
			body: []byte(`{"members":[]}`),
			want: memberCollection([]Member{}),
		},
		{
			body: []byte(`{"members":[{"id":"2745e2525fce8fe","peerURLs":["http://127.0.0.1:7003"],"name":"node3","clientURLs":["http://127.0.0.1:4003"]},{"id":"42134f434382925","peerURLs":["http://127.0.0.1:2380","http://127.0.0.1:7001"],"name":"node1","clientURLs":["http://127.0.0.1:2379","http://127.0.0.1:4001"]},{"id":"94088180e21eb87b","peerURLs":["http://127.0.0.1:7002"],"name":"node2","clientURLs":["http://127.0.0.1:4002"]}]}`),
			want: memberCollection(
				[]Member{
					{
						ID:   "2745e2525fce8fe",
						Name: "node3",
						PeerURLs: []string{
							"http://127.0.0.1:7003",
						},
						ClientURLs: []string{
							"http://127.0.0.1:4003",
						},
					},
					{
						ID:   "42134f434382925",
						Name: "node1",
						PeerURLs: []string{
							"http://127.0.0.1:2380",
							"http://127.0.0.1:7001",
						},
						ClientURLs: []string{
							"http://127.0.0.1:2379",
							"http://127.0.0.1:4001",
						},
					},
					{
						ID:   "94088180e21eb87b",
						Name: "node2",
						PeerURLs: []string{
							"http://127.0.0.1:7002",
						},
						ClientURLs: []string{
							"http://127.0.0.1:4002",
						},
					},
				},
			),
		},
	}

	for i, tt := range tests {
		var got memberCollection
		err := json.Unmarshal(tt.body, &got)
		if err != nil {
			t.Errorf("#%d: unexpected error: %v", i, err)
			continue
		}

		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("#%d: incorrect output: want=%#v, got=%#v", i, tt.want, got)
		}
	}
}

func TestMemberCreateRequestMarshal(t *testing.T) {
	req := memberCreateOrUpdateRequest{
		PeerURLs: types.URLs([]url.URL{
			{Scheme: "http", Host: "127.0.0.1:8081"},
			{Scheme: "https", Host: "127.0.0.1:8080"},
		}),
	}
	want := []byte(`{"peerURLs":["http://127.0.0.1:8081","https://127.0.0.1:8080"]}`)

	got, err := json.Marshal(&req)
	if err != nil {
		t.Fatalf("Marshal returned unexpected err=%v", err)
	}

	if !reflect.DeepEqual(want, got) {
		t.Fatalf("Failed to marshal memberCreateRequest: want=%s, got=%s", want, got)
	}
}

func TestHTTPMembersAPIAddSuccess(t *testing.T) {
	wantAction := &membersAPIActionAdd{
		peerURLs: types.URLs([]url.URL{
			{Scheme: "http", Host: "127.0.0.1:7002"},
		}),
	}

	mAPI := &httpMembersAPI{
		client: &actionAssertingHTTPClient{
			t:   t,
			act: wantAction,
			resp: http.Response{
				StatusCode: http.StatusCreated,
			},
			body: []byte(`{"id":"94088180e21eb87b","peerURLs":["http://127.0.0.1:7002"]}`),
		},
	}

	wantResponseMember := &Member{
		ID:       "94088180e21eb87b",
		PeerURLs: []string{"http://127.0.0.1:7002"},
	}

	m, err := mAPI.Add(context.Background(), "http://127.0.0.1:7002")
	if err != nil {
		t.Errorf("got non-nil err: %#v", err)
	}
	if !reflect.DeepEqual(wantResponseMember, m) {
		t.Errorf("incorrect Member: want=%#v got=%#v", wantResponseMember, m)
	}
}

func TestHTTPMembersAPIAddError(t *testing.T) {
	okPeer := "http://example.com:2379"
	tests := []struct {
		peerURL string
		client  httpClient

		// if wantErr == nil, assert that the returned error is non-nil
		// if wantErr != nil, assert that the returned error matches
		wantErr error
	}{
		// malformed peer URL
		{
			peerURL: ":",
		},

		// generic httpClient failure
		{
			peerURL: okPeer,
			client:  &staticHTTPClient{err: errors.New("fail!")},
		},

		// unrecognized HTTP status code
		{
			peerURL: okPeer,
			client: &staticHTTPClient{
				resp: http.Response{StatusCode: http.StatusTeapot},
			},
		},

		// unmarshal body into membersError on StatusConflict
		{
			peerURL: okPeer,
			client: &staticHTTPClient{
				resp: http.Response{
					StatusCode: http.StatusConflict,
				},
				body: []byte(`{"message":"fail!"}`),
			},
			wantErr: membersError{Message: "fail!"},
		},

		// fail to unmarshal body on StatusConflict
		{
			peerURL: okPeer,
			client: &staticHTTPClient{
				resp: http.Response{
					StatusCode: http.StatusConflict,
				},
				body: []byte(`{"`),
			},
		},

		// fail to unmarshal body on StatusCreated
		{
			peerURL: okPeer,
			client: &staticHTTPClient{
				resp: http.Response{
					StatusCode: http.StatusCreated,
				},
				body: []byte(`{"id":"XX`),
			},
		},
	}

	for i, tt := range tests {
		mAPI := &httpMembersAPI{client: tt.client}
		m, err := mAPI.Add(context.Background(), tt.peerURL)
		if err == nil {
			t.Errorf("#%d: got nil err", i)
		}
		if tt.wantErr != nil && !reflect.DeepEqual(tt.wantErr, err) {
			t.Errorf("#%d: incorrect error: want=%#v got=%#v", i, tt.wantErr, err)
		}
		if m != nil {
			t.Errorf("#%d: got non-nil Member", i)
		}
	}
}

func TestHTTPMembersAPIRemoveSuccess(t *testing.T) {
	wantAction := &membersAPIActionRemove{
		memberID: "94088180e21eb87b",
	}

	mAPI := &httpMembersAPI{
		client: &actionAssertingHTTPClient{
			t:   t,
			act: wantAction,
			resp: http.Response{
				StatusCode: http.StatusNoContent,
			},
		},
	}

	if err := mAPI.Remove(context.Background(), "94088180e21eb87b"); err != nil {
		t.Errorf("got non-nil err: %#v", err)
	}
}

func TestHTTPMembersAPIRemoveFail(t *testing.T) {
	tests := []httpClient{
		// generic error
		&staticHTTPClient{
			err: errors.New("fail!"),
		},

		// unexpected HTTP status code
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusInternalServerError,
			},
		},
	}

	for i, tt := range tests {
		mAPI := &httpMembersAPI{client: tt}
		if err := mAPI.Remove(context.Background(), "94088180e21eb87b"); err == nil {
			t.Errorf("#%d: got nil err", i)
		}
	}
}

func TestHTTPMembersAPIListSuccess(t *testing.T) {
	wantAction := &membersAPIActionList{}
	mAPI := &httpMembersAPI{
		client: &actionAssertingHTTPClient{
			t:   t,
			act: wantAction,
			resp: http.Response{
				StatusCode: http.StatusOK,
			},
			body: []byte(`{"members":[{"id":"94088180e21eb87b","name":"node2","peerURLs":["http://127.0.0.1:7002"],"clientURLs":["http://127.0.0.1:4002"]}]}`),
		},
	}

	wantResponseMembers := []Member{
		{
			ID:         "94088180e21eb87b",
			Name:       "node2",
			PeerURLs:   []string{"http://127.0.0.1:7002"},
			ClientURLs: []string{"http://127.0.0.1:4002"},
		},
	}

	m, err := mAPI.List(context.Background())
	if err != nil {
		t.Errorf("got non-nil err: %#v", err)
	}
	if !reflect.DeepEqual(wantResponseMembers, m) {
		t.Errorf("incorrect Members: want=%#v got=%#v", wantResponseMembers, m)
	}
}

func TestHTTPMembersAPIListError(t *testing.T) {
	tests := []httpClient{
		// generic httpClient failure
		&staticHTTPClient{err: errors.New("fail!")},

		// unrecognized HTTP status code
		&staticHTTPClient{
			resp: http.Response{StatusCode: http.StatusTeapot},
		},

		// fail to unmarshal body on StatusOK
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusOK,
			},
			body: []byte(`[{"id":"XX`),
		},
	}

	for i, tt := range tests {
		mAPI := &httpMembersAPI{client: tt}
		ms, err := mAPI.List(context.Background())
		if err == nil {
			t.Errorf("#%d: got nil err", i)
		}
		if ms != nil {
			t.Errorf("#%d: got non-nil Member slice", i)
		}
	}
}

func TestHTTPMembersAPILeaderSuccess(t *testing.T) {
	wantAction := &membersAPIActionLeader{}
	mAPI := &httpMembersAPI{
		client: &actionAssertingHTTPClient{
			t:   t,
			act: wantAction,
			resp: http.Response{
				StatusCode: http.StatusOK,
			},
			body: []byte(`{"id":"94088180e21eb87b","name":"node2","peerURLs":["http://127.0.0.1:7002"],"clientURLs":["http://127.0.0.1:4002"]}`),
		},
	}

	wantResponseMember := &Member{
		ID:         "94088180e21eb87b",
		Name:       "node2",
		PeerURLs:   []string{"http://127.0.0.1:7002"},
		ClientURLs: []string{"http://127.0.0.1:4002"},
	}

	m, err := mAPI.Leader(context.Background())
	if err != nil {
		t.Errorf("err = %v, want %v", err, nil)
	}
	if !reflect.DeepEqual(wantResponseMember, m) {
		t.Errorf("incorrect member: member = %v, want %v", wantResponseMember, m)
	}
}

func TestHTTPMembersAPILeaderError(t *testing.T) {
	tests := []httpClient{
		// generic httpClient failure
		&staticHTTPClient{err: errors.New("fail!")},

		// unrecognized HTTP status code
		&staticHTTPClient{
			resp: http.Response{StatusCode: http.StatusTeapot},
		},

		// fail to unmarshal body on StatusOK
		&staticHTTPClient{
			resp: http.Response{
				StatusCode: http.StatusOK,
			},
			body: []byte(`[{"id":"XX`),
		},
	}

	for i, tt := range tests {
		mAPI := &httpMembersAPI{client: tt}
		m, err := mAPI.Leader(context.Background())
		if err == nil {
			t.Errorf("#%d: err = nil, want not nil", i)
		}
		if m != nil {
			t.Errorf("member slice = %v, want nil", m)
		}
	}
}
