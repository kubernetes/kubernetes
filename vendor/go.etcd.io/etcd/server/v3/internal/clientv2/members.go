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

package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"path"

	"go.etcd.io/etcd/client/pkg/v3/types"
)

var (
	defaultV2MembersPrefix = "/v2/members"
	defaultLeaderSuffix    = "/leader"
)

type Member struct {
	// ID is the unique identifier of this Member.
	ID string `json:"id"`

	// Name is a human-readable, non-unique identifier of this Member.
	Name string `json:"name"`

	// PeerURLs represents the HTTP(S) endpoints this Member uses to
	// participate in etcd's consensus protocol.
	PeerURLs []string `json:"peerURLs"`

	// ClientURLs represents the HTTP(S) endpoints on which this Member
	// serves its client-facing APIs.
	ClientURLs []string `json:"clientURLs"`
}

type memberCollection []Member

func (c *memberCollection) UnmarshalJSON(data []byte) error {
	d := struct {
		Members []Member
	}{}

	if err := json.Unmarshal(data, &d); err != nil {
		return err
	}

	if d.Members == nil {
		*c = make([]Member, 0)
		return nil
	}

	*c = d.Members
	return nil
}

type memberCreateOrUpdateRequest struct {
	PeerURLs types.URLs
}

func (m *memberCreateOrUpdateRequest) MarshalJSON() ([]byte, error) {
	s := struct {
		PeerURLs []string `json:"peerURLs"`
	}{
		PeerURLs: make([]string, len(m.PeerURLs)),
	}

	for i, u := range m.PeerURLs {
		s.PeerURLs[i] = u.String()
	}

	return json.Marshal(&s)
}

// NewMembersAPI constructs a new MembersAPI that uses HTTP to
// interact with etcd's membership API.
func NewMembersAPI(c Client) MembersAPI {
	return &httpMembersAPI{
		client: c,
	}
}

type MembersAPI interface {
	// List enumerates the current cluster membership.
	List(ctx context.Context) ([]Member, error)

	// Add instructs etcd to accept a new Member into the cluster.
	Add(ctx context.Context, peerURL string) (*Member, error)

	// Remove demotes an existing Member out of the cluster.
	Remove(ctx context.Context, mID string) error

	// Update instructs etcd to update an existing Member in the cluster.
	Update(ctx context.Context, mID string, peerURLs []string) error

	// Leader gets current leader of the cluster
	Leader(ctx context.Context) (*Member, error)
}

type httpMembersAPI struct {
	client httpClient
}

func (m *httpMembersAPI) List(ctx context.Context) ([]Member, error) {
	req := &membersAPIActionList{}
	resp, body, err := m.client.Do(ctx, req)
	if err != nil {
		return nil, err
	}

	if err := assertStatusCode(resp.StatusCode, http.StatusOK); err != nil {
		return nil, err
	}

	var mCollection memberCollection
	if err := json.Unmarshal(body, &mCollection); err != nil {
		return nil, err
	}

	return mCollection, nil
}

func (m *httpMembersAPI) Add(ctx context.Context, peerURL string) (*Member, error) {
	urls, err := types.NewURLs([]string{peerURL})
	if err != nil {
		return nil, err
	}

	req := &membersAPIActionAdd{peerURLs: urls}
	resp, body, err := m.client.Do(ctx, req)
	if err != nil {
		return nil, err
	}

	if err := assertStatusCode(resp.StatusCode, http.StatusCreated, http.StatusConflict); err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusCreated {
		var merr membersError
		if err := json.Unmarshal(body, &merr); err != nil {
			return nil, err
		}
		return nil, merr
	}

	var memb Member
	if err := json.Unmarshal(body, &memb); err != nil {
		return nil, err
	}

	return &memb, nil
}

func (m *httpMembersAPI) Update(ctx context.Context, memberID string, peerURLs []string) error {
	urls, err := types.NewURLs(peerURLs)
	if err != nil {
		return err
	}

	req := &membersAPIActionUpdate{peerURLs: urls, memberID: memberID}
	resp, body, err := m.client.Do(ctx, req)
	if err != nil {
		return err
	}

	if err := assertStatusCode(resp.StatusCode, http.StatusNoContent, http.StatusNotFound, http.StatusConflict); err != nil {
		return err
	}

	if resp.StatusCode != http.StatusNoContent {
		var merr membersError
		if err := json.Unmarshal(body, &merr); err != nil {
			return err
		}
		return merr
	}

	return nil
}

func (m *httpMembersAPI) Remove(ctx context.Context, memberID string) error {
	req := &membersAPIActionRemove{memberID: memberID}
	resp, _, err := m.client.Do(ctx, req)
	if err != nil {
		return err
	}

	return assertStatusCode(resp.StatusCode, http.StatusNoContent, http.StatusGone)
}

func (m *httpMembersAPI) Leader(ctx context.Context) (*Member, error) {
	req := &membersAPIActionLeader{}
	resp, body, err := m.client.Do(ctx, req)
	if err != nil {
		return nil, err
	}

	if err := assertStatusCode(resp.StatusCode, http.StatusOK); err != nil {
		return nil, err
	}

	var leader Member
	if err := json.Unmarshal(body, &leader); err != nil {
		return nil, err
	}

	return &leader, nil
}

type membersAPIActionList struct{}

func (l *membersAPIActionList) HTTPRequest(ep url.URL) *http.Request {
	u := v2MembersURL(ep)
	req, _ := http.NewRequest(http.MethodGet, u.String(), nil)
	return req
}

type membersAPIActionRemove struct {
	memberID string
}

func (d *membersAPIActionRemove) HTTPRequest(ep url.URL) *http.Request {
	u := v2MembersURL(ep)
	u.Path = path.Join(u.Path, d.memberID)
	req, _ := http.NewRequest(http.MethodDelete, u.String(), nil)
	return req
}

type membersAPIActionAdd struct {
	peerURLs types.URLs
}

func (a *membersAPIActionAdd) HTTPRequest(ep url.URL) *http.Request {
	u := v2MembersURL(ep)
	m := memberCreateOrUpdateRequest{PeerURLs: a.peerURLs}
	b, _ := json.Marshal(&m)
	req, _ := http.NewRequest(http.MethodPost, u.String(), bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	return req
}

type membersAPIActionUpdate struct {
	memberID string
	peerURLs types.URLs
}

func (a *membersAPIActionUpdate) HTTPRequest(ep url.URL) *http.Request {
	u := v2MembersURL(ep)
	m := memberCreateOrUpdateRequest{PeerURLs: a.peerURLs}
	u.Path = path.Join(u.Path, a.memberID)
	b, _ := json.Marshal(&m)
	req, _ := http.NewRequest(http.MethodPut, u.String(), bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func assertStatusCode(got int, want ...int) (err error) {
	for _, w := range want {
		if w == got {
			return nil
		}
	}
	return fmt.Errorf("unexpected status code %d", got)
}

type membersAPIActionLeader struct{}

func (l *membersAPIActionLeader) HTTPRequest(ep url.URL) *http.Request {
	u := v2MembersURL(ep)
	u.Path = path.Join(u.Path, defaultLeaderSuffix)
	req, _ := http.NewRequest(http.MethodGet, u.String(), nil)
	return req
}

// v2MembersURL add the necessary path to the provided endpoint
// to route requests to the default v2 members API.
func v2MembersURL(ep url.URL) *url.URL {
	ep.Path = path.Join(ep.Path, defaultV2MembersPrefix)
	return &ep
}

type membersError struct {
	Message string `json:"message"`
	Code    int    `json:"-"`
}

func (e membersError) Error() string {
	return e.Message
}
