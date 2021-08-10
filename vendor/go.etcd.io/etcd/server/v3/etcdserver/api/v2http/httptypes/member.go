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

// Package httptypes defines how etcd's HTTP API entities are serialized to and
// deserialized from JSON.
package httptypes

import (
	"encoding/json"

	"go.etcd.io/etcd/client/pkg/v3/types"
)

type Member struct {
	ID         string   `json:"id"`
	Name       string   `json:"name"`
	PeerURLs   []string `json:"peerURLs"`
	ClientURLs []string `json:"clientURLs"`
}

type MemberCreateRequest struct {
	PeerURLs types.URLs
}

type MemberUpdateRequest struct {
	MemberCreateRequest
}

func (m *MemberCreateRequest) UnmarshalJSON(data []byte) error {
	s := struct {
		PeerURLs []string `json:"peerURLs"`
	}{}

	err := json.Unmarshal(data, &s)
	if err != nil {
		return err
	}

	urls, err := types.NewURLs(s.PeerURLs)
	if err != nil {
		return err
	}

	m.PeerURLs = urls
	return nil
}

type MemberCollection []Member

func (c *MemberCollection) MarshalJSON() ([]byte, error) {
	d := struct {
		Members []Member `json:"members"`
	}{
		Members: []Member(*c),
	}

	return json.Marshal(d)
}
