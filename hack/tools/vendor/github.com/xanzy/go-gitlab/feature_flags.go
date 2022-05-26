//
// Copyright 2021, Sander van Harmelen
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
//

package gitlab

import (
	"fmt"
	"net/url"
)

// FeaturesService handles the communication with the application FeaturesService
// related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/features.html
type FeaturesService struct {
	client *Client
}

// Feature represents a GitLab feature flag.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/features.html
type Feature struct {
	Name  string `json:"name"`
	State string `json:"state"`
	Gates []Gate
}

// Gate represents a gate of a GitLab feature flag.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/features.html
type Gate struct {
	Key   string      `json:"key"`
	Value interface{} `json:"value"`
}

func (f Feature) String() string {
	return Stringify(f)
}

// ListFeatures gets a list of feature flags
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/features.html#list-all-features
func (s *FeaturesService) ListFeatures(options ...RequestOptionFunc) ([]*Feature, *Response, error) {
	req, err := s.client.NewRequest("GET", "features", nil, options)
	if err != nil {
		return nil, nil, err
	}

	var f []*Feature
	resp, err := s.client.Do(req, &f)
	if err != nil {
		return nil, resp, err
	}
	return f, resp, err
}

// SetFeatureFlag sets or creates a feature flag gate
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/features.html#set-or-create-a-feature
func (s *FeaturesService) SetFeatureFlag(name string, value interface{}, options ...RequestOptionFunc) (*Feature, *Response, error) {
	u := fmt.Sprintf("features/%s", url.PathEscape(name))

	opt := struct {
		Value interface{} `url:"value" json:"value"`
	}{
		value,
	}

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	f := &Feature{}
	resp, err := s.client.Do(req, f)
	if err != nil {
		return nil, resp, err
	}
	return f, resp, err
}
