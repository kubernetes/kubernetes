//
// Copyright 2021, Andrea Funto'
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

// VersionService handles communication with the GitLab server instance to
// retrieve its version information via the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/version.md
type VersionService struct {
	client *Client
}

// Version represents a GitLab instance version.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/version.md
type Version struct {
	Version  string `json:"version"`
	Revision string `json:"revision"`
}

func (s Version) String() string {
	return Stringify(s)
}

// GetVersion gets a GitLab server instance version; it is only available to
// authenticated users.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/version.md
func (s *VersionService) GetVersion() (*Version, *Response, error) {
	req, err := s.client.NewRequest("GET", "version", nil, nil)
	if err != nil {
		return nil, nil, err
	}

	v := new(Version)
	resp, err := s.client.Do(req, v)
	if err != nil {
		return nil, resp, err
	}

	return v, resp, err
}
