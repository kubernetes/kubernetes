//
// Copyright 2021, Patrick Webster
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

import "time"

// LicenseService handles communication with the license
// related methods of the GitLab API.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/license.html
type LicenseService struct {
	client *Client
}

// License represents a GitLab license.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/license.html
type License struct {
	ID               int        `json:"id"`
	Plan             string     `json:"plan"`
	CreatedAt        *time.Time `json:"created_at"`
	StartsAt         *ISOTime   `json:"starts_at"`
	ExpiresAt        *ISOTime   `json:"expires_at"`
	HistoricalMax    int        `json:"historical_max"`
	MaximumUserCount int        `json:"maximum_user_count"`
	Expired          bool       `json:"expired"`
	Overage          int        `json:"overage"`
	UserLimit        int        `json:"user_limit"`
	ActiveUsers      int        `json:"active_users"`
	Licensee         struct {
		Name    string `json:"Name"`
		Company string `json:"Company"`
		Email   string `json:"Email"`
	} `json:"licensee"`
	// Add on codes that may occur in legacy licenses that don't have a plan yet.
	// https://gitlab.com/gitlab-org/gitlab/-/blob/master/ee/app/models/license.rb
	AddOns struct {
		GitLabAuditorUser int `json:"GitLab_Auditor_User"`
		GitLabDeployBoard int `json:"GitLab_DeployBoard"`
		GitLabFileLocks   int `json:"GitLab_FileLocks"`
		GitLabGeo         int `json:"GitLab_Geo"`
		GitLabServiceDesk int `json:"GitLab_ServiceDesk"`
	} `json:"add_ons"`
}

func (l License) String() string {
	return Stringify(l)
}

// GetLicense retrieves information about the current license.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/license.html#retrieve-information-about-the-current-license
func (s *LicenseService) GetLicense() (*License, *Response, error) {
	req, err := s.client.NewRequest("GET", "license", nil, nil)
	if err != nil {
		return nil, nil, err
	}

	l := new(License)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, err
}

// AddLicenseOptions represents the available AddLicense() options.
//
// https://docs.gitlab.com/ee/api/license.html#add-a-new-license
type AddLicenseOptions struct {
	License *string `url:"license" json:"license"`
}

// AddLicense adds a new license.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/license.html#add-a-new-license
func (s *LicenseService) AddLicense(opt *AddLicenseOptions, options ...RequestOptionFunc) (*License, *Response, error) {
	req, err := s.client.NewRequest("POST", "license", opt, options)
	if err != nil {
		return nil, nil, err
	}

	l := new(License)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, err
}
