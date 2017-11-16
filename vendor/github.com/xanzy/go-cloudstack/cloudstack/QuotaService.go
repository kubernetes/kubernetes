//
// Copyright 2016, Sander van Harmelen
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

package cloudstack

import (
	"encoding/json"
	"net/url"
)

type QuotaIsEnabledParams struct {
	p map[string]interface{}
}

func (p *QuotaIsEnabledParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	return u
}

// You should always use this function to get a new QuotaIsEnabledParams instance,
// as then you are sure you have configured all required params
func (s *QuotaService) NewQuotaIsEnabledParams() *QuotaIsEnabledParams {
	p := &QuotaIsEnabledParams{}
	p.p = make(map[string]interface{})
	return p
}

// Return true if the plugin is enabled
func (s *QuotaService) QuotaIsEnabled(p *QuotaIsEnabledParams) (*QuotaIsEnabledResponse, error) {
	resp, err := s.cs.newRequest("quotaIsEnabled", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r QuotaIsEnabledResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type QuotaIsEnabledResponse struct {
	Isenabled bool `json:"isenabled,omitempty"`
}
