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

type GetCloudIdentifierParams struct {
	p map[string]interface{}
}

func (p *GetCloudIdentifierParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["userid"]; found {
		u.Set("userid", v.(string))
	}
	return u
}

func (p *GetCloudIdentifierParams) SetUserid(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["userid"] = v
	return
}

// You should always use this function to get a new GetCloudIdentifierParams instance,
// as then you are sure you have configured all required params
func (s *CloudIdentifierService) NewGetCloudIdentifierParams(userid string) *GetCloudIdentifierParams {
	p := &GetCloudIdentifierParams{}
	p.p = make(map[string]interface{})
	p.p["userid"] = userid
	return p
}

// Retrieves a cloud identifier.
func (s *CloudIdentifierService) GetCloudIdentifier(p *GetCloudIdentifierParams) (*GetCloudIdentifierResponse, error) {
	resp, err := s.cs.newRequest("getCloudIdentifier", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r GetCloudIdentifierResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

type GetCloudIdentifierResponse struct {
	Cloudidentifier string `json:"cloudidentifier,omitempty"`
	Signature       string `json:"signature,omitempty"`
	Userid          string `json:"userid,omitempty"`
}
