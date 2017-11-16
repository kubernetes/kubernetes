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
	"strconv"
)

type UploadCustomCertificateParams struct {
	p map[string]interface{}
}

func (p *UploadCustomCertificateParams) toURLValues() url.Values {
	u := url.Values{}
	if p.p == nil {
		return u
	}
	if v, found := p.p["certificate"]; found {
		u.Set("certificate", v.(string))
	}
	if v, found := p.p["domainsuffix"]; found {
		u.Set("domainsuffix", v.(string))
	}
	if v, found := p.p["id"]; found {
		vv := strconv.Itoa(v.(int))
		u.Set("id", vv)
	}
	if v, found := p.p["name"]; found {
		u.Set("name", v.(string))
	}
	if v, found := p.p["privatekey"]; found {
		u.Set("privatekey", v.(string))
	}
	return u
}

func (p *UploadCustomCertificateParams) SetCertificate(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["certificate"] = v
	return
}

func (p *UploadCustomCertificateParams) SetDomainsuffix(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["domainsuffix"] = v
	return
}

func (p *UploadCustomCertificateParams) SetId(v int) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["id"] = v
	return
}

func (p *UploadCustomCertificateParams) SetName(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["name"] = v
	return
}

func (p *UploadCustomCertificateParams) SetPrivatekey(v string) {
	if p.p == nil {
		p.p = make(map[string]interface{})
	}
	p.p["privatekey"] = v
	return
}

// You should always use this function to get a new UploadCustomCertificateParams instance,
// as then you are sure you have configured all required params
func (s *CertificateService) NewUploadCustomCertificateParams(certificate string, domainsuffix string) *UploadCustomCertificateParams {
	p := &UploadCustomCertificateParams{}
	p.p = make(map[string]interface{})
	p.p["certificate"] = certificate
	p.p["domainsuffix"] = domainsuffix
	return p
}

// Uploads a custom certificate for the console proxy VMs to use for SSL. Can be used to upload a single certificate signed by a known CA. Can also be used, through multiple calls, to upload a chain of certificates from CA to the custom certificate itself.
func (s *CertificateService) UploadCustomCertificate(p *UploadCustomCertificateParams) (*UploadCustomCertificateResponse, error) {
	resp, err := s.cs.newRequest("uploadCustomCertificate", p.toURLValues())
	if err != nil {
		return nil, err
	}

	var r UploadCustomCertificateResponse
	if err := json.Unmarshal(resp, &r); err != nil {
		return nil, err
	}

	// If we have a async client, we need to wait for the async result
	if s.cs.async {
		b, err := s.cs.GetAsyncJobResult(r.JobID, s.cs.timeout)
		if err != nil {
			if err == AsyncTimeoutErr {
				return &r, err
			}
			return nil, err
		}

		b, err = getRawValue(b)
		if err != nil {
			return nil, err
		}

		if err := json.Unmarshal(b, &r); err != nil {
			return nil, err
		}
	}
	return &r, nil
}

type UploadCustomCertificateResponse struct {
	JobID   string `json:"jobid,omitempty"`
	Message string `json:"message,omitempty"`
}
