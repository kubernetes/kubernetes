// Copyright 2015 The rkt Authors
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

package config

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
)

const (
	authHeader string = "Authorization"
)

type authV1JsonParser struct{}

type authV1 struct {
	Domains     []string        `json:"domains"`
	Type        string          `json:"type"`
	Credentials json.RawMessage `json:"credentials"`
}

type basicV1 struct {
	User     string `json:"user"`
	Password string `json:"password"`
}

type oauthV1 struct {
	Token string `json:"token"`
}

type dockerAuthV1JsonParser struct{}

type dockerAuthV1 struct {
	Registries  []string `json:"registries"`
	Credentials basicV1  `json:"credentials"`
}

func init() {
	addParser("auth", "v1", &authV1JsonParser{})
	addParser("dockerAuth", "v1", &dockerAuthV1JsonParser{})
	registerSubDir("auth.d", []string{"auth", "dockerAuth"})
}

type basicAuthHeaderer struct {
	user     string
	password string
}

func (h *basicAuthHeaderer) Header() http.Header {
	headers := make(http.Header)
	creds := []byte(fmt.Sprintf("%s:%s", h.user, h.password))
	encodedCreds := base64.StdEncoding.EncodeToString(creds)
	headers.Add(authHeader, "Basic "+encodedCreds)

	return headers
}

type oAuthBearerTokenHeaderer struct {
	token string
}

func (h *oAuthBearerTokenHeaderer) Header() http.Header {
	headers := make(http.Header)
	headers.Add(authHeader, "Bearer "+h.token)

	return headers
}

func (p *authV1JsonParser) parse(config *Config, raw []byte) error {
	var auth authV1
	if err := json.Unmarshal(raw, &auth); err != nil {
		return err
	}
	if len(auth.Domains) == 0 {
		return fmt.Errorf("no domains specified")
	}
	if len(auth.Type) == 0 {
		return fmt.Errorf("no auth type specified")
	}
	var (
		err      error
		headerer Headerer
	)
	switch auth.Type {
	case "basic":
		headerer, err = p.getBasicV1Headerer(auth.Credentials)
	case "oauth":
		headerer, err = p.getOAuthV1Headerer(auth.Credentials)
	default:
		err = fmt.Errorf("unknown auth type: %q", auth.Type)
	}
	if err != nil {
		return err
	}
	for _, domain := range auth.Domains {
		if _, ok := config.AuthPerHost[domain]; ok {
			return fmt.Errorf("auth for domain %q is already specified", domain)
		}
		config.AuthPerHost[domain] = headerer
	}
	return nil
}

func (p *authV1JsonParser) getBasicV1Headerer(raw json.RawMessage) (Headerer, error) {
	var basic basicV1
	if err := json.Unmarshal(raw, &basic); err != nil {
		return nil, err
	}
	if err := validateBasicV1(&basic); err != nil {
		return nil, err
	}
	return &basicAuthHeaderer{
		user:     basic.User,
		password: basic.Password,
	}, nil
}

func (p *authV1JsonParser) getOAuthV1Headerer(raw json.RawMessage) (Headerer, error) {
	var oauth oauthV1
	if err := json.Unmarshal(raw, &oauth); err != nil {
		return nil, err
	}
	if len(oauth.Token) == 0 {
		return nil, fmt.Errorf("no oauth bearer token specified")
	}
	return &oAuthBearerTokenHeaderer{
		token: oauth.Token,
	}, nil
}

func (p *dockerAuthV1JsonParser) parse(config *Config, raw []byte) error {
	var auth dockerAuthV1
	if err := json.Unmarshal(raw, &auth); err != nil {
		return err
	}
	if len(auth.Registries) == 0 {
		return fmt.Errorf("no registries specified")
	}
	if err := validateBasicV1(&auth.Credentials); err != nil {
		return err
	}
	basic := BasicCredentials{
		User:     auth.Credentials.User,
		Password: auth.Credentials.Password,
	}
	for _, registry := range auth.Registries {
		if _, ok := config.DockerCredentialsPerRegistry[registry]; ok {
			return fmt.Errorf("credentials for docker registry %q are already specified", registry)
		}
		config.DockerCredentialsPerRegistry[registry] = basic
	}
	return nil
}

func validateBasicV1(basic *basicV1) error {
	if basic == nil {
		return fmt.Errorf("no credentials")
	}
	if len(basic.User) == 0 {
		return fmt.Errorf("user not specified")
	}
	if len(basic.Password) == 0 {
		return fmt.Errorf("password not specified")
	}
	return nil
}
