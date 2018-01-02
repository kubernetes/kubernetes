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
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/signer/v4"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	authHeader       string = "Authorization"
	defaultAWSRegion string = "us-east-1"
	awsS3Service     string = "s3"
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

type awsV1 struct {
	AccessKeyID     string `json:"accessKeyID"`
	SecretAccessKey string `json:"secretAccessKey"`
	Region          string `json:"awsRegion"`
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
	auth basicV1
}

func (h *basicAuthHeaderer) GetHeader() http.Header {
	headers := make(http.Header)
	creds := []byte(fmt.Sprintf("%s:%s", h.auth.User, h.auth.Password))
	encodedCreds := base64.StdEncoding.EncodeToString(creds)
	headers.Add(authHeader, "Basic "+encodedCreds)

	return headers
}

func (h *basicAuthHeaderer) SignRequest(r *http.Request) *http.Request {
	r.Header.Set(authHeader, h.GetHeader().Get(authHeader))

	return r
}

type oAuthBearerTokenHeaderer struct {
	auth oauthV1
}

func (h *oAuthBearerTokenHeaderer) GetHeader() http.Header {
	headers := make(http.Header)
	headers.Add(authHeader, "Bearer "+h.auth.Token)

	return headers
}

func (h *oAuthBearerTokenHeaderer) SignRequest(r *http.Request) *http.Request {
	r.Header.Set(authHeader, h.GetHeader().Get(authHeader))

	return r
}

type awsAuthHeaderer struct {
	accessKeyID     string
	secretAccessKey string
	region          string
}

func (h *awsAuthHeaderer) GetHeader() http.Header {
	return make(http.Header)
}

func (h *awsAuthHeaderer) SignRequest(r *http.Request) *http.Request {
	region := h.region

	var body io.ReadSeeker
	if r.Body != nil {
		body = r.Body.(io.ReadSeeker)
	}

	if len(region) == 0 {
		region = guessAWSRegion(r.URL.Host)
	}
	v4.Sign(&request.Request{
		ClientInfo: metadata.ClientInfo{
			SigningRegion: region,
			SigningName:   awsS3Service,
		},
		Config: aws.Config{
			Credentials: credentials.NewStaticCredentials(h.accessKeyID, h.secretAccessKey, ""),
		},
		HTTPRequest: r,
		Body:        body,
		Time:        time.Now(),
	})

	return r
}

func guessAWSRegion(host string) string {
	// Separate out potential :<port>
	hostParts := strings.Split(host, ":")
	host = strings.ToLower(hostParts[0])

	parts := strings.Split(host, ".")

	if len(parts) < 3 || parts[len(parts)-2] != "amazonaws" {
		return defaultAWSRegion
	}

	// Try to guess region based on url, but if nothing
	// matches, just fall back to defaultAWSRegion.
	if strings.HasSuffix(host, "s3.amazonaws.com") || strings.HasSuffix(host, "s3-external-1.amazonaws.com") {
		return "us-east-1"
	} else if len(parts) > 3 && strings.HasPrefix(parts[len(parts)-3], "s3-") {
		return parts[len(parts)-3][3:]
	} else if len(parts) > 4 && parts[len(parts)-4] == "s3" {
		return parts[len(parts)-3]
	} else {
		return defaultAWSRegion
	}
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
	case "aws":
		headerer, err = p.getAWSV1Headerer(auth.Credentials)
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
		auth: basic,
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
		auth: oauth,
	}, nil
}

func (p *authV1JsonParser) getAWSV1Headerer(raw json.RawMessage) (Headerer, error) {
	var aws awsV1
	if err := json.Unmarshal(raw, &aws); err != nil {
		return nil, err
	}
	if len(aws.AccessKeyID) == 0 {
		return nil, fmt.Errorf("no AWS Access Key ID specified")
	}
	if len(aws.SecretAccessKey) == 0 {
		return nil, fmt.Errorf("no AWS Secret Access Key specified")
	}
	return &awsAuthHeaderer{
		accessKeyID:     aws.AccessKeyID,
		secretAccessKey: aws.SecretAccessKey,
		region:          aws.Region,
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
