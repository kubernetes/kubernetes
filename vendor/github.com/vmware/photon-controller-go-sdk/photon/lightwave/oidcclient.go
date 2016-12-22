// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package lightwave

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
)

const tokenScope string = "openid offline_access"

type OIDCClient struct {
	httpClient *http.Client
	logger     *log.Logger

	Endpoint string
	Options  *OIDCClientOptions
}

type OIDCClientOptions struct {
	// Whether or not to ignore any TLS errors when talking to photon,
	// false by default.
	IgnoreCertificate bool

	// List of root CA's to use for server validation
	// nil by default.
	RootCAs *x509.CertPool

	// The scope values to use when requesting tokens
	TokenScope string
}

func NewOIDCClient(endpoint string, options *OIDCClientOptions, logger *log.Logger) (c *OIDCClient) {
	if logger == nil {
		logger = log.New(ioutil.Discard, "", log.LstdFlags)
	}

	options = buildOptions(options)
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: options.IgnoreCertificate,
			RootCAs:            options.RootCAs},
	}

	c = &OIDCClient{
		httpClient: &http.Client{Transport: tr},
		logger:     logger,

		Endpoint: strings.TrimRight(endpoint, "/"),
		Options:  options,
	}
	return
}

func buildOptions(options *OIDCClientOptions) (result *OIDCClientOptions) {
	result = &OIDCClientOptions{
		TokenScope: tokenScope,
	}

	if options == nil {
		return
	}

	result.IgnoreCertificate = options.IgnoreCertificate

	if options.RootCAs != nil {
		result.RootCAs = options.RootCAs
	}

	if options.TokenScope != "" {
		result.TokenScope = options.TokenScope
	}

	return
}

func (client *OIDCClient) buildUrl(path string) (url string) {
	return fmt.Sprintf("%s%s", client.Endpoint, path)
}

// Cert download helper

const certDownloadPath string = "/afd/vecs/ssl"

type lightWaveCert struct {
	Value string `json:"encoded"`
}

func (client *OIDCClient) GetRootCerts() (certList []*x509.Certificate, err error) {
	// turn TLS verification off for
	originalTr := client.httpClient.Transport
	defer client.setTransport(originalTr)

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
		},
	}
	client.setTransport(tr)

	// get the certs
	resp, err := client.httpClient.Get(client.buildUrl(certDownloadPath))
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		err = fmt.Errorf("Unexpected error retrieving auth server certs: %v %s", resp.StatusCode, resp.Status)
		return
	}

	// parse the certs
	certsData := &[]lightWaveCert{}
	err = json.NewDecoder(resp.Body).Decode(certsData)
	if err != nil {
		return
	}

	certList = make([]*x509.Certificate, len(*certsData))
	for idx, cert := range *certsData {
		block, _ := pem.Decode([]byte(cert.Value))
		if block == nil {
			err = fmt.Errorf("Unexpected response format: %v", certsData)
			return nil, err
		}

		decodedCert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			return nil, err
		}

		certList[idx] = decodedCert
	}

	return
}

func (client *OIDCClient) setTransport(tr http.RoundTripper) {
	client.httpClient.Transport = tr
}

// Toke request helpers

const tokenPath string = "/openidconnect/token"
const passwordGrantFormatString = "grant_type=password&username=%s&password=%s&scope=%s"
const refreshTokenGrantFormatString = "grant_type=refresh_token&refresh_token=%s"

type OIDCTokenResponse struct {
	AccessToken  string `json:"access_token"`
	ExpiresIn    int    `json:"expires_in"`
	RefreshToken string `json:"refresh_token,omitempty"`
	IdToken      string `json:"id_token"`
	TokenType    string `json:"token_type"`
}

func (client *OIDCClient) GetTokenByPasswordGrant(username string, password string) (tokens *OIDCTokenResponse, err error) {
	body := fmt.Sprintf(passwordGrantFormatString, username, password, client.Options.TokenScope)
	return client.getToken(body)
}

func (client *OIDCClient) GetTokenByRefreshTokenGrant(refreshToken string) (tokens *OIDCTokenResponse, err error) {
	body := fmt.Sprintf(refreshTokenGrantFormatString, refreshToken)
	return client.getToken(body)
}

func (client *OIDCClient) getToken(body string) (tokens *OIDCTokenResponse, err error) {
	request, err := http.NewRequest("POST", client.buildUrl(tokenPath), strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Add("Content-Type", "application/x-www-form-urlencoded")

	resp, err := client.httpClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	err = client.checkResponse(resp)
	if err != nil {
		return nil, err
	}

	tokens = &OIDCTokenResponse{}
	err = json.NewDecoder(resp.Body).Decode(tokens)
	if err != nil {
		return nil, err
	}

	return
}

type OIDCError struct {
	Code    string `json:"error"`
	Message string `json:"error_description"`
}

func (e OIDCError) Error() string {
	return fmt.Sprintf("%v: %v", e.Code, e.Message)
}

func (client *OIDCClient) checkResponse(response *http.Response) (err error) {
	if response.StatusCode/100 == 2 {
		return
	}

	respBody, readErr := ioutil.ReadAll(response.Body)
	if err != nil {
		return fmt.Errorf(
			"Status: %v, Body: %v [%v]", response.Status, string(respBody[:]), readErr)
	}

	var oidcErr OIDCError
	err = json.Unmarshal(respBody, &oidcErr)
	if err != nil {
		return fmt.Errorf(
			"Status: %v, Body: %v [%v]", response.Status, string(respBody[:]), readErr)
	}

	return oidcErr
}
