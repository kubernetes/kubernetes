// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package photon

import (
	"encoding/json"
	"fmt"
	"github.com/vmware/photon-controller-go-sdk/photon/lightwave"
)

// Contains functionality for auth API.
type AuthAPI struct {
	client *Client
}

const authUrl string = "/auth"

// Gets authentication info.
func (api *AuthAPI) Get() (info *AuthInfo, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+authUrl, "")
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	info = &AuthInfo{}
	err = json.NewDecoder(res.Body).Decode(info)
	return
}

// Gets Tokens from username/password.
func (api *AuthAPI) GetTokensByPassword(username string, password string) (tokenOptions *TokenOptions, err error) {
	oidcClient, err := api.buildOIDCClient()
	if err != nil {
		return
	}

	tokenResponse, err := oidcClient.GetTokenByPasswordGrant(username, password)
	if err != nil {
		return
	}

	return api.toTokenOptions(tokenResponse), nil
}

// Gets tokens from refresh token.
func (api *AuthAPI) GetTokensByRefreshToken(refreshtoken string) (tokenOptions *TokenOptions, err error) {
	oidcClient, err := api.buildOIDCClient()
	if err != nil {
		return
	}

	tokenResponse, err := oidcClient.GetTokenByRefreshTokenGrant(refreshtoken)
	if err != nil {
		return
	}

	return api.toTokenOptions(tokenResponse), nil
}

func (api *AuthAPI) getAuthEndpoint() (endpoint string, err error) {
	authInfo, err := api.client.Auth.Get()
	if err != nil {
		return
	}

	if !authInfo.Enabled {
		return "", SdkError{Message: "Authentication not enabled on this endpoint"}
	}

	if authInfo.Port == 0 {
		authInfo.Port = 443
	}

	return fmt.Sprintf("https://%s:%d", authInfo.Endpoint, authInfo.Port), nil
}

func (api *AuthAPI) buildOIDCClient() (client *lightwave.OIDCClient, err error) {
	authEndPoint, err := api.getAuthEndpoint()
	if err != nil {
		return
	}

	return lightwave.NewOIDCClient(
		authEndPoint,
		api.buildOIDCClientOptions(&api.client.options),
		api.client.restClient.logger), nil
}

const tokenScope string = "openid offline_access rs_esxcloud at_groups"

func (api *AuthAPI) buildOIDCClientOptions(options *ClientOptions) *lightwave.OIDCClientOptions {
	return &lightwave.OIDCClientOptions{
		IgnoreCertificate: api.client.options.IgnoreCertificate,
		RootCAs:           api.client.options.RootCAs,
		TokenScope:        tokenScope,
	}
}

func (api *AuthAPI) toTokenOptions(response *lightwave.OIDCTokenResponse) *TokenOptions {
	return &TokenOptions{
		AccessToken:  response.AccessToken,
		ExpiresIn:    response.ExpiresIn,
		RefreshToken: response.RefreshToken,
		IdToken:      response.IdToken,
		TokenType:    response.TokenType,
	}
}

// Parse the given token details.
func (api *AuthAPI) parseTokenDetails(token string) (jwtToken *lightwave.JWTToken, err error) {
	jwtToken = lightwave.ParseTokenDetails(token)
	return jwtToken, nil
}

// Parse the given token raw details.
func (api *AuthAPI) parseRawTokenDetails(token string) (jwtToken []string, err error) {
	jwtToken, err = lightwave.ParseRawTokenDetails(token)
	return jwtToken, err
}
