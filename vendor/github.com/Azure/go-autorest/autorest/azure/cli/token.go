package cli

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/date"
	"github.com/mitchellh/go-homedir"
)

// Token represents an AccessToken from the Azure CLI
type Token struct {
	AccessToken      string `json:"accessToken"`
	Authority        string `json:"_authority"`
	ClientID         string `json:"_clientId"`
	ExpiresOn        string `json:"expiresOn"`
	IdentityProvider string `json:"identityProvider"`
	IsMRRT           bool   `json:"isMRRT"`
	RefreshToken     string `json:"refreshToken"`
	Resource         string `json:"resource"`
	TokenType        string `json:"tokenType"`
	UserID           string `json:"userId"`
}

// ToADALToken converts an Azure CLI `Token`` to an `adal.Token``
func (t Token) ToADALToken() (converted adal.Token, err error) {
	tokenExpirationDate, err := ParseExpirationDate(t.ExpiresOn)
	if err != nil {
		err = fmt.Errorf("Error parsing Token Expiration Date %q: %+v", t.ExpiresOn, err)
		return
	}

	difference := tokenExpirationDate.Sub(date.UnixEpoch())

	converted = adal.Token{
		AccessToken:  t.AccessToken,
		Type:         t.TokenType,
		ExpiresIn:    "3600",
		ExpiresOn:    strconv.Itoa(int(difference.Seconds())),
		RefreshToken: t.RefreshToken,
		Resource:     t.Resource,
	}
	return
}

// AccessTokensPath returns the path where access tokens are stored from the Azure CLI
func AccessTokensPath() (string, error) {
	return homedir.Expand("~/.azure/accessTokens.json")
}

// ParseExpirationDate parses either a Azure CLI or CloudShell date into a time object
func ParseExpirationDate(input string) (*time.Time, error) {
	// CloudShell (and potentially the Azure CLI in future)
	expirationDate, cloudShellErr := time.Parse(time.RFC3339, input)
	if cloudShellErr != nil {
		// Azure CLI (Python) e.g. 2017-08-31 19:48:57.998857 (plus the local timezone)
		const cliFormat = "2006-01-02 15:04:05.999999"
		expirationDate, cliErr := time.ParseInLocation(cliFormat, input, time.Local)
		if cliErr == nil {
			return &expirationDate, nil
		}

		return nil, fmt.Errorf("Error parsing expiration date %q.\n\nCloudShell Error: \n%+v\n\nCLI Error:\n%+v", input, cloudShellErr, cliErr)
	}

	return &expirationDate, nil
}

// LoadTokens restores a set of Token objects from a file located at 'path'.
func LoadTokens(path string) ([]Token, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file (%s) while loading token: %v", path, err)
	}
	defer file.Close()

	var tokens []Token

	dec := json.NewDecoder(file)
	if err = dec.Decode(&tokens); err != nil {
		return nil, fmt.Errorf("failed to decode contents of file (%s) into a `cli.Token` representation: %v", path, err)
	}

	return tokens, nil
}
