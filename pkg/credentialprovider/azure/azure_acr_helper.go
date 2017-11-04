/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Copyright 2017 Microsoft Corporation

MIT License

Copyright (c) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
*/

// Source: https://github.com/Azure/acr-docker-credential-helper/blob/a79b541f3ee761f6cc4511863ed41fb038c19464/src/docker-credential-acr/acr_login.go

package azure

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"unicode"

	jwt "github.com/dgrijalva/jwt-go"
)

type authDirective struct {
	service string
	realm   string
}

type accessTokenPayload struct {
	TenantID string `json:"tid"`
}

type acrTokenPayload struct {
	Expiration int64  `json:"exp"`
	TenantID   string `json:"tenant"`
	Credential string `json:"credential"`
}

type acrAuthResponse struct {
	RefreshToken string `json:"refresh_token"`
}

// 5 minutes buffer time to allow timeshift between local machine and AAD
const timeShiftBuffer = 300
const userAgentHeader = "User-Agent"
const userAgent = "kubernetes-credentialprovider-acr"

const dockerTokenLoginUsernameGUID = "00000000-0000-0000-0000-000000000000"

var client = &http.Client{}

func receiveChallengeFromLoginServer(serverAddress string) (*authDirective, error) {
	challengeURL := url.URL{
		Scheme: "https",
		Host:   serverAddress,
		Path:   "v2/",
	}
	var err error
	var r *http.Request
	r, _ = http.NewRequest("GET", challengeURL.String(), nil)
	r.Header.Add(userAgentHeader, userAgent)

	var challenge *http.Response
	if challenge, err = client.Do(r); err != nil {
		return nil, fmt.Errorf("Error reaching registry endpoint %s, error: %s", challengeURL.String(), err)
	}
	defer challenge.Body.Close()

	if challenge.StatusCode != 401 {
		return nil, fmt.Errorf("Registry did not issue a valid AAD challenge, status: %d", challenge.StatusCode)
	}

	var authHeader []string
	var ok bool
	if authHeader, ok = challenge.Header["Www-Authenticate"]; !ok {
		return nil, fmt.Errorf("Challenge response does not contain header 'Www-Authenticate'")
	}

	if len(authHeader) != 1 {
		return nil, fmt.Errorf("Registry did not issue a valid AAD challenge, authenticate header [%s]",
			strings.Join(authHeader, ", "))
	}

	authSections := strings.SplitN(authHeader[0], " ", 2)
	authType := strings.ToLower(authSections[0])
	var authParams *map[string]string
	if authParams, err = parseAssignments(authSections[1]); err != nil {
		return nil, fmt.Errorf("Unable to understand the contents of Www-Authenticate header %s", authSections[1])
	}

	// verify headers
	if !strings.EqualFold("Bearer", authType) {
		return nil, fmt.Errorf("Www-Authenticate: expected realm: Bearer, actual: %s", authType)
	}
	if len((*authParams)["service"]) == 0 {
		return nil, fmt.Errorf("Www-Authenticate: missing header \"service\"")
	}
	if len((*authParams)["realm"]) == 0 {
		return nil, fmt.Errorf("Www-Authenticate: missing header \"realm\"")
	}

	return &authDirective{
		service: (*authParams)["service"],
		realm:   (*authParams)["realm"],
	}, nil
}

func parseAcrToken(identityToken string) (token *acrTokenPayload, err error) {
	tokenSegments := strings.Split(identityToken, ".")
	if len(tokenSegments) < 2 {
		return nil, fmt.Errorf("Invalid existing refresh token length: %d", len(tokenSegments))
	}
	payloadSegmentEncoded := tokenSegments[1]
	var payloadBytes []byte
	if payloadBytes, err = jwt.DecodeSegment(payloadSegmentEncoded); err != nil {
		return nil, fmt.Errorf("Error decoding payload segment from refresh token, error: %s", err)
	}
	var payload acrTokenPayload
	if err = json.Unmarshal(payloadBytes, &payload); err != nil {
		return nil, fmt.Errorf("Error unmarshalling acr payload, error: %s", err)
	}
	return &payload, nil
}

func performTokenExchange(
	serverAddress string,
	directive *authDirective,
	tenant string,
	accessToken string) (string, error) {
	var err error
	data := url.Values{
		"service":       []string{directive.service},
		"grant_type":    []string{"access_token_refresh_token"},
		"access_token":  []string{accessToken},
		"refresh_token": []string{accessToken},
		"tenant":        []string{tenant},
	}

	var realmURL *url.URL
	if realmURL, err = url.Parse(directive.realm); err != nil {
		return "", fmt.Errorf("Www-Authenticate: invalid realm %s", directive.realm)
	}
	authEndpoint := fmt.Sprintf("%s://%s/oauth2/exchange", realmURL.Scheme, realmURL.Host)

	datac := data.Encode()
	var r *http.Request
	r, _ = http.NewRequest("POST", authEndpoint, bytes.NewBufferString(datac))
	r.Header.Add(userAgentHeader, userAgent)
	r.Header.Add("Content-Type", "application/x-www-form-urlencoded")
	r.Header.Add("Content-Length", strconv.Itoa(len(datac)))

	var exchange *http.Response
	if exchange, err = client.Do(r); err != nil {
		return "", fmt.Errorf("Www-Authenticate: failed to reach auth url %s", authEndpoint)
	}

	defer exchange.Body.Close()
	if exchange.StatusCode != 200 {
		return "", fmt.Errorf("Www-Authenticate: auth url %s responded with status code %d", authEndpoint, exchange.StatusCode)
	}

	var content []byte
	if content, err = ioutil.ReadAll(exchange.Body); err != nil {
		return "", fmt.Errorf("Www-Authenticate: error reading response from %s", authEndpoint)
	}

	var authResp acrAuthResponse
	if err = json.Unmarshal(content, &authResp); err != nil {
		return "", fmt.Errorf("Www-Authenticate: unable to read response %s", content)
	}

	return authResp.RefreshToken, nil
}

// Try and parse a string of assignments in the form of:
// key1 = value1, key2 = "value 2", key3 = ""
// Note: this method and handle quotes but does not handle escaping of quotes
func parseAssignments(statements string) (*map[string]string, error) {
	var cursor int
	result := make(map[string]string)
	var errorMsg = fmt.Errorf("malformed header value: %s", statements)
	for {
		// parse key
		equalIndex := nextOccurrence(statements, cursor, "=")
		if equalIndex == -1 {
			return nil, errorMsg
		}
		key := strings.TrimSpace(statements[cursor:equalIndex])

		// parse value
		cursor = nextNoneSpace(statements, equalIndex+1)
		if cursor == -1 {
			return nil, errorMsg
		}
		// case: value is quoted
		if statements[cursor] == '"' {
			cursor = cursor + 1
			// like I said, not handling escapes, but this will skip any comma that's
			// within the quotes which is somewhat more likely
			closeQuoteIndex := nextOccurrence(statements, cursor, "\"")
			if closeQuoteIndex == -1 {
				return nil, errorMsg
			}
			value := statements[cursor:closeQuoteIndex]
			result[key] = value

			commaIndex := nextNoneSpace(statements, closeQuoteIndex+1)
			if commaIndex == -1 {
				// no more comma, done
				return &result, nil
			} else if statements[commaIndex] != ',' {
				// expect comma immediately after close quote
				return nil, errorMsg
			} else {
				cursor = commaIndex + 1
			}
		} else {
			commaIndex := nextOccurrence(statements, cursor, ",")
			endStatements := commaIndex == -1
			var untrimmed string
			if endStatements {
				untrimmed = statements[cursor:commaIndex]
			} else {
				untrimmed = statements[cursor:]
			}
			value := strings.TrimSpace(untrimmed)

			if len(value) == 0 {
				// disallow empty value without quote
				return nil, errorMsg
			}

			result[key] = value

			if endStatements {
				return &result, nil
			}
			cursor = commaIndex + 1
		}
	}
}

func nextOccurrence(str string, start int, sep string) int {
	if start >= len(str) {
		return -1
	}
	offset := strings.Index(str[start:], sep)
	if offset == -1 {
		return -1
	}
	return offset + start
}

func nextNoneSpace(str string, start int) int {
	if start >= len(str) {
		return -1
	}
	offset := strings.IndexFunc(str[start:], func(c rune) bool { return !unicode.IsSpace(c) })
	if offset == -1 {
		return -1
	}
	return offset + start
}
