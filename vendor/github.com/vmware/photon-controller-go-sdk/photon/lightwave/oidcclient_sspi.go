// Copyright (c) 2017 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

// +build windows

package lightwave

import (
	"encoding/base64"
	"fmt"
	"github.com/vmware/photon-controller-go-sdk/SSPI"
	"math/rand"
	"net"
	"net/url"
	"strings"
	"time"
)

const gssTicketGrantFormatString = "grant_type=urn:vmware:grant_type:gss_ticket&gss_ticket=%s&context_id=%s&scope=%s"

// GetTokensFromWindowsLogInContext gets tokens based on Windows logged in context
// Here is how it works:
// 1. Get the SPN (Service Principal Name) in the format host/FQDN of lightwave. This is needed for SSPI/Kerberos protocol
// 2. Call Windows API AcquireCredentialsHandle() using SSPI library. This will give the current users credential handle
// 3. Using this handle call Windows API AcquireCredentialsHandle(). This will give you byte[]
// 4. Encode this byte[] and send it to OIDC server over HTTP (using POST)
// 5. OIDC server can send either of the following
//    - Access tokens. In this case return access tokens to client
//    - Error in the format: invalid_grant: gss_continue_needed:'context id':'token from server'
// 6. In case you get error, parse it and get the token from server
// 7. Feed this token to step 3 and repeat steps till you get the access tokens from server
func (client *OIDCClient) GetTokensFromWindowsLogInContext() (tokens *OIDCTokenResponse, err error) {
	spn, err := client.buildSPN()
	if err != nil {
		return nil, err
	}

	auth, _ := SSPI.GetAuth("", "", spn, "")

	userContext, err := auth.InitialBytes()
	if err != nil {
		return nil, err
	}

	// In case of multiple req/res between client and server (as explained in above comment),
	// server needs to maintain the mapping of context id -> token
	// So we need to generate random string as a context id
	// If we use same context id for all the requests, results can be erroneous
	contextId := client.generateRandomString()
	body := fmt.Sprintf(gssTicketGrantFormatString, url.QueryEscape(base64.StdEncoding.EncodeToString(userContext)), contextId, client.Options.TokenScope)
	tokens, err = client.getToken(body)

	for {
		if err == nil {
			break
		}

		// In case of error the response will be in format: invalid_grant: gss_continue_needed:'context id':'token from server'
		gssToken := client.validateAndExtractGSSResponse(err, contextId)
		if gssToken == "" {
			return nil, err
		}

		data, err := base64.StdEncoding.DecodeString(gssToken)
		if err != nil {
			return nil, err
		}

		userContext, err := auth.NextBytes(data)
		body := fmt.Sprintf(gssTicketGrantFormatString, url.QueryEscape(base64.StdEncoding.EncodeToString(userContext)), contextId, client.Options.TokenScope)
		tokens, err = client.getToken(body)
	}

	return tokens, err
}

// Gets the SPN (Service Principal Name) in the format host/FQDN of lightwave
func (client *OIDCClient) buildSPN() (spn string, err error) {
	u, err := url.Parse(client.Endpoint)
	if err != nil {
		return "", err
	}

	host, _, err := net.SplitHostPort(u.Host)
	if err != nil {
		return "", err
	}

	addr, err := net.LookupAddr(host)
	if err != nil {
		return "", err
	}

	var s = strings.TrimSuffix(addr[0], ".")
	return "host/" + s, nil
}

// validateAndExtractGSSResponse parse the error from server and returns token from server
// In case of error from the server, response will be in format: invalid_grant: gss_continue_needed:'context id':'token from server'
// So, we check for the above format in error and then return the token from server
// If error is not in above format, we return empty string
func (client *OIDCClient) validateAndExtractGSSResponse(err error, contextId string) string {
	parts := strings.Split(err.Error(), ":")
	if !(len(parts) == 4 && strings.TrimSpace(parts[1]) == "gss_continue_needed" && parts[2] == contextId) {
		return ""
	} else {
		return parts[3]
	}
}

func (client *OIDCClient) generateRandomString() string {
	const length = 10
	const asciiA = 65
	const asciiZ = 90
	rand.Seed(time.Now().UTC().UnixNano())
	bytes := make([]byte, length)
	for i := 0; i < length; i++ {
		bytes[i] = byte(randInt(asciiA, asciiZ))
	}
	return string(bytes)
}

func randInt(min int, max int) int {
	return min + rand.Intn(max-min)
}
