//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azidentity

import (
	"fmt"
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/internal/log"
)

// EventAuthentication entries contain information about authentication.
// This includes information like the names of environment variables
// used when obtaining credentials and the type of credential used.
const EventAuthentication log.Event = "Authentication"

func logGetTokenSuccess(cred azcore.TokenCredential, opts policy.TokenRequestOptions) {
	if !log.Should(EventAuthentication) {
		return
	}
	scope := strings.Join(opts.Scopes, ", ")
	msg := fmt.Sprintf("%T.GetToken() acquired a token for scope %s\n", cred, scope)
	log.Write(EventAuthentication, msg)
}
