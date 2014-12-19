/*
Copyright 2014 Google Inc. All rights reserved.

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

package osinserver

import (
	"fmt"
	"net/http"

	"github.com/RangelReale/osin"
)

// NewDefaultServerConfig returns an osin.ServerConfig that allows all authorize and access types
func NewDefaultServerConfig() *osin.ServerConfig {
	config := osin.NewServerConfig()

	config.AllowedAuthorizeTypes = osin.AllowedAuthorizeType{
		osin.CODE,
		osin.TOKEN,
	}
	config.AllowedAccessTypes = osin.AllowedAccessType{
		osin.AUTHORIZATION_CODE,
		osin.REFRESH_TOKEN,
		osin.PASSWORD,
		osin.CLIENT_CREDENTIALS,
		osin.ASSERTION,
	}
	config.AllowGetAccessRequest = true
	config.RedirectUriSeparator = ","
	config.ErrorStatusCode = 400

	return config
}

// defaultError implements ErrorHandler
type defaultErrorHandler struct{}

// NewDefaultErrorHandler returns a simple ErrorHandler
func NewDefaultErrorHandler() ErrorHandler {
	return defaultErrorHandler{}
}

// HandleError implements ErrorHandler
func (defaultErrorHandler) HandleError(err error, w http.ResponseWriter, req *http.Request) {
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprintf(w, "Error: %s", err)
}
