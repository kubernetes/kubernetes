package adal

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
	"net/http"
)

const (
	contentType      = "Content-Type"
	mimeTypeFormPost = "application/x-www-form-urlencoded"
)

// Sender is the interface that wraps the Do method to send HTTP requests.
//
// The standard http.Client conforms to this interface.
type Sender interface {
	Do(*http.Request) (*http.Response, error)
}

// SenderFunc is a method that implements the Sender interface.
type SenderFunc func(*http.Request) (*http.Response, error)

// Do implements the Sender interface on SenderFunc.
func (sf SenderFunc) Do(r *http.Request) (*http.Response, error) {
	return sf(r)
}

// SendDecorator takes and possibly decorates, by wrapping, a Sender. Decorators may affect the
// http.Request and pass it along or, first, pass the http.Request along then react to the
// http.Response result.
type SendDecorator func(Sender) Sender

// CreateSender creates, decorates, and returns, as a Sender, the default http.Client.
func CreateSender(decorators ...SendDecorator) Sender {
	return DecorateSender(&http.Client{}, decorators...)
}

// DecorateSender accepts a Sender and a, possibly empty, set of SendDecorators, which is applies to
// the Sender. Decorators are applied in the order received, but their affect upon the request
// depends on whether they are a pre-decorator (change the http.Request and then pass it along) or a
// post-decorator (pass the http.Request along and react to the results in http.Response).
func DecorateSender(s Sender, decorators ...SendDecorator) Sender {
	for _, decorate := range decorators {
		s = decorate(s)
	}
	return s
}
