// Copyright 2018 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build go1.9

package option

import (
	"golang.org/x/oauth2/google"
	"google.golang.org/api/internal"
)

type withCreds google.Credentials

func (w *withCreds) Apply(o *internal.DialSettings) {
	o.Credentials = (*google.Credentials)(w)
}

func WithCredentials(creds *google.Credentials) ClientOption {
	return (*withCreds)(creds)
}
