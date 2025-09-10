// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package authn

import "context"

type multiKeychain struct {
	keychains []Keychain
}

// Assert that our multi-keychain implements Keychain.
var _ (Keychain) = (*multiKeychain)(nil)

// NewMultiKeychain composes a list of keychains into one new keychain.
func NewMultiKeychain(kcs ...Keychain) Keychain {
	return &multiKeychain{keychains: kcs}
}

// Resolve implements Keychain.
func (mk *multiKeychain) Resolve(target Resource) (Authenticator, error) {
	return mk.ResolveContext(context.Background(), target)
}

func (mk *multiKeychain) ResolveContext(ctx context.Context, target Resource) (Authenticator, error) {
	for _, kc := range mk.keychains {
		auth, err := Resolve(ctx, kc, target)
		if err != nil {
			return nil, err
		}
		if auth != Anonymous {
			return auth, nil
		}
	}
	return Anonymous, nil
}
