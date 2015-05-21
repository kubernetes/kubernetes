/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package rkt

import (
	"bytes"
	"fmt"

	utilErrors "github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/coreos/rkt/pkg/keystore"
)

// The trusted key object. More information about `rkt trust` can be found in
// https://github.com/coreos/rkt/blob/master/Documentation/commands.md#rkt-trust.
type TrustedKey struct {
	// The name of the key.
	Name string `json:"name" description:"the name of the key"`
	// The prefix to limit trust to.
	Prefix string `json:"prefix,omitempty" description:"the prefix to limit trust to"`
	// Whether add this key as a root key.
	Root bool `json:"root,omitempty" description:"whether this key is a root key"`
	// The data of the key, it is base64 encoded in order to be unmarshaled by json.
	Data []byte `json:"data" description:"the base64 encoded key data"`
}

// importTrustedKeys reads a file which containers the trusted key list, and then imports
// those trusted keys.
func importTrustedKeys(trustedKeys []TrustedKey) error {
	var errlist []error

	ks := keystore.New(nil)
	for _, key := range trustedKeys {
		if key.Root {
			if _, err := ks.StoreTrustedKeyRoot(bytes.NewReader(key.Data)); err != nil {
				errlist = append(errlist, fmt.Errorf("key %q is not imported: %v", key.Name, err))
			}
			continue
		}

		if key.Prefix != "" {
			if _, err := ks.StoreTrustedKeyPrefix(key.Prefix, bytes.NewReader(key.Data)); err != nil {
				errlist = append(errlist, fmt.Errorf("key %q is not imported: %v", key.Name, err))
			}
			continue
		}

		errlist = append(errlist, fmt.Errorf("key %q is not imported: must specify either root or prefix"))
	}

	return utilErrors.NewAggregate(errlist)
}
