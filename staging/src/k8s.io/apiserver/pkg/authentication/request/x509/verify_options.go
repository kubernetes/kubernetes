/*
Copyright 2019 The Kubernetes Authors.

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

package x509

import (
	"crypto/x509"
	"fmt"

	"k8s.io/client-go/util/cert"
)

// StaticVerifierFn is a VerifyOptionFunc that always returns the same value.  This allows verify options that cannot change.
func StaticVerifierFn(opts x509.VerifyOptions) VerifyOptionFunc {
	return func() x509.VerifyOptions {
		return opts
	}
}

// NewStaticVerifierFromFile creates a new verification func from a file.  It reads the content and then fails.
// It will return a nil function if you pass an empty CA file.
func NewStaticVerifierFromFile(clientCA string) (VerifyOptionFunc, error) {
	if len(clientCA) == 0 {
		return nil, nil
	}

	// Wrap with an x509 verifier
	var err error
	opts := DefaultVerifyOptions()
	opts.Roots, err = cert.NewPool(clientCA)
	if err != nil {
		return nil, fmt.Errorf("error loading certs from  %s: %v", clientCA, err)
	}

	return StaticVerifierFn(opts), nil
}
