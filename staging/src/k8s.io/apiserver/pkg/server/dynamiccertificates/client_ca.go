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

package dynamiccertificates

import (
	"bytes"
)

// dynamicCertificateContent holds the content that overrides the baseTLSConfig
type dynamicCertificateContent struct {
	// clientCA holds the content for the clientCA bundle
	clientCA    caBundleContent
	servingCert certKeyContent
	sniCerts    []sniCertKeyContent
}

// caBundleContent holds the content for the clientCA bundle.  Wrapping the bytes makes the Equals work nicely with the
// method receiver.
type caBundleContent struct {
	caBundle []byte
}

func (c *dynamicCertificateContent) Equal(rhs *dynamicCertificateContent) bool {
	if c == nil || rhs == nil {
		return c == rhs
	}

	if !c.clientCA.Equal(&rhs.clientCA) {
		return false
	}

	if !c.servingCert.Equal(&rhs.servingCert) {
		return false
	}

	if len(c.sniCerts) != len(rhs.sniCerts) {
		return false
	}

	for i := range c.sniCerts {
		if !c.sniCerts[i].Equal(&rhs.sniCerts[i]) {
			return false
		}
	}

	return true
}

func (c *caBundleContent) Equal(rhs *caBundleContent) bool {
	if c == nil || rhs == nil {
		return c == rhs
	}

	return bytes.Equal(c.caBundle, rhs.caBundle)
}
