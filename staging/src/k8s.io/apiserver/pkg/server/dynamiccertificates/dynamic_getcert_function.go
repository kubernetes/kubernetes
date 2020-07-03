/*
Copyright 2020 The Kubernetes Authors.

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
	"crypto/tls"
	"fmt"
	"sync/atomic"
)

type certError struct {
	cert   *tls.Certificate
	errStr string
}

// TODO: doc
type DynamicGetCertFunctionsProvider interface {
	GetCert() (*tls.Certificate, error)
}

// TODO: doc
type DynamicGetCertFunction struct {
	ce                     atomic.Value
	certKeyContentProvider CertKeyContentProvider
}

var _ Listener = &DynamicGetCertFunction{}

// TODO: doc
func NewDynamicGetCertFunction(certKeyContentProvider CertKeyContentProvider) (*DynamicGetCertFunction, error) {
	ret := &DynamicGetCertFunction{
		certKeyContentProvider: certKeyContentProvider,
	}

	ret.Enqueue()
	_, err := ret.GetCert()

	return ret, err
}

func (c *DynamicGetCertFunction) GetCert() (*tls.Certificate, error) {
	rawCertError := c.ce.Load()
	if rawCertError == nil {
		return nil, nil
	}
	certErr, ok := rawCertError.(*certError)
	if !ok {
		return nil, fmt.Errorf("internal error, unable to convert the stored value to *certError")
	}
	if len(certErr.errStr) == 0 {
		return certErr.cert, nil
	}
	return certErr.cert, fmt.Errorf("%s", certErr.errStr)
}

func (c *DynamicGetCertFunction) Enqueue() {
	rawCert, rawKey := c.certKeyContentProvider.CurrentCertKeyContent()
	newCert, err := tls.X509KeyPair(rawCert, rawKey)

	newCertError := &certError{cert: &newCert}
	if err != nil {
		newCertError.errStr = err.Error()
	}

	c.ce.Store(newCertError)
}
