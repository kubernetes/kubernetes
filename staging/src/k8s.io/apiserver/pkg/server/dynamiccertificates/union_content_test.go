/*
Copyright 2026 The Kubernetes Authors.

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
	"crypto/x509"
	"sync/atomic"
	"testing"
)

// flappingCAContent serves a valid CA bundle on its first CurrentCABundleContent call and
// nothing afterwards. This models a ConfigMapCAController whose configmap is deleted between
// two consecutive reads of the same consumer.
type flappingCAContent struct {
	name  string
	ca    []byte
	calls atomic.Int32
}

var _ CAContentProvider = &flappingCAContent{}

func (f *flappingCAContent) Name() string         { return f.name }
func (f *flappingCAContent) AddListener(Listener) {}

func (f *flappingCAContent) CurrentCABundleContent() []byte {
	if f.calls.Add(1) == 1 {
		return f.ca
	}
	return nil
}

func (f *flappingCAContent) VerifyOptions() (x509.VerifyOptions, bool) {
	ca := f.CurrentCABundleContent()
	if len(ca) == 0 {
		return x509.VerifyOptions{}, false
	}
	v, err := newCABundleAndVerifier(f.name, ca)
	if err != nil {
		return x509.VerifyOptions{}, false
	}
	return v.verifyOptions, true
}

func TestUnionCAContentVerifyOptionsSingleRead(t *testing.T) {
	provider := &flappingCAContent{name: "flapping-ca", ca: serverCert}
	union := NewUnionCAContentProvider(provider)

	// the content disappears right after the first read, so VerifyOptions must act on that
	// single observation instead of re-reading and tripping over the now empty bundle
	if _, ok := union.VerifyOptions(); !ok {
		t.Fatal("expected VerifyOptions to build a verifier from the bundle observed by its first read")
	}
	if got := provider.calls.Load(); got != 1 {
		t.Errorf("expected exactly one content read per VerifyOptions call, got %d", got)
	}

	if _, ok := union.VerifyOptions(); ok {
		t.Fatal("expected VerifyOptions to report unavailable once the bundle is empty")
	}
}
