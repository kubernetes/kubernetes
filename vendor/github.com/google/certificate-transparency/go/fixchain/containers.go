package fixchain

import (
	"sync"

	"github.com/google/certificate-transparency/go/x509"
)

type dedupedChain struct {
	certs []*x509.Certificate
}

func (d *dedupedChain) addCert(cert *x509.Certificate) {
	// Check that the certificate isn't being added twice.
	for _, c := range d.certs {
		if c.Equal(cert) {
			return
		}
	}
	d.certs = append(d.certs, cert)
}

func (d *dedupedChain) addCertToFront(cert *x509.Certificate) {
	// Check that the certificate isn't being added twice.
	for _, c := range d.certs {
		if c.Equal(cert) {
			return
		}
	}
	d.certs = append([]*x509.Certificate{cert}, d.certs...)
}

func newDedupedChain(chain []*x509.Certificate) *dedupedChain {
	d := &dedupedChain{}
	for _, cert := range chain {
		d.addCert(cert)
	}
	return d
}

type lockedMap struct {
	m map[[hashSize]byte]bool
	sync.RWMutex
}

func newLockedMap() *lockedMap {
	return &lockedMap{m: make(map[[hashSize]byte]bool)}
}

func (m *lockedMap) get(hash [hashSize]byte) bool {
	m.RLock()
	defer m.RUnlock()
	return m.m[hash]
}

func (m *lockedMap) set(hash [hashSize]byte, b bool) {
	m.Lock()
	defer m.Unlock()
	m.m[hash] = b
}
