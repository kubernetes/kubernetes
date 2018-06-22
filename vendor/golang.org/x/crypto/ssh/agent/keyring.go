// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agent

import (
	"bytes"
	"crypto/rand"
	"crypto/subtle"
	"errors"
	"fmt"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
)

type privKey struct {
	signer  ssh.Signer
	comment string
	expire  *time.Time
}

type keyring struct {
	mu   sync.Mutex
	keys []privKey

	locked     bool
	passphrase []byte
}

var errLocked = errors.New("agent: locked")

// NewKeyring returns an Agent that holds keys in memory.  It is safe
// for concurrent use by multiple goroutines.
func NewKeyring() Agent {
	return &keyring{}
}

// RemoveAll removes all identities.
func (r *keyring) RemoveAll() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.locked {
		return errLocked
	}

	r.keys = nil
	return nil
}

// removeLocked does the actual key removal. The caller must already be holding the
// keyring mutex.
func (r *keyring) removeLocked(want []byte) error {
	found := false
	for i := 0; i < len(r.keys); {
		if bytes.Equal(r.keys[i].signer.PublicKey().Marshal(), want) {
			found = true
			r.keys[i] = r.keys[len(r.keys)-1]
			r.keys = r.keys[:len(r.keys)-1]
			continue
		} else {
			i++
		}
	}

	if !found {
		return errors.New("agent: key not found")
	}
	return nil
}

// Remove removes all identities with the given public key.
func (r *keyring) Remove(key ssh.PublicKey) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.locked {
		return errLocked
	}

	return r.removeLocked(key.Marshal())
}

// Lock locks the agent. Sign and Remove will fail, and List will return an empty list.
func (r *keyring) Lock(passphrase []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.locked {
		return errLocked
	}

	r.locked = true
	r.passphrase = passphrase
	return nil
}

// Unlock undoes the effect of Lock
func (r *keyring) Unlock(passphrase []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.locked {
		return errors.New("agent: not locked")
	}
	if len(passphrase) != len(r.passphrase) || 1 != subtle.ConstantTimeCompare(passphrase, r.passphrase) {
		return fmt.Errorf("agent: incorrect passphrase")
	}

	r.locked = false
	r.passphrase = nil
	return nil
}

// expireKeysLocked removes expired keys from the keyring. If a key was added
// with a lifetimesecs contraint and seconds >= lifetimesecs seconds have
// ellapsed, it is removed. The caller *must* be holding the keyring mutex.
func (r *keyring) expireKeysLocked() {
	for _, k := range r.keys {
		if k.expire != nil && time.Now().After(*k.expire) {
			r.removeLocked(k.signer.PublicKey().Marshal())
		}
	}
}

// List returns the identities known to the agent.
func (r *keyring) List() ([]*Key, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.locked {
		// section 2.7: locked agents return empty.
		return nil, nil
	}

	r.expireKeysLocked()
	var ids []*Key
	for _, k := range r.keys {
		pub := k.signer.PublicKey()
		ids = append(ids, &Key{
			Format:  pub.Type(),
			Blob:    pub.Marshal(),
			Comment: k.comment})
	}
	return ids, nil
}

// Insert adds a private key to the keyring. If a certificate
// is given, that certificate is added as public key. Note that
// any constraints given are ignored.
func (r *keyring) Add(key AddedKey) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.locked {
		return errLocked
	}
	signer, err := ssh.NewSignerFromKey(key.PrivateKey)

	if err != nil {
		return err
	}

	if cert := key.Certificate; cert != nil {
		signer, err = ssh.NewCertSigner(cert, signer)
		if err != nil {
			return err
		}
	}

	p := privKey{
		signer:  signer,
		comment: key.Comment,
	}

	if key.LifetimeSecs > 0 {
		t := time.Now().Add(time.Duration(key.LifetimeSecs) * time.Second)
		p.expire = &t
	}

	r.keys = append(r.keys, p)

	return nil
}

// Sign returns a signature for the data.
func (r *keyring) Sign(key ssh.PublicKey, data []byte) (*ssh.Signature, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.locked {
		return nil, errLocked
	}

	r.expireKeysLocked()
	wanted := key.Marshal()
	for _, k := range r.keys {
		if bytes.Equal(k.signer.PublicKey().Marshal(), wanted) {
			return k.signer.Sign(rand.Reader, data)
		}
	}
	return nil, errors.New("not found")
}

// Signers returns signers for all the known keys.
func (r *keyring) Signers() ([]ssh.Signer, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.locked {
		return nil, errLocked
	}

	r.expireKeysLocked()
	s := make([]ssh.Signer, 0, len(r.keys))
	for _, k := range r.keys {
		s = append(s, k.signer)
	}
	return s, nil
}
