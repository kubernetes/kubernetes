package memberlist

import (
	"bytes"
	"fmt"
	"sync"
)

type Keyring struct {
	// Keys stores the key data used during encryption and decryption. It is
	// ordered in such a way where the first key (index 0) is the primary key,
	// which is used for encrypting messages, and is the first key tried during
	// message decryption.
	keys [][]byte

	// The keyring lock is used while performing IO operations on the keyring.
	l sync.Mutex
}

// Init allocates substructures
func (k *Keyring) init() {
	k.keys = make([][]byte, 0)
}

// NewKeyring constructs a new container for a set of encryption keys. The
// keyring contains all key data used internally by memberlist.
//
// While creating a new keyring, you must do one of:
//   - Omit keys and primary key, effectively disabling encryption
//   - Pass a set of keys plus the primary key
//   - Pass only a primary key
//
// If only a primary key is passed, then it will be automatically added to the
// keyring. If creating a keyring with multiple keys, one key must be designated
// primary by passing it as the primaryKey. If the primaryKey does not exist in
// the list of secondary keys, it will be automatically added at position 0.
//
// A key should be either 16, 24, or 32 bytes to select AES-128,
// AES-192, or AES-256.
func NewKeyring(keys [][]byte, primaryKey []byte) (*Keyring, error) {
	keyring := &Keyring{}
	keyring.init()

	if len(keys) > 0 || len(primaryKey) > 0 {
		if len(primaryKey) == 0 {
			return nil, fmt.Errorf("Empty primary key not allowed")
		}
		if err := keyring.AddKey(primaryKey); err != nil {
			return nil, err
		}
		for _, key := range keys {
			if err := keyring.AddKey(key); err != nil {
				return nil, err
			}
		}
	}

	return keyring, nil
}

// ValidateKey will check to see if the key is valid and returns an error if not.
//
// key should be either 16, 24, or 32 bytes to select AES-128,
// AES-192, or AES-256.
func ValidateKey(key []byte) error {
	if l := len(key); l != 16 && l != 24 && l != 32 {
		return fmt.Errorf("key size must be 16, 24 or 32 bytes")
	}
	return nil
}

// AddKey will install a new key on the ring. Adding a key to the ring will make
// it available for use in decryption. If the key already exists on the ring,
// this function will just return noop.
//
// key should be either 16, 24, or 32 bytes to select AES-128,
// AES-192, or AES-256.
func (k *Keyring) AddKey(key []byte) error {
	if err := ValidateKey(key); err != nil {
		return err
	}

	// No-op if key is already installed
	for _, installedKey := range k.keys {
		if bytes.Equal(installedKey, key) {
			return nil
		}
	}

	keys := append(k.keys, key)
	primaryKey := k.GetPrimaryKey()
	if primaryKey == nil {
		primaryKey = key
	}
	k.installKeys(keys, primaryKey)
	return nil
}

// UseKey changes the key used to encrypt messages. This is the only key used to
// encrypt messages, so peers should know this key before this method is called.
func (k *Keyring) UseKey(key []byte) error {
	for _, installedKey := range k.keys {
		if bytes.Equal(key, installedKey) {
			k.installKeys(k.keys, key)
			return nil
		}
	}
	return fmt.Errorf("Requested key is not in the keyring")
}

// RemoveKey drops a key from the keyring. This will return an error if the key
// requested for removal is currently at position 0 (primary key).
func (k *Keyring) RemoveKey(key []byte) error {
	if bytes.Equal(key, k.keys[0]) {
		return fmt.Errorf("Removing the primary key is not allowed")
	}
	for i, installedKey := range k.keys {
		if bytes.Equal(key, installedKey) {
			keys := append(k.keys[:i], k.keys[i+1:]...)
			k.installKeys(keys, k.keys[0])
		}
	}
	return nil
}

// installKeys will take out a lock on the keyring, and replace the keys with a
// new set of keys. The key indicated by primaryKey will be installed as the new
// primary key.
func (k *Keyring) installKeys(keys [][]byte, primaryKey []byte) {
	k.l.Lock()
	defer k.l.Unlock()

	newKeys := [][]byte{primaryKey}
	for _, key := range keys {
		if !bytes.Equal(key, primaryKey) {
			newKeys = append(newKeys, key)
		}
	}
	k.keys = newKeys
}

// GetKeys returns the current set of keys on the ring.
func (k *Keyring) GetKeys() [][]byte {
	k.l.Lock()
	defer k.l.Unlock()

	return k.keys
}

// GetPrimaryKey returns the key on the ring at position 0. This is the key used
// for encrypting messages, and is the first key tried for decrypting messages.
func (k *Keyring) GetPrimaryKey() (key []byte) {
	k.l.Lock()
	defer k.l.Unlock()

	if len(k.keys) > 0 {
		key = k.keys[0]
	}
	return
}
