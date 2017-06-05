// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agent

import "testing"

func addTestKey(t *testing.T, a Agent, keyName string) {
	err := a.Add(AddedKey{
		PrivateKey: testPrivateKeys[keyName],
		Comment:    keyName,
	})
	if err != nil {
		t.Fatalf("failed to add key %q: %v", keyName, err)
	}
}

func removeTestKey(t *testing.T, a Agent, keyName string) {
	err := a.Remove(testPublicKeys[keyName])
	if err != nil {
		t.Fatalf("failed to remove key %q: %v", keyName, err)
	}
}

func validateListedKeys(t *testing.T, a Agent, expectedKeys []string) {
	listedKeys, err := a.List()
	if err != nil {
		t.Fatalf("failed to list keys: %v", err)
		return
	}
	actualKeys := make(map[string]bool)
	for _, key := range listedKeys {
		actualKeys[key.Comment] = true
	}

	matchedKeys := make(map[string]bool)
	for _, expectedKey := range expectedKeys {
		if !actualKeys[expectedKey] {
			t.Fatalf("expected key %q, but was not found", expectedKey)
		} else {
			matchedKeys[expectedKey] = true
		}
	}

	for actualKey := range actualKeys {
		if !matchedKeys[actualKey] {
			t.Fatalf("key %q was found, but was not expected", actualKey)
		}
	}
}

func TestKeyringAddingAndRemoving(t *testing.T) {
	keyNames := []string{"dsa", "ecdsa", "rsa", "user"}

	// add all test private keys
	k := NewKeyring()
	for _, keyName := range keyNames {
		addTestKey(t, k, keyName)
	}
	validateListedKeys(t, k, keyNames)

	// remove a key in the middle
	keyToRemove := keyNames[1]
	keyNames = append(keyNames[:1], keyNames[2:]...)

	removeTestKey(t, k, keyToRemove)
	validateListedKeys(t, k, keyNames)

	// remove all keys
	err := k.RemoveAll()
	if err != nil {
		t.Fatalf("failed to remove all keys: %v", err)
	}
	validateListedKeys(t, k, []string{})
}
