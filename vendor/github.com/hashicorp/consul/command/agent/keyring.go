package agent

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/memberlist"
	"github.com/hashicorp/serf/serf"
)

const (
	serfLANKeyring = "serf/local.keyring"
	serfWANKeyring = "serf/remote.keyring"
)

// initKeyring will create a keyring file at a given path.
func initKeyring(path, key string) error {
	var keys []string

	if _, err := base64.StdEncoding.DecodeString(key); err != nil {
		return fmt.Errorf("Invalid key: %s", err)
	}

	// Just exit if the file already exists.
	if _, err := os.Stat(path); err == nil {
		return nil
	}

	keys = append(keys, key)
	keyringBytes, err := json.Marshal(keys)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}

	fh, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0600)
	if err != nil {
		return err
	}
	defer fh.Close()

	if _, err := fh.Write(keyringBytes); err != nil {
		os.Remove(path)
		return err
	}

	return nil
}

// loadKeyringFile will load a gossip encryption keyring out of a file. The file
// must be in JSON format and contain a list of encryption key strings.
func loadKeyringFile(c *serf.Config) error {
	if c.KeyringFile == "" {
		return nil
	}

	if _, err := os.Stat(c.KeyringFile); err != nil {
		return err
	}

	// Read in the keyring file data
	keyringData, err := ioutil.ReadFile(c.KeyringFile)
	if err != nil {
		return err
	}

	// Decode keyring JSON
	keys := make([]string, 0)
	if err := json.Unmarshal(keyringData, &keys); err != nil {
		return err
	}

	// Decode base64 values
	keysDecoded := make([][]byte, len(keys))
	for i, key := range keys {
		keyBytes, err := base64.StdEncoding.DecodeString(key)
		if err != nil {
			return err
		}
		keysDecoded[i] = keyBytes
	}

	// Guard against empty keyring
	if len(keysDecoded) == 0 {
		return fmt.Errorf("no keys present in keyring file: %s", c.KeyringFile)
	}

	// Create the keyring
	keyring, err := memberlist.NewKeyring(keysDecoded, keysDecoded[0])
	if err != nil {
		return err
	}

	c.MemberlistConfig.Keyring = keyring

	// Success!
	return nil
}

// keyringProcess is used to abstract away the semantic similarities in
// performing various operations on the encryption keyring.
func (a *Agent) keyringProcess(args *structs.KeyringRequest) (*structs.KeyringResponses, error) {
	var reply structs.KeyringResponses
	if a.server == nil {
		return nil, fmt.Errorf("keyring operations must run against a server node")
	}
	if err := a.RPC("Internal.KeyringOperation", args, &reply); err != nil {
		return &reply, err
	}

	return &reply, nil
}

// ListKeys lists out all keys installed on the collective Consul cluster. This
// includes both servers and clients in all DC's.
func (a *Agent) ListKeys(token string) (*structs.KeyringResponses, error) {
	args := structs.KeyringRequest{Operation: structs.KeyringList}
	args.Token = token
	return a.keyringProcess(&args)
}

// InstallKey installs a new gossip encryption key
func (a *Agent) InstallKey(key, token string) (*structs.KeyringResponses, error) {
	args := structs.KeyringRequest{Key: key, Operation: structs.KeyringInstall}
	args.Token = token
	return a.keyringProcess(&args)
}

// UseKey changes the primary encryption key used to encrypt messages
func (a *Agent) UseKey(key, token string) (*structs.KeyringResponses, error) {
	args := structs.KeyringRequest{Key: key, Operation: structs.KeyringUse}
	args.Token = token
	return a.keyringProcess(&args)
}

// RemoveKey will remove a gossip encryption key from the keyring
func (a *Agent) RemoveKey(key, token string) (*structs.KeyringResponses, error) {
	args := structs.KeyringRequest{Key: key, Operation: structs.KeyringRemove}
	args.Token = token
	return a.keyringProcess(&args)
}
