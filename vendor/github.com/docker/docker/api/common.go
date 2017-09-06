package api

import (
	"encoding/json"
	"encoding/pem"
	"fmt"
	"os"
	"path/filepath"

	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/libtrust"
)

// Common constants for daemon and client.
const (
	// DefaultVersion of Current REST API
	DefaultVersion string = "1.31"

	// NoBaseImageSpecifier is the symbol used by the FROM
	// command to specify that no base image is to be used.
	NoBaseImageSpecifier string = "scratch"
)

// LoadOrCreateTrustKey attempts to load the libtrust key at the given path,
// otherwise generates a new one
func LoadOrCreateTrustKey(trustKeyPath string) (libtrust.PrivateKey, error) {
	err := system.MkdirAll(filepath.Dir(trustKeyPath), 0700, "")
	if err != nil {
		return nil, err
	}
	trustKey, err := libtrust.LoadKeyFile(trustKeyPath)
	if err == libtrust.ErrKeyFileDoesNotExist {
		trustKey, err = libtrust.GenerateECP256PrivateKey()
		if err != nil {
			return nil, fmt.Errorf("Error generating key: %s", err)
		}
		encodedKey, err := serializePrivateKey(trustKey, filepath.Ext(trustKeyPath))
		if err != nil {
			return nil, fmt.Errorf("Error serializing key: %s", err)
		}
		if err := ioutils.AtomicWriteFile(trustKeyPath, encodedKey, os.FileMode(0600)); err != nil {
			return nil, fmt.Errorf("Error saving key file: %s", err)
		}
	} else if err != nil {
		return nil, fmt.Errorf("Error loading key file %s: %s", trustKeyPath, err)
	}
	return trustKey, nil
}

func serializePrivateKey(key libtrust.PrivateKey, ext string) (encoded []byte, err error) {
	if ext == ".json" || ext == ".jwk" {
		encoded, err = json.Marshal(key)
		if err != nil {
			return nil, fmt.Errorf("unable to encode private key JWK: %s", err)
		}
	} else {
		pemBlock, err := key.PEMBlock()
		if err != nil {
			return nil, fmt.Errorf("unable to encode private key PEM: %s", err)
		}
		encoded = pem.EncodeToMemory(pemBlock)
	}
	return
}
