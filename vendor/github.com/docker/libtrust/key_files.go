package libtrust

import (
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

var (
	// ErrKeyFileDoesNotExist indicates that the private key file does not exist.
	ErrKeyFileDoesNotExist = errors.New("key file does not exist")
)

func readKeyFileBytes(filename string) ([]byte, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		if os.IsNotExist(err) {
			err = ErrKeyFileDoesNotExist
		} else {
			err = fmt.Errorf("unable to read key file %s: %s", filename, err)
		}

		return nil, err
	}

	return data, nil
}

/*
	Loading and Saving of Public and Private Keys in either PEM or JWK format.
*/

// LoadKeyFile opens the given filename and attempts to read a Private Key
// encoded in either PEM or JWK format (if .json or .jwk file extension).
func LoadKeyFile(filename string) (PrivateKey, error) {
	contents, err := readKeyFileBytes(filename)
	if err != nil {
		return nil, err
	}

	var key PrivateKey

	if strings.HasSuffix(filename, ".json") || strings.HasSuffix(filename, ".jwk") {
		key, err = UnmarshalPrivateKeyJWK(contents)
		if err != nil {
			return nil, fmt.Errorf("unable to decode private key JWK: %s", err)
		}
	} else {
		key, err = UnmarshalPrivateKeyPEM(contents)
		if err != nil {
			return nil, fmt.Errorf("unable to decode private key PEM: %s", err)
		}
	}

	return key, nil
}

// LoadPublicKeyFile opens the given filename and attempts to read a Public Key
// encoded in either PEM or JWK format (if .json or .jwk file extension).
func LoadPublicKeyFile(filename string) (PublicKey, error) {
	contents, err := readKeyFileBytes(filename)
	if err != nil {
		return nil, err
	}

	var key PublicKey

	if strings.HasSuffix(filename, ".json") || strings.HasSuffix(filename, ".jwk") {
		key, err = UnmarshalPublicKeyJWK(contents)
		if err != nil {
			return nil, fmt.Errorf("unable to decode public key JWK: %s", err)
		}
	} else {
		key, err = UnmarshalPublicKeyPEM(contents)
		if err != nil {
			return nil, fmt.Errorf("unable to decode public key PEM: %s", err)
		}
	}

	return key, nil
}

// SaveKey saves the given key to a file using the provided filename.
// This process will overwrite any existing file at the provided location.
func SaveKey(filename string, key PrivateKey) error {
	var encodedKey []byte
	var err error

	if strings.HasSuffix(filename, ".json") || strings.HasSuffix(filename, ".jwk") {
		// Encode in JSON Web Key format.
		encodedKey, err = json.MarshalIndent(key, "", "    ")
		if err != nil {
			return fmt.Errorf("unable to encode private key JWK: %s", err)
		}
	} else {
		// Encode in PEM format.
		pemBlock, err := key.PEMBlock()
		if err != nil {
			return fmt.Errorf("unable to encode private key PEM: %s", err)
		}
		encodedKey = pem.EncodeToMemory(pemBlock)
	}

	err = ioutil.WriteFile(filename, encodedKey, os.FileMode(0600))
	if err != nil {
		return fmt.Errorf("unable to write private key file %s: %s", filename, err)
	}

	return nil
}

// SavePublicKey saves the given public key to the file.
func SavePublicKey(filename string, key PublicKey) error {
	var encodedKey []byte
	var err error

	if strings.HasSuffix(filename, ".json") || strings.HasSuffix(filename, ".jwk") {
		// Encode in JSON Web Key format.
		encodedKey, err = json.MarshalIndent(key, "", "    ")
		if err != nil {
			return fmt.Errorf("unable to encode public key JWK: %s", err)
		}
	} else {
		// Encode in PEM format.
		pemBlock, err := key.PEMBlock()
		if err != nil {
			return fmt.Errorf("unable to encode public key PEM: %s", err)
		}
		encodedKey = pem.EncodeToMemory(pemBlock)
	}

	err = ioutil.WriteFile(filename, encodedKey, os.FileMode(0644))
	if err != nil {
		return fmt.Errorf("unable to write public key file %s: %s", filename, err)
	}

	return nil
}

// Public Key Set files

type jwkSet struct {
	Keys []json.RawMessage `json:"keys"`
}

// LoadKeySetFile loads a key set
func LoadKeySetFile(filename string) ([]PublicKey, error) {
	if strings.HasSuffix(filename, ".json") || strings.HasSuffix(filename, ".jwk") {
		return loadJSONKeySetFile(filename)
	}

	// Must be a PEM format file
	return loadPEMKeySetFile(filename)
}

func loadJSONKeySetRaw(data []byte) ([]json.RawMessage, error) {
	if len(data) == 0 {
		// This is okay, just return an empty slice.
		return []json.RawMessage{}, nil
	}

	keySet := jwkSet{}

	err := json.Unmarshal(data, &keySet)
	if err != nil {
		return nil, fmt.Errorf("unable to decode JSON Web Key Set: %s", err)
	}

	return keySet.Keys, nil
}

func loadJSONKeySetFile(filename string) ([]PublicKey, error) {
	contents, err := readKeyFileBytes(filename)
	if err != nil && err != ErrKeyFileDoesNotExist {
		return nil, err
	}

	return UnmarshalPublicKeyJWKSet(contents)
}

func loadPEMKeySetFile(filename string) ([]PublicKey, error) {
	data, err := readKeyFileBytes(filename)
	if err != nil && err != ErrKeyFileDoesNotExist {
		return nil, err
	}

	return UnmarshalPublicKeyPEMBundle(data)
}

// AddKeySetFile adds a key to a key set
func AddKeySetFile(filename string, key PublicKey) error {
	if strings.HasSuffix(filename, ".json") || strings.HasSuffix(filename, ".jwk") {
		return addKeySetJSONFile(filename, key)
	}

	// Must be a PEM format file
	return addKeySetPEMFile(filename, key)
}

func addKeySetJSONFile(filename string, key PublicKey) error {
	encodedKey, err := json.Marshal(key)
	if err != nil {
		return fmt.Errorf("unable to encode trusted client key: %s", err)
	}

	contents, err := readKeyFileBytes(filename)
	if err != nil && err != ErrKeyFileDoesNotExist {
		return err
	}

	rawEntries, err := loadJSONKeySetRaw(contents)
	if err != nil {
		return err
	}

	rawEntries = append(rawEntries, json.RawMessage(encodedKey))
	entriesWrapper := jwkSet{Keys: rawEntries}

	encodedEntries, err := json.MarshalIndent(entriesWrapper, "", "    ")
	if err != nil {
		return fmt.Errorf("unable to encode trusted client keys: %s", err)
	}

	err = ioutil.WriteFile(filename, encodedEntries, os.FileMode(0644))
	if err != nil {
		return fmt.Errorf("unable to write trusted client keys file %s: %s", filename, err)
	}

	return nil
}

func addKeySetPEMFile(filename string, key PublicKey) error {
	// Encode to PEM, open file for appending, write PEM.
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_APPEND|os.O_RDWR, os.FileMode(0644))
	if err != nil {
		return fmt.Errorf("unable to open trusted client keys file %s: %s", filename, err)
	}
	defer file.Close()

	pemBlock, err := key.PEMBlock()
	if err != nil {
		return fmt.Errorf("unable to encoded trusted key: %s", err)
	}

	_, err = file.Write(pem.EncodeToMemory(pemBlock))
	if err != nil {
		return fmt.Errorf("unable to write trusted keys file: %s", err)
	}

	return nil
}
