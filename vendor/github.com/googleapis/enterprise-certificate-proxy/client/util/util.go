// Package util provides helper functions for the client.
package util

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
)

const configFileName = "enterprise_certificate_config.json"

// EnterpriseCertificateConfig contains parameters for initializing signer.
type EnterpriseCertificateConfig struct {
	Libs Libs `json:"libs"`
}

// Libs specifies the locations of helper libraries.
type Libs struct {
	SignerBinary string `json:"signer_binary"`
}

// LoadSignerBinaryPath retrieves the path of the signer binary from the config file.
func LoadSignerBinaryPath(configFilePath string) (path string, err error) {
	jsonFile, err := os.Open(configFilePath)
	if err != nil {
		return "", err
	}

	byteValue, err := ioutil.ReadAll(jsonFile)
	if err != nil {
		return "", err
	}
	var config EnterpriseCertificateConfig
	err = json.Unmarshal(byteValue, &config)
	if err != nil {
		return "", err
	}
	signerBinaryPath := config.Libs.SignerBinary
	if signerBinaryPath == "" {
		return "", errors.New("Signer binary path is missing.")
	}
	return signerBinaryPath, nil
}

func guessHomeDir() string {
	// Prefer $HOME over user.Current due to glibc bug: golang.org/issue/13470
	if v := os.Getenv("HOME"); v != "" {
		return v
	}
	// Else, fall back to user.Current:
	if u, err := user.Current(); err == nil {
		return u.HomeDir
	}
	return ""
}

func getDefaultConfigFileDirectory() (directory string) {
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("APPDATA"), "gcloud")
	} else {
		return filepath.Join(guessHomeDir(), ".config/gcloud")
	}
}

// GetDefaultConfigFilePath returns the default path of the enterprise certificate config file created by gCloud.
func GetDefaultConfigFilePath() (path string) {
	return filepath.Join(getDefaultConfigFileDirectory(), configFileName)
}
