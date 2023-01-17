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

const configFileName = "certificate_config.json"

// EnterpriseCertificateConfig contains parameters for initializing signer.
type EnterpriseCertificateConfig struct {
	Libs Libs `json:"libs"`
}

// Libs specifies the locations of helper libraries.
type Libs struct {
	ECP string `json:"ecp"`
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
	signerBinaryPath := config.Libs.ECP
	if signerBinaryPath == "" {
		return "", errors.New("signer binary path is missing")
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
	}
	return filepath.Join(guessHomeDir(), ".config/gcloud")
}

// GetDefaultConfigFilePath returns the default path of the enterprise certificate config file created by gCloud.
func GetDefaultConfigFilePath() (path string) {
	return filepath.Join(getDefaultConfigFileDirectory(), configFileName)
}
