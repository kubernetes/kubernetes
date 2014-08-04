// Copyright 2014 Docker authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.
//
// This file is a partial mirror of (based on)
// https://github.com/docker/docker/blob/v1.1.2/registry/auth.go
//
// Only parts relevant to decoding the .dockercfg file are mirrored,
// and the helpers for writing the configuration are ignored since
// they are beyond the scope of a docker client library

package utils

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

// Only used for user auth + account creation
const INDEXSERVER = "https://index.docker.io/v1/"

func IndexServerAddress() string {
	return INDEXSERVER
}

type AuthConfig struct {
	Username      string `json:"username,omitempty"`
	Password      string `json:"password,omitempty"`
	Auth          string `json:"auth,omitempty"`
	Email         string `json:"email,omitempty"`
	ServerAddress string `json:"serveraddress,omitempty"`
}

type ConfigFile struct {
	Configs map[string]AuthConfig `json:"configs,omitempty"`
}

// decode the auth string
func decodeAuth(authStr string) (string, string, error) {
	decLen := base64.StdEncoding.DecodedLen(len(authStr))
	decoded := make([]byte, decLen)
	authByte := []byte(authStr)
	n, err := base64.StdEncoding.Decode(decoded, authByte)
	if err != nil {
		return "", "", err
	}
	if n > decLen {
		return "", "", fmt.Errorf("Something went wrong decoding auth config")
	}
	arr := strings.SplitN(string(decoded), ":", 2)
	if len(arr) != 2 {
		return "", "", fmt.Errorf("Invalid auth configuration file")
	}
	password := strings.Trim(arr[1], "\x00")
	return arr[0], password, nil
}

// load up the auth config information and return values
func LoadConfig(configPath string) (*ConfigFile, error) {
	configFile := ConfigFile{Configs: make(map[string]AuthConfig)}
	if _, err := os.Stat(configPath); err != nil {
		return &configFile, err
	}
	b, err := ioutil.ReadFile(configPath)
	if err != nil {
		return &configFile, err
	}

	if err := json.Unmarshal(b, &configFile.Configs); err != nil {
		arr := strings.Split(string(b), "\n")
		if len(arr) < 2 {
			return &configFile, fmt.Errorf("The Auth config file is empty")
		}
		authConfig := AuthConfig{}
		origAuth := strings.Split(arr[0], " = ")
		if len(origAuth) != 2 {
			return &configFile, fmt.Errorf("Invalid Auth config file")
		}
		authConfig.Username, authConfig.Password, err = decodeAuth(origAuth[1])
		if err != nil {
			return &configFile, err
		}
		origEmail := strings.Split(arr[1], " = ")
		if len(origEmail) != 2 {
			return &configFile, fmt.Errorf("Invalid Auth config file")
		}
		authConfig.Email = origEmail[1]
		authConfig.ServerAddress = IndexServerAddress()
		configFile.Configs[IndexServerAddress()] = authConfig
	} else {
		for k, authConfig := range configFile.Configs {
			authConfig.Username, authConfig.Password, err = decodeAuth(authConfig.Auth)
			if err != nil {
				return &configFile, err
			}
			authConfig.Auth = ""
			configFile.Configs[k] = authConfig
			authConfig.ServerAddress = k
		}
	}
	return &configFile, nil
}
