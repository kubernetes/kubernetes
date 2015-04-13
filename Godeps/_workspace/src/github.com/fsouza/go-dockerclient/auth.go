// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"os"
	"path"
	"strings"
)

// AuthConfiguration represents authentication options to use in the PushImage
// method. It represents the authentication in the Docker index server.
type AuthConfiguration struct {
	Username      string `json:"username,omitempty"`
	Password      string `json:"password,omitempty"`
	Email         string `json:"email,omitempty"`
	ServerAddress string `json:"serveraddress,omitempty"`
}

// AuthConfigurations represents authentication options to use for the
// PushImage method accommodating the new X-Registry-Config header
type AuthConfigurations struct {
	Configs map[string]AuthConfiguration `json:"configs"`
}

// dockerConfig represents a registry authentation configuration from the
// .dockercfg file.
type dockerConfig struct {
	Auth  string `json:"auth"`
	Email string `json:"email"`
}

// NewAuthConfigurationsFromDockerCfg returns AuthConfigurations from the
// ~/.dockercfg file.
func NewAuthConfigurationsFromDockerCfg() (*AuthConfigurations, error) {
	p := path.Join(os.Getenv("HOME"), ".dockercfg")
	r, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	return NewAuthConfigurations(r)
}

// NewAuthConfigurations returns AuthConfigurations from a JSON encoded string in the
// same format as the .dockercfg file.
func NewAuthConfigurations(r io.Reader) (*AuthConfigurations, error) {
	var auth *AuthConfigurations
	var confs map[string]dockerConfig
	if err := json.NewDecoder(r).Decode(&confs); err != nil {
		return nil, err
	}
	auth, err := authConfigs(confs)
	if err != nil {
		return nil, err
	}
	return auth, nil
}

// authConfigs converts a dockerConfigs map to a AuthConfigurations object.
func authConfigs(confs map[string]dockerConfig) (*AuthConfigurations, error) {
	c := &AuthConfigurations{
		Configs: make(map[string]AuthConfiguration),
	}
	for reg, conf := range confs {
		data, err := base64.StdEncoding.DecodeString(conf.Auth)
		if err != nil {
			return nil, err
		}
		userpass := strings.Split(string(data), ":")
		c.Configs[reg] = AuthConfiguration{
			Email:         conf.Email,
			Username:      userpass[0],
			Password:      userpass[1],
			ServerAddress: reg,
		}
	}
	return c, nil
}
