// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"strings"
)

// ErrCannotParseDockercfg is the error returned by NewAuthConfigurations when the dockercfg cannot be parsed.
var ErrCannotParseDockercfg = errors.New("Failed to read authentication from dockercfg")

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

// AuthConfigurations119 is used to serialize a set of AuthConfigurations
// for Docker API >= 1.19.
type AuthConfigurations119 map[string]AuthConfiguration

// dockerConfig represents a registry authentation configuration from the
// .dockercfg file.
type dockerConfig struct {
	Auth  string `json:"auth"`
	Email string `json:"email"`
}

// NewAuthConfigurationsFromDockerCfg returns AuthConfigurations from the
// ~/.dockercfg file.
func NewAuthConfigurationsFromDockerCfg() (*AuthConfigurations, error) {
	var r io.Reader
	var err error
	p := path.Join(os.Getenv("HOME"), ".docker", "config.json")
	r, err = os.Open(p)
	if err != nil {
		p := path.Join(os.Getenv("HOME"), ".dockercfg")
		r, err = os.Open(p)
		if err != nil {
			return nil, err
		}
	}
	return NewAuthConfigurations(r)
}

// NewAuthConfigurations returns AuthConfigurations from a JSON encoded string in the
// same format as the .dockercfg file.
func NewAuthConfigurations(r io.Reader) (*AuthConfigurations, error) {
	var auth *AuthConfigurations
	confs, err := parseDockerConfig(r)
	if err != nil {
		return nil, err
	}
	auth, err = authConfigs(confs)
	if err != nil {
		return nil, err
	}
	return auth, nil
}

func parseDockerConfig(r io.Reader) (map[string]dockerConfig, error) {
	buf := new(bytes.Buffer)
	buf.ReadFrom(r)
	byteData := buf.Bytes()

	var confsWrapper map[string]map[string]dockerConfig
	if err := json.Unmarshal(byteData, &confsWrapper); err == nil {
		if confs, ok := confsWrapper["auths"]; ok {
			return confs, nil
		}
	}

	var confs map[string]dockerConfig
	if err := json.Unmarshal(byteData, &confs); err != nil {
		return nil, err
	}
	return confs, nil
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
		if len(userpass) != 2 {
			return nil, ErrCannotParseDockercfg
		}
		c.Configs[reg] = AuthConfiguration{
			Email:         conf.Email,
			Username:      userpass[0],
			Password:      userpass[1],
			ServerAddress: reg,
		}
	}
	return c, nil
}

// AuthCheck validates the given credentials. It returns nil if successful.
//
// See https://goo.gl/m2SleN for more details.
func (c *Client) AuthCheck(conf *AuthConfiguration) error {
	if conf == nil {
		return fmt.Errorf("conf is nil")
	}
	body, statusCode, err := c.do("POST", "/auth", doOptions{data: conf})
	if err != nil {
		return err
	}
	if statusCode > 400 {
		return fmt.Errorf("auth error (%d): %s", statusCode, body)
	}
	return nil
}
