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
	"io/ioutil"
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

// NewAuthConfigurationsFromFile returns AuthConfigurations from a path containing JSON
// in the same format as the .dockercfg file.
func NewAuthConfigurationsFromFile(path string) (*AuthConfigurations, error) {
	r, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return NewAuthConfigurations(r)
}

func cfgPaths(dockerConfigEnv string, homeEnv string) []string {
	var paths []string
	if dockerConfigEnv != "" {
		paths = append(paths, path.Join(dockerConfigEnv, "config.json"))
	}
	if homeEnv != "" {
		paths = append(paths, path.Join(homeEnv, ".docker", "config.json"))
		paths = append(paths, path.Join(homeEnv, ".dockercfg"))
	}
	return paths
}

// NewAuthConfigurationsFromDockerCfg returns AuthConfigurations from
// system config files. The following files are checked in the order listed:
// - $DOCKER_CONFIG/config.json if DOCKER_CONFIG set in the environment,
// - $HOME/.docker/config.json
// - $HOME/.dockercfg
func NewAuthConfigurationsFromDockerCfg() (*AuthConfigurations, error) {
	err := fmt.Errorf("No docker configuration found")
	var auths *AuthConfigurations

	pathsToTry := cfgPaths(os.Getenv("DOCKER_CONFIG"), os.Getenv("HOME"))
	for _, path := range pathsToTry {
		auths, err = NewAuthConfigurationsFromFile(path)
		if err == nil {
			return auths, nil
		}
	}
	return auths, err
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

	confsWrapper := struct {
		Auths map[string]dockerConfig `json:"auths"`
	}{}
	if err := json.Unmarshal(byteData, &confsWrapper); err == nil {
		if len(confsWrapper.Auths) > 0 {
			return confsWrapper.Auths, nil
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
		userpass := strings.SplitN(string(data), ":", 2)
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

// AuthStatus returns the authentication status for Docker API versions >= 1.23.
type AuthStatus struct {
	Status        string `json:"Status,omitempty" yaml:"Status,omitempty"`
	IdentityToken string `json:"IdentityToken,omitempty" yaml:"IdentityToken,omitempty"`
}

// AuthCheck validates the given credentials. It returns nil if successful.
//
// For Docker API versions >= 1.23, the AuthStatus struct will be populated, otherwise it will be empty.`
//
// See https://goo.gl/6nsZkH for more details.
func (c *Client) AuthCheck(conf *AuthConfiguration) (AuthStatus, error) {
	var authStatus AuthStatus
	if conf == nil {
		return authStatus, fmt.Errorf("conf is nil")
	}
	resp, err := c.do("POST", "/auth", doOptions{data: conf})
	if err != nil {
		return authStatus, err
	}
	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return authStatus, err
	}
	if len(data) == 0 {
		return authStatus, nil
	}
	if err := json.Unmarshal(data, &authStatus); err != nil {
		return authStatus, err
	}
	return authStatus, nil
}
