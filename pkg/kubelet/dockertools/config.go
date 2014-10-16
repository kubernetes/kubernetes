/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package dockertools

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

const (
	dockerConfigFileLocation = ".dockercfg"
)

func readDockerConfigFile() (cfg dockerConfig, err error) {
	absDockerConfigFileLocation, err := filepath.Abs(dockerConfigFileLocation)
	if err != nil {
		glog.Errorf("while trying to canonicalize %s: %v", dockerConfigFileLocation, err)
	}
	absDockerConfigFileLocation, err = filepath.Abs(dockerConfigFileLocation)
	glog.V(2).Infof("looking for .dockercfg at %s", absDockerConfigFileLocation)
	contents, err := ioutil.ReadFile(absDockerConfigFileLocation)
	if err != nil {
		glog.Errorf("while trying to read %s: %v", absDockerConfigFileLocation, err)
		return nil, err
	}
	if err = json.Unmarshal(contents, &cfg); err != nil {
		glog.Errorf("while trying to parse %s: %v", absDockerConfigFileLocation, err)
		return nil, err
	}
	return
}

// dockerConfig represents the config file used by the docker CLI.
// This config that represents the credentials that should be used
// when pulling images from specific image repositories.
type dockerConfig map[string]dockerConfigEntry

func (dc dockerConfig) addToKeyring(dk *dockerKeyring) {
	for loc, ident := range dc {
		creds := docker.AuthConfiguration{
			Username: ident.Username,
			Password: ident.Password,
			Email:    ident.Email,
		}

		parsed, err := url.Parse(loc)
		if err != nil {
			glog.Errorf("Entry %q in dockercfg invalid (%v), ignoring", loc, err)
			continue
		}

		dk.add(parsed.Host+parsed.Path, creds)
	}
}

type dockerConfigEntry struct {
	Username string
	Password string
	Email    string
}

// dockerConfigEntryWithAuth is used solely for deserializing the Auth field
// into a dockerConfigEntry during JSON deserialization.
type dockerConfigEntryWithAuth struct {
	Username string
	Password string
	Email    string
	Auth     string
}

func (ident *dockerConfigEntry) UnmarshalJSON(data []byte) error {
	var tmp dockerConfigEntryWithAuth
	err := json.Unmarshal(data, &tmp)
	if err != nil {
		return err
	}

	ident.Username = tmp.Username
	ident.Password = tmp.Password
	ident.Email = tmp.Email

	if len(tmp.Auth) == 0 {
		return nil
	}

	ident.Username, ident.Password, err = decodeDockerConfigFieldAuth(tmp.Auth)
	return err
}

// decodeDockerConfigFieldAuth deserializes the "auth" field from dockercfg into a
// username and a password. The format of the auth field is base64(<username>:<password>).
func decodeDockerConfigFieldAuth(field string) (username, password string, err error) {
	decoded, err := base64.StdEncoding.DecodeString(field)
	if err != nil {
		return
	}

	parts := strings.SplitN(string(decoded), ":", 2)
	if len(parts) != 2 {
		err = fmt.Errorf("unable to parse auth field")
		return
	}

	username = parts[0]
	password = parts[1]

	return
}
