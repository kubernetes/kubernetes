/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package credentialprovider

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/golang/glog"
)

// DockerConfig represents the config file used by the docker CLI.
// This config that represents the credentials that should be used
// when pulling images from specific image repositories.
type DockerConfig map[string]DockerConfigEntry

type DockerConfigEntry struct {
	Username string
	Password string
	Email    string
}

var (
	preferredPathLock sync.Mutex
	preferredPath     = ""
	workingDirPath    = ""
	homeDirPath       = os.Getenv("HOME")
	rootDirPath       = "/"

	configFileName = ".dockercfg"
)

func SetPreferredDockercfgPath(path string) {
	preferredPathLock.Lock()
	defer preferredPathLock.Unlock()
	preferredPath = path
}

func GetPreferredDockercfgPath() string {
	preferredPathLock.Lock()
	defer preferredPathLock.Unlock()
	return preferredPath
}

func ReadDockerConfigFile() (cfg DockerConfig, err error) {
	dockerConfigFileLocations := []string{GetPreferredDockercfgPath(), workingDirPath, homeDirPath, rootDirPath}
	for _, configPath := range dockerConfigFileLocations {
		absDockerConfigFileLocation, err := filepath.Abs(filepath.Join(configPath, configFileName))
		if err != nil {
			glog.Errorf("while trying to canonicalize %s: %v", configPath, err)
			continue
		}
		glog.V(4).Infof("looking for .dockercfg at %s", absDockerConfigFileLocation)
		contents, err := ioutil.ReadFile(absDockerConfigFileLocation)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			glog.V(4).Infof("while trying to read %s: %v", absDockerConfigFileLocation, err)
			continue
		}
		cfg, err := readDockerConfigFileFromBytes(contents)
		if err == nil {
			glog.V(4).Infof("found .dockercfg at %s", absDockerConfigFileLocation)
			return cfg, nil
		}
	}
	return nil, fmt.Errorf("couldn't find valid .dockercfg after checking in %v", dockerConfigFileLocations)
}

// HttpError wraps a non-StatusOK error code as an error.
type HttpError struct {
	StatusCode int
	Url        string
}

// Error implements error
func (he *HttpError) Error() string {
	return fmt.Sprintf("http status code: %d while fetching url %s",
		he.StatusCode, he.Url)
}

func ReadUrl(url string, client *http.Client, header *http.Header) (body []byte, err error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	if header != nil {
		req.Header = *header
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		glog.V(2).Infof("body of failing http response: %v", resp.Body)
		return nil, &HttpError{
			StatusCode: resp.StatusCode,
			Url:        url,
		}
	}

	contents, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return contents, nil
}

func ReadDockerConfigFileFromUrl(url string, client *http.Client, header *http.Header) (cfg DockerConfig, err error) {
	if contents, err := ReadUrl(url, client, header); err != nil {
		return nil, err
	} else {
		return readDockerConfigFileFromBytes(contents)
	}
}

func readDockerConfigFileFromBytes(contents []byte) (cfg DockerConfig, err error) {
	if err = json.Unmarshal(contents, &cfg); err != nil {
		glog.Errorf("while trying to parse blob %q: %v", contents, err)
		return nil, err
	}
	return
}

// dockerConfigEntryWithAuth is used solely for deserializing the Auth field
// into a dockerConfigEntry during JSON deserialization.
type dockerConfigEntryWithAuth struct {
	Username string `json:"username,omitempty"`
	Password string `json:"password,omitempty"`
	Email    string `json:"email,omitempty"`
	Auth     string `json:"auth,omitempty"`
}

func (ident *DockerConfigEntry) UnmarshalJSON(data []byte) error {
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

func (ident DockerConfigEntry) MarshalJSON() ([]byte, error) {
	toEncode := dockerConfigEntryWithAuth{ident.Username, ident.Password, ident.Email, ""}
	toEncode.Auth = encodeDockerConfigFieldAuth(ident.Username, ident.Password)

	return json.Marshal(toEncode)
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

func encodeDockerConfigFieldAuth(username, password string) string {
	fieldValue := username + ":" + password

	return base64.StdEncoding.EncodeToString([]byte(fieldValue))
}
