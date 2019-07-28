/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"k8s.io/klog"
)

const (
	maxReadLength = 10 * 1 << 20 // 10MB
)

// DockerConfigJson represents ~/.docker/config.json file info
// see https://github.com/docker/docker/pull/12009
type DockerConfigJson struct {
	Auths DockerConfig `json:"auths"`
	// +optional
	HttpHeaders map[string]string `json:"HttpHeaders,omitempty"`
}

// DockerConfig represents the config file used by the docker CLI.
// This config that represents the credentials that should be used
// when pulling images from specific image repositories.
type DockerConfig map[string]DockerConfigEntry

type DockerConfigEntry struct {
	Username string
	Password string
	Email    string
	Provider DockerConfigProvider
}

var (
	preferredPathLock sync.Mutex
	preferredPath     = ""
	workingDirPath    = ""
	homeDirPath, _    = os.UserHomeDir()
	rootDirPath       = "/"
	homeJsonDirPath   = filepath.Join(homeDirPath, ".docker")
	rootJsonDirPath   = filepath.Join(rootDirPath, ".docker")

	configFileName     = ".dockercfg"
	configJsonFileName = "config.json"
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

//DefaultDockercfgPaths returns default search paths of .dockercfg
func DefaultDockercfgPaths() []string {
	return []string{GetPreferredDockercfgPath(), workingDirPath, homeDirPath, rootDirPath}
}

//DefaultDockerConfigJSONPaths returns default search paths of .docker/config.json
func DefaultDockerConfigJSONPaths() []string {
	return []string{GetPreferredDockercfgPath(), workingDirPath, homeJsonDirPath, rootJsonDirPath}
}

// ReadDockercfgFile attempts to read a legacy dockercfg file from the given paths.
// if searchPaths is empty, the default paths are used.
func ReadDockercfgFile(searchPaths []string) (cfg DockerConfig, err error) {
	if len(searchPaths) == 0 {
		searchPaths = DefaultDockercfgPaths()
	}

	for _, configPath := range searchPaths {
		absDockerConfigFileLocation, err := filepath.Abs(filepath.Join(configPath, configFileName))
		if err != nil {
			klog.Errorf("while trying to canonicalize %s: %v", configPath, err)
			continue
		}
		klog.V(4).Infof("looking for .dockercfg at %s", absDockerConfigFileLocation)
		contents, err := ioutil.ReadFile(absDockerConfigFileLocation)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			klog.V(4).Infof("while trying to read %s: %v", absDockerConfigFileLocation, err)
			continue
		}
		cfg, err := readDockerConfigFileFromBytes(contents)
		if err == nil {
			klog.V(4).Infof("found .dockercfg at %s", absDockerConfigFileLocation)
			return cfg, nil
		}
	}
	return nil, fmt.Errorf("couldn't find valid .dockercfg after checking in %v", searchPaths)
}

// ReadDockerConfigJSONFile attempts to read a docker config.json file from the given paths.
// if searchPaths is empty, the default paths are used.
func ReadDockerConfigJSONFile(searchPaths []string) (cfg DockerConfig, err error) {
	if len(searchPaths) == 0 {
		searchPaths = DefaultDockerConfigJSONPaths()
	}
	for _, configPath := range searchPaths {
		absDockerConfigFileLocation, err := filepath.Abs(filepath.Join(configPath, configJsonFileName))
		if err != nil {
			klog.Errorf("while trying to canonicalize %s: %v", configPath, err)
			continue
		}
		klog.V(4).Infof("looking for %s at %s", configJsonFileName, absDockerConfigFileLocation)
		cfg, err = ReadSpecificDockerConfigJsonFile(absDockerConfigFileLocation)
		if err != nil {
			if !os.IsNotExist(err) {
				klog.V(4).Infof("while trying to read %s: %v", absDockerConfigFileLocation, err)
			}
			continue
		}
		klog.V(4).Infof("found valid %s at %s", configJsonFileName, absDockerConfigFileLocation)
		return cfg, nil
	}
	return nil, fmt.Errorf("couldn't find valid %s after checking in %v", configJsonFileName, searchPaths)

}

//ReadSpecificDockerConfigJsonFile attempts to read docker configJSON from a given file path.
func ReadSpecificDockerConfigJsonFile(filePath string) (cfg DockerConfig, err error) {
	var contents []byte

	if contents, err = ioutil.ReadFile(filePath); err != nil {
		return nil, err
	}
	return readDockerConfigJsonFileFromBytes(contents)
}

func ReadDockerConfigFile() (cfg DockerConfig, err error) {
	if cfg, err := ReadDockerConfigJSONFile(nil); err == nil {
		return cfg, nil
	}
	// Can't find latest config file so check for the old one
	return ReadDockercfgFile(nil)
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
		klog.V(2).Infof("body of failing http response: %v", resp.Body)
		return nil, &HttpError{
			StatusCode: resp.StatusCode,
			Url:        url,
		}
	}

	limitedReader := &io.LimitedReader{R: resp.Body, N: maxReadLength}
	contents, err := ioutil.ReadAll(limitedReader)
	if err != nil {
		return nil, err
	}

	if limitedReader.N <= 0 {
		return nil, errors.New("the read limit is reached")
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
		klog.Errorf("while trying to parse blob %q: %v", contents, err)
		return nil, err
	}
	return
}

func readDockerConfigJsonFileFromBytes(contents []byte) (cfg DockerConfig, err error) {
	var cfgJson DockerConfigJson
	if err = json.Unmarshal(contents, &cfgJson); err != nil {
		klog.Errorf("while trying to parse blob %q: %v", contents, err)
		return nil, err
	}
	cfg = cfgJson.Auths
	return
}

// dockerConfigEntryWithAuth is used solely for deserializing the Auth field
// into a dockerConfigEntry during JSON deserialization.
type dockerConfigEntryWithAuth struct {
	// +optional
	Username string `json:"username,omitempty"`
	// +optional
	Password string `json:"password,omitempty"`
	// +optional
	Email string `json:"email,omitempty"`
	// +optional
	Auth string `json:"auth,omitempty"`
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
