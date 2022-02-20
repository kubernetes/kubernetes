/*
Copyright 2021 The Kubernetes Authors.

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

package gcpcredential

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"

	"k8s.io/cloud-provider/credentialconfig"
	"k8s.io/klog/v2"
)

const (
	maxReadLength = 10 * 1 << 20 // 10MB
)

// HTTPError wraps a non-StatusOK error code as an error.
type HTTPError struct {
	StatusCode int
	URL        string
}

// Error implements error
func (he *HTTPError) Error() string {
	return fmt.Sprintf("http status code: %d while fetching url %s",
		he.StatusCode, he.URL)
}

// ReadURL read contents from given url
func ReadURL(url string, client *http.Client, header *http.Header) (body []byte, err error) {
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
		klog.V(2).InfoS("Failed to read URL", "statusCode", resp.StatusCode, "URL", url)
		return nil, &HTTPError{
			StatusCode: resp.StatusCode,
			URL:        url,
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

// ReadDockerConfigFileFromURL read a docker config file from the given url
func ReadDockerConfigFileFromURL(url string, client *http.Client, header *http.Header) (cfg credentialconfig.RegistryConfig, err error) {
	if contents, err := ReadURL(url, client, header); err == nil {
		return ReadDockerConfigFileFromBytes(contents)
	}

	return nil, err
}

type internalRegistryConfig map[string]RegistryConfigEntry

// ReadDockerConfigFileFromBytes read a docker config file from the given bytes
func ReadDockerConfigFileFromBytes(contents []byte) (cfg credentialconfig.RegistryConfig, err error) {
	serializableCfg := internalRegistryConfig{}
	if err = json.Unmarshal(contents, &serializableCfg); err != nil {
		return nil, errors.New("error occurred while trying to unmarshal json")
	}
	return convertToExternalConfig(serializableCfg), nil
}

func convertToExternalConfig(in internalRegistryConfig) (cfg credentialconfig.RegistryConfig) {
	configMap := credentialconfig.RegistryConfig{}
	for k, v := range in {
		configMap[k] = credentialconfig.RegistryConfigEntry{
			Username: v.Username,
			Password: v.Password,
			Email:    v.Email,
		}
	}
	return configMap
}
