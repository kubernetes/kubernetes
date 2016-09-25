// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Client library to programmatically access cAdvisor API.
package v2

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"

	v1 "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"
)

// Client represents the base URL for a cAdvisor client.
type Client struct {
	baseUrl string
}

// NewClient returns a new client with the specified base URL.
func NewClient(url string) (*Client, error) {
	if !strings.HasSuffix(url, "/") {
		url += "/"
	}

	return &Client{
		baseUrl: fmt.Sprintf("%sapi/v2.1/", url),
	}, nil
}

// MachineInfo returns the JSON machine information for this client.
// A non-nil error result indicates a problem with obtaining
// the JSON machine information data.
func (self *Client) MachineInfo() (minfo *v1.MachineInfo, err error) {
	u := self.machineInfoUrl()
	ret := new(v1.MachineInfo)
	if err = self.httpGetJsonData(ret, nil, u, "machine info"); err != nil {
		return
	}
	minfo = ret
	return
}

// MachineStats returns the JSON machine statistics for this client.
// A non-nil error result indicates a problem with obtaining
// the JSON machine information data.
func (self *Client) MachineStats() ([]v2.MachineStats, error) {
	var ret []v2.MachineStats
	u := self.machineStatsUrl()
	err := self.httpGetJsonData(&ret, nil, u, "machine stats")
	return ret, err
}

// VersionInfo returns the version info for cAdvisor.
func (self *Client) VersionInfo() (version string, err error) {
	u := self.versionInfoUrl()
	version, err = self.httpGetString(u, "version info")
	return
}

// Attributes returns hardware and software attributes of the machine.
func (self *Client) Attributes() (attr *v2.Attributes, err error) {
	u := self.attributesUrl()
	ret := new(v2.Attributes)
	if err = self.httpGetJsonData(ret, nil, u, "attributes"); err != nil {
		return
	}
	attr = ret
	return
}

// Stats returns stats for the requested container.
func (self *Client) Stats(name string, request *v2.RequestOptions) (map[string]v2.ContainerInfo, error) {
	u := self.statsUrl(name)
	ret := make(map[string]v2.ContainerInfo)
	data := url.Values{
		"type":      []string{request.IdType},
		"count":     []string{strconv.Itoa(request.Count)},
		"recursive": []string{strconv.FormatBool(request.Recursive)},
	}

	u = fmt.Sprintf("%s?%s", u, data.Encode())
	if err := self.httpGetJsonData(&ret, nil, u, "stats"); err != nil {
		return nil, err
	}
	return ret, nil
}

func (self *Client) machineInfoUrl() string {
	return self.baseUrl + path.Join("machine")
}

func (self *Client) machineStatsUrl() string {
	return self.baseUrl + path.Join("machinestats")
}

func (self *Client) versionInfoUrl() string {
	return self.baseUrl + path.Join("version")
}

func (self *Client) attributesUrl() string {
	return self.baseUrl + path.Join("attributes")
}

func (self *Client) statsUrl(name string) string {
	return self.baseUrl + path.Join("stats", name)
}

func (self *Client) httpGetResponse(postData interface{}, urlPath, infoName string) ([]byte, error) {
	var resp *http.Response
	var err error

	if postData != nil {
		data, marshalErr := json.Marshal(postData)
		if marshalErr != nil {
			return nil, fmt.Errorf("unable to marshal data: %v", marshalErr)
		}
		resp, err = http.Post(urlPath, "application/json", bytes.NewBuffer(data))
	} else {
		resp, err = http.Get(urlPath)
	}
	if err != nil {
		return nil, fmt.Errorf("unable to post %q to %q: %v", infoName, urlPath, err)
	}
	if resp == nil {
		return nil, fmt.Errorf("received empty response for %q from %q", infoName, urlPath)
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		err = fmt.Errorf("unable to read all %q from %q: %v", infoName, urlPath, err)
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("request %q failed with error: %q", urlPath, strings.TrimSpace(string(body)))
	}
	return body, nil
}

func (self *Client) httpGetString(url, infoName string) (string, error) {
	body, err := self.httpGetResponse(nil, url, infoName)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

func (self *Client) httpGetJsonData(data, postData interface{}, url, infoName string) error {
	body, err := self.httpGetResponse(postData, url, infoName)
	if err != nil {
		return err
	}
	if err = json.Unmarshal(body, data); err != nil {
		err = fmt.Errorf("unable to unmarshal %q (Body: %q) from %q with error: %v", infoName, string(body), url, err)
		return err
	}
	return nil
}
