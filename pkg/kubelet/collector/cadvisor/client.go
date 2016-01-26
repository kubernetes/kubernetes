// +build cgo,linux

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package cadvisor

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"

	"github.com/google/cadvisor/events"
	"github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"

	"github.com/golang/glog"
)

type ClientInterface interface {
	Start() error
	GetMachineInfo() (*v1.MachineInfo, error)
	GetVersionInfo() (*v1.VersionInfo, error)
	GetFsInfo(label string) ([]v2.FsInfo, error)
	GetContainerInfo(containerName string, query *v1.ContainerInfoRequest) (*v1.ContainerInfo, error)
	SubcontainersInfo(containerName string, query *v1.ContainerInfoRequest) ([]*v1.ContainerInfo, error)
	DockerContainer(dockerName string, query *v1.ContainerInfoRequest) (v1.ContainerInfo, error)
	WatchForEvents(request *events.Request) (*events.EventChannel, error)
}

// This is taken from github.com/google/cadvisor/client/client.go as cadvisor
// client is supported only by the latest cadvisor, which is not currently used
// by kubernetes HEAD. Most of this file should go away when that is merged

// Client represents the base URL for a cAdvisor client
type Client struct {
	baseUrl string
}

var _ ClientInterface = new(Client)

// NewClient returns a new client with the specified base URL
func NewClient(url string) (*Client, error) {
	if !strings.HasSuffix(url, "/") {
		url += "/"
	}

	return &Client{
		baseUrl: fmt.Sprintf("%sapi/", url),
	}, nil
}

func (self *Client) Start() error {
	return nil
}

// MachineInfo returns the JSON machine information for this client.
// A non-nil error result indicates a problem with obtaining
// the JSON machine information data.
func (self *Client) GetMachineInfo() (minfo *v1.MachineInfo, err error) {
	u := self.machineInfoUrl()
	ret := new(v1.MachineInfo)
	if err = self.httpGetJsonData(ret, nil, u, "machine info"); err != nil {
		return
	}
	minfo = ret
	return
}

func (self *Client) GetVersionInfo() (v *v1.VersionInfo, err error) {
	u := self.versionInfoUrl()
	ret := new(v2.Attributes)
	if err = self.httpGetJsonData(ret, nil, u, "version info"); err != nil {
		return
	}
	v = &v1.VersionInfo{
		KernelVersion:      ret.KernelVersion,
		ContainerOsVersion: ret.ContainerOsVersion,
		DockerVersion:      ret.DockerVersion,
		CadvisorVersion:    ret.CadvisorVersion,
	}
	return
}

func (self *Client) GetFsInfo(label string) (fs []v2.FsInfo, err error) {
	u := self.fsInfoUrl(label)
	ret := new([]v2.FsInfo)
	if err = self.httpGetJsonData(ret, nil, u, fmt.Sprintf("fs info for %q", label)); err != nil {
		return
	}
	fs = *ret
	return
}

// ContainerInfo returns the JSON container information for the specified
// container and request.
func (self *Client) GetContainerInfo(name string, query *v1.ContainerInfoRequest) (cinfo *v1.ContainerInfo, err error) {
	u := self.containerInfoUrl(name)
	ret := new(v1.ContainerInfo)
	if err = self.httpGetJsonData(ret, query, u, fmt.Sprintf("container info for %q", name)); err != nil {
		return
	}
	cinfo = ret
	return
}

// Returns the information about all subcontainers (recursive) of the specified container (including itself).
func (self *Client) SubcontainersInfo(name string, query *v1.ContainerInfoRequest) ([]*v1.ContainerInfo, error) {
	var response []*v1.ContainerInfo
	url := self.subcontainersInfoUrl(name)
	err := self.httpGetJsonData(&response, query, url, fmt.Sprintf("subcontainers container info for %q", name))
	if err != nil {
		return []*v1.ContainerInfo{}, err

	}
	return response, nil
}

// Returns the JSON container information for the specified
// Docker container and request.
func (self *Client) DockerContainer(name string, query *v1.ContainerInfoRequest) (cinfo v1.ContainerInfo, err error) {
	u := self.dockerInfoUrl(name)
	ret := make(map[string]v1.ContainerInfo)
	if err = self.httpGetJsonData(&ret, query, u, fmt.Sprintf("Docker container info for %q", name)); err != nil {
		return
	}
	if len(ret) != 1 {
		err = fmt.Errorf("expected to only receive 1 Docker container: %+v", ret)
		return
	}
	for _, cont := range ret {
		cinfo = cont
	}
	return
}

func (self *Client) WatchForEvents(request *events.Request) (ec *events.EventChannel, err error) {
	u, err := self.createEventsUrl(request)
	if err != nil {
		return
	}
	// Todo(huang195): is there a better way to set watchID?
	ec = events.NewEventChannel(100)
	go func() {
		err = self.eventStreamingInfo(u, ec.GetChannel())
		if err != nil {
			glog.Errorf("got error retrieving event info: %v", err)
			return
		}
	}()
	return
}

func (self *Client) createEventsUrl(request *events.Request) (string, error) {
	params := url.Values{}
	params.Set("stream", "true")

	for k := range request.EventType {
		switch k {
		case v1.EventOom:
			params.Set("oom_events", "true")
		case v1.EventOomKill:
			params.Set("oom_kill_events", "true")
		case v1.EventContainerCreation:
			params.Set("creation_events", "true")
		case v1.EventContainerDeletion:
			params.Set("deletion_events", "true")
		default:
			return "", fmt.Errorf("Request EventType (%v) not recognized", k)
		}
	}

	if request.MaxEventsReturned > 0 {
		params.Set("max_events", strconv.Itoa(request.MaxEventsReturned))
	}

	if request.IncludeSubcontainers {
		params.Set("subcontainers", "true")
	}

	return "?" + params.Encode(), nil
}

// Streams all events that occur that satisfy the request into the channel
// that is passed
func (self *Client) eventStreamingInfo(name string, einfo chan *v1.Event) (err error) {
	u := self.eventsInfoUrl(name)
	if err = self.getEventStreamingData(u, einfo); err != nil {
		return
	}
	return nil
}

func (self *Client) machineInfoUrl() string {
	return self.baseUrl + "v1.3/machine"
}

func (self *Client) versionInfoUrl() string {
	return self.baseUrl + "v2.0/attributes"
}

func (self *Client) fsInfoUrl(label string) string {
	return self.baseUrl + "v2.0/" + fmt.Sprintf("storage?label=%s", label)
}

func (self *Client) containerInfoUrl(name string) string {
	return self.baseUrl + "v1.3/" + path.Join("containers", name)
}

func (self *Client) subcontainersInfoUrl(name string) string {
	return self.baseUrl + "v1.3/" + path.Join("subcontainers", name)
}

func (self *Client) dockerInfoUrl(name string) string {
	return self.baseUrl + "v1.3/" + path.Join("docker", name)
}

func (self *Client) eventsInfoUrl(name string) string {
	return self.baseUrl + "v1.3/" + path.Join("events", name)
}

func (self *Client) httpGetJsonData(data, postData interface{}, url, infoName string) error {
	var resp *http.Response
	var err error

	if postData != nil {
		data, marshalErr := json.Marshal(postData)
		if marshalErr != nil {
			return fmt.Errorf("unable to marshal data: %v", marshalErr)
		}
		resp, err = http.Post(url, "application/json", bytes.NewBuffer(data))
	} else {
		resp, err = http.Get(url)
	}
	if err != nil {
		return fmt.Errorf("unable to get %q from %q: %v", infoName, url, err)
	}
	if resp == nil {
		return fmt.Errorf("received empty response for %q from %q", infoName, url)
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		err = fmt.Errorf("unable to read all %q from %q: %v", infoName, url, err)
		return err
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("request %q failed with error: %q", url, strings.TrimSpace(string(body)))
	}
	if err = json.Unmarshal(body, data); err != nil {
		err = fmt.Errorf("unable to unmarshal %q (Body: %q) from %q with error: %v", infoName, string(body), url, err)
		return err
	}
	return nil
}

func (self *Client) getEventStreamingData(url string, einfo chan *v1.Event) error {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Status code is not OK: %v (%s)", resp.StatusCode, resp.Status)
	}

	dec := json.NewDecoder(resp.Body)
	var m *v1.Event = &v1.Event{}
	for {
		err := dec.Decode(m)
		if err != nil {
			if err == io.EOF {
				break
			}
			// if called without &stream=true will not be able to parse event and will trigger fatal
			glog.Fatalf("Received error %v", err)
		}
		einfo <- m
	}
	return nil
}
