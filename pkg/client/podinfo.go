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

package client

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"reflect"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// ErrPodInfoNotAvailable may be returned when the requested pod info is not available.
var ErrPodInfoNotAvailable = errors.New("no pod info available")

// PodInfoGetter is an interface for things that can get information about a pod's containers.
// Injectable for easy testing.
type PodInfoGetter interface {
	// GetPodInfo returns information about all containers which are part
	// Returns an api.PodInfo, or an error if one occurs.
	GetPodInfo(host, podID string) (api.PodInfo, error)
	// WatchPodInfo watches for changes in pod information.
	WatchPodInfo(host, podID string) (watch.Interface, error)
}

// HTTPPodInfoGetter is the default implementation of PodInfoGetter, accesses the kubelet over HTTP.
type HTTPPodInfoGetter struct {
	Client *http.Client
	Port   uint
}

func (c *HTTPPodInfoGetter) WatchPodInfo(host, podID string) (watch.Interface, error) {
	return newHTTPPodInfoWatch(c, host, podID)
}

// GetPodInfo gets information about the specified pod.
func (c *HTTPPodInfoGetter) GetPodInfo(host, podID string) (api.PodInfo, error) {
	request, err := http.NewRequest(
		"GET",
		fmt.Sprintf(
			"http://%s/podInfo?podID=%s",
			net.JoinHostPort(host, strconv.FormatUint(uint64(c.Port), 10)),
			podID),
		nil)
	if err != nil {
		return nil, err
	}
	response, err := c.Client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNotFound {
		return nil, ErrPodInfoNotAvailable
	}
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	// Check that this data can be unmarshalled
	info := api.PodInfo{}
	err = json.Unmarshal(body, &info)
	if err != nil {
		return nil, err
	}
	return info, nil
}

// FakePodInfoGetter is a fake implementation of PodInfoGetter. It is useful for testing.
type FakePodInfoGetter struct {
	data api.PodInfo
	err  error
}

// GetPodInfo is a fake implementation of PodInfoGetter.GetPodInfo.
func (c *FakePodInfoGetter) GetPodInfo(host, podID string) (api.PodInfo, error) {
	return c.data, c.err
}

type empty struct{}

type httpPodInfoWatch struct {
	getter PodInfoGetter
	obj    api.PodInfo
	host   string
	podID  string
	events chan watch.Event
	stop   bool
}

func newHTTPPodInfoWatch(getter PodInfoGetter, host, podID string) (watch.Interface, error) {
	obj, err := getter.GetPodInfo(host, podID)
	if err != nil {
		return nil, err
	}
	res := &httpPodInfoWatch{
		getter: getter,
		obj:    obj,
		host:   host,
		podID:  podID,
		events: make(chan watch.Event),
		stop:   false,
	}
	go func() {
		wait.Poll(time.Second*10, time.Second*10, res.pollFunc)
	}()
	return res, nil
}

func (h *httpPodInfoWatch) pollFunc() (bool, error) {
	if h.stop {
		return true, nil
	}
	newObj, err := h.getter.GetPodInfo(h.host, h.podID)
	if err == nil && !reflect.DeepEqual(h.obj, newObj) {
		h.events <- watch.Event{
			Type:   watch.Modified,
			Object: newObj,
		}
		h.obj = newObj
	}
	return false, nil
}

func (h *httpPodInfoWatch) ResultChan() <-chan watch.Event {
	return h.events
}

func (h *httpPodInfoWatch) Stop() {
	close(h.events)
	h.stop = true
}
