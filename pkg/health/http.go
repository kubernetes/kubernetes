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

package health

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// HTTPGetInterface is an abstract interface for testability. It abstracts the interface of http.Client.Get.
// This is exported because some other packages may want to do direct HTTP checks.
type HTTPGetInterface interface {
	Get(url string) (*http.Response, error)
}

// HTTPHealthChecker is an implementation of HealthChecker which checks container health by sending HTTP Get requests.
type HTTPHealthChecker struct {
	client HTTPGetInterface
}

func NewHTTPHealthChecker(client *http.Client) HealthChecker {
	return &HTTPHealthChecker{client: &http.Client{}}
}

// getURLParts parses the components of the target URL.  For testability.
func getURLParts(currentState api.PodState, container api.Container) (string, int, string, error) {
	params := container.LivenessProbe.HTTPGet
	if params == nil {
		return "", -1, "", fmt.Errorf("no HTTP parameters specified: %v", container)
	}
	port := -1
	switch params.Port.Kind {
	case util.IntstrInt:
		port = params.Port.IntVal
	case util.IntstrString:
		port = findPortByName(container, params.Port.StrVal)
		if port == -1 {
			// Last ditch effort - maybe it was an int stored as string?
			var err error
			if port, err = strconv.Atoi(params.Port.StrVal); err != nil {
				return "", -1, "", err
			}
		}
	}
	if port == -1 {
		return "", -1, "", fmt.Errorf("unknown port: %v", params.Port)
	}
	var host string
	if len(params.Host) > 0 {
		host = params.Host
	} else {
		host = currentState.PodIP
	}

	return host, port, params.Path, nil
}

// formatURL formats a URL from args.  For testability.
func formatURL(host string, port int, path string) string {
	u := url.URL{
		Scheme: "http",
		Host:   net.JoinHostPort(host, strconv.Itoa(port)),
		Path:   path,
	}
	return u.String()
}

// DoHTTPCheck checks if a GET request to the url succeeds.
// If the HTTP response code is successful (i.e. 400 > code >= 200), it returns Healthy.
// If the HTTP response code is unsuccessful, it returns Unhealthy.
// It returns Unknown and err if the HTTP communication itself fails.
// This is exported because some other packages may want to do direct HTTP checks.
func DoHTTPCheck(url string, client HTTPGetInterface) (Status, error) {
	res, err := client.Get(url)
	if err != nil {
		return Unknown, err
	}
	defer res.Body.Close()
	if res.StatusCode >= http.StatusOK && res.StatusCode < http.StatusBadRequest {
		return Healthy, nil
	}
	glog.V(1).Infof("Health check failed for %s, Response: %v", url, *res)
	return Unhealthy, nil
}

// HealthCheck checks if the container is healthy by trying sending HTTP Get requests to the container.
func (h *HTTPHealthChecker) HealthCheck(podFullName, podUUID string, currentState api.PodState, container api.Container) (Status, error) {
	host, port, path, err := getURLParts(currentState, container)
	if err != nil {
		return Unknown, err
	}
	return DoHTTPCheck(formatURL(host, port, path), h.client)
}

func (h *HTTPHealthChecker) CanCheck(probe *api.LivenessProbe) bool {
	return probe.HTTPGet != nil
}
