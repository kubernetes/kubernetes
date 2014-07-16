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
	"net/http"

	"github.com/golang/glog"
)

// Status is a enum type which describes a health status of a container.
type Status int

// These are the valid values of type Status.
const (
	Healthy Status = iota
	Unhealthy
	Unknown
)

// HTTPGetInterface is an abstract interface for testability. It abstracts the interface of http.Client.Get.
type HTTPGetInterface interface {
	Get(url string) (*http.Response, error)
}

// Check checks if GET request to the url succeeds.
// If the HTTP response code is successful, it returns Healthy.
// If the HTTP response code is unsuccessful, it returns Unhealthy.
// And it return Unknown and err if the HTTP communication itself fails.
func Check(url string, client HTTPGetInterface) (Status, error) {
	res, err := client.Get(url)
	if res.Body != nil {
		defer res.Body.Close()
	}
	if err != nil {
		return Unknown, err
	}
	if res.StatusCode >= http.StatusOK && res.StatusCode < http.StatusBadRequest {
		return Healthy, nil
	}
	glog.V(1).Infof("Health check failed for %s, Response: %v", url, *res)
	return Unhealthy, nil
}
