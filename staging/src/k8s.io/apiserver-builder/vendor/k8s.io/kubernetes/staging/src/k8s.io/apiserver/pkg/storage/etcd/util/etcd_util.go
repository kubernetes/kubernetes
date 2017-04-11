/*
Copyright 2015 The Kubernetes Authors.

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

package util

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"

	etcd "github.com/coreos/etcd/client"
)

// IsEtcdNotFound returns true if and only if err is an etcd not found error.
func IsEtcdNotFound(err error) bool {
	return isEtcdErrorNum(err, etcd.ErrorCodeKeyNotFound)
}

// IsEtcdNodeExist returns true if and only if err is an etcd node already exist error.
func IsEtcdNodeExist(err error) bool {
	return isEtcdErrorNum(err, etcd.ErrorCodeNodeExist)
}

// IsEtcdTestFailed returns true if and only if err is an etcd write conflict.
func IsEtcdTestFailed(err error) bool {
	return isEtcdErrorNum(err, etcd.ErrorCodeTestFailed)
}

// IsEtcdWatchExpired returns true if and only if err indicates the watch has expired.
func IsEtcdWatchExpired(err error) bool {
	// NOTE: This seems weird why it wouldn't be etcd.ErrorCodeWatcherCleared
	//       I'm using the previous matching value
	return isEtcdErrorNum(err, etcd.ErrorCodeEventIndexCleared)
}

// IsEtcdUnreachable returns true if and only if err indicates the server could not be reached.
func IsEtcdUnreachable(err error) bool {
	// NOTE: The logic has changed previous error code no longer applies
	return err == etcd.ErrClusterUnavailable
}

// isEtcdErrorNum returns true if and only if err is an etcd error, whose errorCode matches errorCode
func isEtcdErrorNum(err error, errorCode int) bool {
	if err != nil {
		if etcdError, ok := err.(etcd.Error); ok {
			return etcdError.Code == errorCode
		}
		// NOTE: There are other error types returned
	}
	return false
}

// GetEtcdVersion performs a version check against the provided Etcd server,
// returning the string response, and error (if any).
func GetEtcdVersion(host string) (string, error) {
	response, err := http.Get(host + "/version")
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unsuccessful response from etcd server %q: %v", host, err)
	}
	versionBytes, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return "", err
	}
	return string(versionBytes), nil
}

type etcdHealth struct {
	// Note this has to be public so the json library can modify it.
	Health string `json:"health"`
}

func EtcdHealthCheck(data []byte) error {
	obj := etcdHealth{}
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	if obj.Health != "true" {
		return fmt.Errorf("Unhealthy status: %s", obj.Health)
	}
	return nil
}
