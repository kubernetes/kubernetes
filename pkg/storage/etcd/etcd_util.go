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

package etcd

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os/exec"

	"k8s.io/kubernetes/pkg/tools"

	goetcd "github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
)

// IsEtcdNotFound returns true iff err is an etcd not found error.
func IsEtcdNotFound(err error) bool {
	return isEtcdErrorNum(err, tools.EtcdErrorCodeNotFound)
}

// IsEtcdNodeExist returns true iff err is an etcd node already exist error.
func IsEtcdNodeExist(err error) bool {
	return isEtcdErrorNum(err, tools.EtcdErrorCodeNodeExist)
}

// IsEtcdTestFailed returns true iff err is an etcd write conflict.
func IsEtcdTestFailed(err error) bool {
	return isEtcdErrorNum(err, tools.EtcdErrorCodeTestFailed)
}

// IsEtcdWatchStoppedByUser returns true iff err is a client triggered stop.
func IsEtcdWatchStoppedByUser(err error) bool {
	return goetcd.ErrWatchStoppedByUser == err
}

// isEtcdErrorNum returns true iff err is an etcd error, whose errorCode matches errorCode
func isEtcdErrorNum(err error, errorCode int) bool {
	etcdError, ok := err.(*goetcd.EtcdError)
	return ok && etcdError != nil && etcdError.ErrorCode == errorCode
}

// etcdErrorIndex returns the index associated with the error message and whether the
// index was available.
func etcdErrorIndex(err error) (uint64, bool) {
	if etcdError, ok := err.(*goetcd.EtcdError); ok {
		return etcdError.Index, true
	}
	return 0, false
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

func startEtcd() (*exec.Cmd, error) {
	cmd := exec.Command("etcd")
	err := cmd.Start()
	if err != nil {
		return nil, err
	}
	return cmd, nil
}

func NewEtcdClientStartServerIfNecessary(server string) (tools.EtcdClient, error) {
	_, err := GetEtcdVersion(server)
	if err != nil {
		glog.Infof("Failed to find etcd, attempting to start.")
		_, err := startEtcd()
		if err != nil {
			return nil, err
		}
	}

	servers := []string{server}
	return goetcd.NewClient(servers), nil
}

type etcdHealth struct {
	// Note this has to be public so the json library can modify it.
	Health string `json:health`
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
