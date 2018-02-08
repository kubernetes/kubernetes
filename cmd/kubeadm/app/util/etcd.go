/*
Copyright 2017 The Kubernetes Authors.

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
	"context"
	"github.com/coreos/etcd/clientv3"
	"time"
)

// EtcdCluster is an interface to get etcd cluster related information
type EtcdCluster interface {
	GetEtcdClusterStatus() (*clientv3.StatusResponse, error)
}

// LocalEtcdCluster represents an instance of a local etcd cluster
type LocalEtcdCluster struct{}

// GetEtcdClusterStatus returns nil for status Up or error for status Down
func (cluster LocalEtcdCluster) GetEtcdClusterStatus() (*clientv3.StatusResponse, error) {
	ep := []string{"localhost:2379"}
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   ep,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return nil, err
	}
	defer cli.Close()

	resp, err := cli.Status(context.Background(), ep[0])
	if err != nil {
		return nil, err
	}

	return resp, nil
}
