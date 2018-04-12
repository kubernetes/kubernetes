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

package etcd

import (
	"context"
	"crypto/tls"
	"fmt"
	"log"
	"reflect"
	"sort"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/pkg/transport"
)

// EtcdClusterInterrogator is an interface to get etcd cluster related information
type EtcdClusterInterrogator interface {
	GetEtcdClusterStatus() (*EtcdStatus, error)
}

// EtcdCluster provides connection parameters for an etcd cluster
type EtcdCluster struct {
	Endpoints []string
	TLS       *tls.Config
}

// EtcdStatusChecker an interface for etcd status validation functions
type EtcdStatusChecker interface {
	VersionsMatch() bool
}

// EtcdStatus provides data about an etcd cluster
type EtcdStatus struct {
	Versions []string
	Members  []uint64
}

// VersionsMatch ensure that all etcd versions are identical
func (s EtcdStatus) VersionsMatch() bool {
	var version string
	for i, v := range s.Versions {
		if i == 0 {
			version = v
		} else if version != v {
			return false
		}
	}
	return true
}

// New define a new EtcdCluster
func New(endpoints []string, ca, cert, key string) (*EtcdCluster, error) {

	var tlsConfig *tls.Config
	var err error

	if ca != "" && cert != "" && key != "" {
		tlsInfo := transport.TLSInfo{
			CertFile:      cert,
			KeyFile:       key,
			TrustedCAFile: ca,
		}
		tlsConfig, err = tlsInfo.ClientConfig()
		if err != nil {
			return nil, err
		}
	}

	return &EtcdCluster{
		Endpoints: endpoints,
		TLS:       tlsConfig,
	}, nil
}

// GetEtcdClusterStatus connects to all etcd endpoints in EtcdCluster and
// provides data on the cluster. In event a member is unresponsive, fails with
// error inidicating so.
func (cluster EtcdCluster) GetEtcdClusterStatus() (*EtcdStatus, error) {

	var status EtcdStatus
	for _, ep := range cluster.Endpoints {
		cli, err := clientv3.New(clientv3.Config{
			Endpoints:   []string{ep},
			DialTimeout: 5 * time.Second,
			TLS:         cluster.TLS,
		})
		if err != nil {
			return nil, fmt.Errorf("Failed to connect to etcd at %s: %v", ep, err)
		}
		defer cli.Close()

		memberResp, err := cli.MemberList(context.Background())
		if err != nil {
			log.Fatal(err)
		}

		if len(status.Members) == 0 {
			status.Members = sortedMembers(memberResp.Members)
		} else {
			if !reflect.DeepEqual(status.Members, sortedMembers(memberResp.Members)) {
				return nil, fmt.Errorf("Cluster has mismatched members")
			}
		}

		statusResp, err := cli.Status(context.Background(), ep)
		if err != nil {
			return nil, err
		}
		status.Versions = append(status.Versions, statusResp.Version)
	}
	return &status, nil
}

type uint64slice []uint64

func (s uint64slice) Len() int {
	return len(s)
}

func (s uint64slice) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s uint64slice) Less(i, j int) bool {
	return s[i] < s[j]
}

func sortedMembers(members []*etcdserverpb.Member) []uint64 {
	var memberIds uint64slice
	for _, m := range members {
		memberIds = append(memberIds, m.ID)
	}
	sort.Sort(memberIds)
	return memberIds
}
