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

// podmaster is a simple utility, it attempts to acquire and maintain a lease-lock from etcd using compare-and-swap.
// if it is the master, it copies a source file into a destination file.  If it is not the master, it makes sure it is removed.
//
// typical usage is to copy a Pod manifest from a staging directory into the kubelet's directory, for example:
//   podmaster --etcd-servers=http://127.0.0.1:4001 --key=scheduler --source-file=/kubernetes/kube-scheduler.manifest --dest-file=/manifests/kube-scheduler.manifest
package controller

import (
	"strings"
	"time"

	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
)

type Config struct {
	EtcdServers string
	Key         string
	whoami      string
	ttl         uint64
	sleep       time.Duration
	lastLease   time.Time
	Running     func() bool
	Lease       func() bool
	Unlease     func() bool
}

// runs the election loop. never returns.
func (c *Config) leaseAndUpdateLoop(etcdClient *etcd.Client) {
	for {
		master, err := c.acquireOrRenewLease(etcdClient)
		if err != nil {
			glog.Errorf("Error in master election: %v", err)
			if uint64(time.Now().Sub(c.lastLease).Seconds()) < c.ttl {
				continue
			}
			// Our lease has expired due to our own accounting, pro-actively give it
			// up, even if we couldn't contact etcd.
			glog.Infof("Too much time has elapsed, giving up lease.")
			master = false
		}
		if err := c.update(master); err != nil {
			glog.Errorf("Error updating files: %v", err)
		}
		time.Sleep(c.sleep)
	}
}

// acquireOrRenewLease either races to acquire a new master lease, or update the existing master's lease
// returns true if we have the lease, and an error if one occurs.
// TODO: use the master election utility once it is merged in.
func (c *Config) acquireOrRenewLease(etcdClient *etcd.Client) (bool, error) {
	result, err := etcdClient.Get(c.Key, false, false)
	if err != nil {
		if etcdstorage.IsEtcdNotFound(err) {
			// there is no current master, try to become master, create will fail if the key already exists
			_, err := etcdClient.Create(c.Key, c.whoami, c.ttl)
			if err != nil {
				return false, err
			}
			c.lastLease = time.Now()
			return true, nil
		}
		return false, err
	}
	if result.Node.Value == c.whoami {
		glog.Infof("key already exists, we are the master (%s)", result.Node.Value)
		// we extend our lease @ 1/2 of the existing TTL, this ensures the master doesn't flap around
		if result.Node.Expiration.Sub(time.Now()) < time.Duration(c.ttl/2)*time.Second {
			_, err := etcdClient.CompareAndSwap(c.Key, c.whoami, c.ttl, c.whoami, result.Node.ModifiedIndex)
			if err != nil {
				return false, err
			}
		}
		c.lastLease = time.Now()
		return true, nil
	}
	glog.Infof("key already exists, the master is %s, sleeping.", result.Node.Value)
	return false, nil
}

// Update acts on the current state of the lease.
func (c *Config) update(master bool) error {
	switch {
	case master && !c.Running():
		c.Lease()
		return nil
	case !master && c.Running():
		c.Unlease()
		return nil
	}
	return nil
}

func RunLease(c *Config) {
	machines := strings.Split(c.EtcdServers, ",")
	etcdClient := etcd.NewClient(machines)

	go c.leaseAndUpdateLoop(etcdClient)

	glog.Infof("running lease update loop ")
}
