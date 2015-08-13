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
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"os"
	"time"
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
	Cli         *client.Client
}

// runs the election loop. never returns.
func (c *Config) leaseAndUpdateLoop() {
	for {
		master, err := c.acquireOrRenewLease()
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
func (c *Config) acquireOrRenewLease() (bool, error) {
	//TODO It seems that for the type of leasing done by daemons for HA, we
	//should put locks in the default NS.  Confirm this or create a new NS for HA locks.
	ilock := c.Cli.Locks(api.NamespaceDefault)
	acquiredLock, err := ilock.Get(c.Key)
	//No lock exists, lets create one if possible.
	if err != nil {

		acquiredLock, err = ilock.Create(
			&api.Lock{
				ObjectMeta: api.ObjectMeta{
					Name:      c.Key,
					Namespace: api.NamespaceDefault,
				},
				Spec: api.LockSpec{
					HeldBy:    c.whoami,
					LeaseTime: c.ttl,
				},
			})
		if err != nil {
			glog.Errorf("Lock was NOT created, ERROR = %v", err)
			c.lastLease = time.Now()
			return false, err
		} else {
			glog.Errorf("Lock created successfully %v !", acquiredLock)
		}
	}

	// UPDATE will fail if another node has the lock.  In any case, if an UPDATE fails,
	// we cannot take the lock, so the result is the same - return false and return error details.
	_, err = ilock.Update(acquiredLock)
	if err != nil {
		glog.Errorf("Acquire lock failed.  We don't have the lock, master is %v", acquiredLock)
		return false, err
	}

	glog.Errorf("Acquired lock successfully.  We are the master, yipppeee!")
	return true, nil
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

	//set some reasonable defaults.
	if len(c.whoami) == 0 {
		hostname, err := os.Hostname()
		if err != nil {
			glog.Fatalf("Failed to get hostname: %v", err)
		}
		c.whoami = hostname
		glog.Infof("--whoami is empty, defaulting to %s", c.whoami)
	}
	if c.ttl < 1 {
		c.ttl = 30
		glog.Infof("Set default to 30 seconds for lease time to live")
	}

	go c.leaseAndUpdateLoop()

	glog.Infof("running lease update loop ")
}
