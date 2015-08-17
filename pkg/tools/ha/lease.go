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

//This utility uses the Lock API in the client to acquire a lock.
//It provides a uniform and consistent semantics for.
// - Logging how many times a lease is gained/lost.
// - Starting/Stopping a daemon (via callbacks).
// - Logging errors which might occur if a daemon's lease isn't consistent with its running state.
package ha

import (
	"fmt"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/clientcmd/api"
	"os"
	"time"
)

// A program that is running may "use" a lease.
// This struct has some metadata about that program.
// It is used for debugging/logging (i.e. print how many times we got the lease)
// It also is used to track the state of a program (i.e. if lease obtained, Running should be true)
type LeaseUser struct {

	//Note: In general 'Running' should be 'false' when you first start an HA process, and it should
	//be flipped to 'true' once you acquire the lock and start up.
	Running      bool // This is how users verify that their process is running.
	LeasesGained int  // Number of times lease has been granted.
	LeasesLost   int  // Number of times the lease was lost.
}

// Configuration options needed for running the lease loop.
type Config struct {
	Key       string
	whoami    string
	ttl       uint64
	sleep     time.Duration
	lastLease time.Time
	// These two functions return "err" or else nil.
	// They should also update information in the LeaseUserInfo struct
	// about the state of the lease owner.
	LeaseGained   func(lu *LeaseUser) bool
	LeaseLost     func(lu *LeaseUser) bool
	Cli           *client.Client
	LeaseUserInfo *LeaseUser
}

// RunHA runs a process in a highly available fashion.
func RunHA(kubecfg string, master string, start func(l *LeaseUser) bool, stop func(l *LeaseUser) bool, lockName string) {

	//We need a kubeconfig in order to use the locking API, so we create it here.
	kubeconfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: kubecfg},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: master}}).ClientConfig()

	if err != nil {
		glog.Infof("Exiting, couldn't create kube configuration with parameters cfg=%v and master=%v ", kubecfg, master)
		os.Exit(1)
	}

	kubeClient, err := client.New(kubeconfig)

	leaseUserInfo := LeaseUser{
		Running:      false,
		LeasesGained: 0,
		LeasesLost:   0,
	}

	//Acquire a lock before starting.
	//TODO some of these will change now that implementing robs lock.
	//we can delete some params...
	mcfg := Config{
		Key:           lockName,
		LeaseUserInfo: &leaseUserInfo,
		LeaseGained:   start,
		LeaseLost:     stop,
		Cli:           kubeClient}

	RunLease(&mcfg)
	glog.Infof("zz Done with starting the lease loop for %v.", lockName)
}

var countsleeps = 0

// leaseAndUpdateLoop runs the loop to acquire a lease.  This will continue trying to get a lease
// until it succeeds, and then callbacks sent in by the user will be triggered.
// Likewise, it can give up the lease, in which case the LeaseLost() callbacks will be triggered.
func (c *Config) leaseAndUpdateLoop() {
	for {
		glog.Errorf("zzzz")
		master, err := c.acquireOrRenewLease()
		glog.Errorf("zzzz acquired")
		if err != nil {
			glog.Errorf("Error in lock acquisition: %v, looping", err)
		} else {
			glog.Errorf("Checking ttl = %v , [[[ last %v ]]]  [[[ now %v ]]],  time difference = %v ", c.ttl, c.lastLease, time.Now(), time.Now().Sub(c.lastLease))

			//We may need to "unset" the master status... if a leasettl expired!
			//If Location is nil, then time is uninitialized.
			if !c.lastLease.IsZero() && uint64(time.Now().Sub(c.lastLease).Seconds()) >= c.ttl {
				glog.Errorf("Too much time has elapsed, giving up lease.")
				master = false
			}
			if err := c.update(master); err != nil {
				glog.Errorf("Error updating master status %v", err)
			} else {
				glog.Errorf("Done updating master status %v", c)
			}
		}
		countsleeps++
		glog.Errorf("..sleep.. %v %v", c.sleep, countsleeps)
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
			return false, err
		} else {
			glog.Errorf("Lock created successfully %v !", acquiredLock)
			//Set the time of the lock to "now".
			c.lastLease = time.Now()
		}
	}

	// UPDATE will fail if another node has the lock.  In any case, if an UPDATE fails,
	// we cannot take the lock, so the result is the same - return false and return error details.
	_, err = ilock.Update(acquiredLock)
	if err != nil {
		glog.Errorf("Acquire lock failed.  We don't have the lock, master is %v", acquiredLock)
		return false, err
	}

	//In the case where the daemon has restarted, and happens to find a lock from etcd - we need to set lease time to now.
	//Also, in the case that a fresh lock is created, we need to set the lease time to now.
	if c.lastLease.IsZero() {
		c.lastLease = time.Now()
	}
	glog.Errorf("Acquired lock successfully.  We are the master, yipppeee!")

	//When we return true, this should result in a update to master status.
	return true, nil
}

// Update acts on the current state of the lease.
// This method heavily relies on correct implementation of the functions in the Config interface.
func (c *Config) update(master bool) error {
	glog.Errorf("Updating state Lease info and lease gained/lost: master = %v, Leases: gained: %v lost: %v", master, c.LeaseUserInfo.LeasesGained, c.LeaseUserInfo.LeasesLost)
	switch {
	case master && !c.LeaseUserInfo.Running:
		glog.Errorf("master + !running")
		go c.LeaseGained(c.LeaseUserInfo)
		time.Sleep(1 * time.Second)
		c.LeaseUserInfo.LeasesGained++
		if !c.LeaseUserInfo.Running {
			return fmt.Errorf("Process %v did not update its Running field to TRUE after ACQUIRING the lease!", c)
		} else {
			return nil
		}
	case !master && c.LeaseUserInfo.Running:
		glog.Errorf("!master + running")
		go c.LeaseLost(c.LeaseUserInfo)
		time.Sleep(1 * time.Second)
		c.LeaseUserInfo.LeasesLost++
		if !c.LeaseUserInfo.Running {
			return fmt.Errorf("Process %v did not update its Running field to FALSE after LOSING the lease!", c)
		} else {
			return nil
		}
	default:
		glog.Errorf("We don't need to do any updating ( Master: %v , Running: %v )", master, c.LeaseUserInfo.Running)
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
		glog.Infof("Set default to %v seconds for lease", c.ttl)
	}
	if c.sleep < 1 {
		c.sleep = 5 * time.Second
		glog.Infof("Set default to %v seconds for sleep", c.sleep)
	}

	glog.Infof("Config : %v, sleep %v", c, c.sleep)

	go c.leaseAndUpdateLoop()

	glog.Infof("running lease update loop ")
}
