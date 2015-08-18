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
	Key    string
	whoami string
	ttl    uint64
	sleep  time.Duration
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

		//Proactively give up the lease every 50 tries.
		var master bool = false
		var err error = nil

		//Lets see if we are the master.
		if countsleeps < 50 {
			master, err = c.acquireOrRenewLease()
			if err != nil {
				glog.Errorf("Error in determining if master. %v, looping", err)
			} else {
				glog.Errorf("Master info acquired %v", master)
			}
		}

		if err := c.update(master); err != nil {
			glog.Errorf("Error updating master status %v", err)
		} else {
			glog.Errorf("Done updating master status %v", c)
		}
		countsleeps++
		glog.Errorf("..sleep.. %v %v", c.sleep, countsleeps)
		time.Sleep(c.sleep)
	}
}

// A lock can be invalid if (1) it doesnt exist or (2) we decided to delete it.
func (c *Config) getValidLockOrDelete() (*api.Lock, error) {
	ilock := c.Cli.Locks(api.NamespaceDefault)
	acquiredLock, err := ilock.Get(c.Key)
	if err != nil {
		glog.Errorf("Error could not get any lock : %v", err)
		return nil, err
	}

	//Read the renewal time.  We will delete the lock unless the renewal time
	//is within our TTL.
	rTime, _ := time.Parse("Mon Jan 2 15:04:05 -0700 MST 2006", acquiredLock.Spec.RenewTime)

	//If the current time is larger than the last update time, the lock is old,
	//regardless of the owner, we can safely delete it - because any lock owner
	//by now would have renewed it unless it has died off.
	if uint64(time.Since(rTime)/time.Second) >= c.ttl {
		ilock.Delete(c.Key)
		glog.Errorf("Deleted a stale lock (last renewal was %v, current time is %v).  Time to make a new one !", rTime, time.Now())
		return nil, nil
	}

	//Lock looks good.  Lets return it.
	return acquiredLock, nil
}

// acquireOrRenewLease either races to acquire a new master lease, or update the existing master's lease
// returns true if we have the lease, and an error if one occurs.
// TODO: use the master election utility once it is merged in.
func (c *Config) acquireOrRenewLease() (bool, error) {
	acquiredLock, err := c.getValidLockOrDelete()
	ilock := c.Cli.Locks(api.NamespaceDefault)
	//No lock exists, lets create one if possible.
	if acquiredLock == nil {
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
		}
	}
	// UPDATE will fail if another node has the lock.  In any case, if an UPDATE fails,
	// we cannot take the lock, so the result is the same - return false and return error details.
	glog.Errorf("Updating acquired lock %v", acquiredLock)
	_, err = ilock.Update(acquiredLock)
	if err != nil {
		glog.Errorf("Acquire lock failed.  We don't have the lock, master is %v", acquiredLock)
		return false, err
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
