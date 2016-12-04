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

// Package leaderelection implements leader election of a set of endpoints.
// It uses an annotation in the endpoints object to store the record of the
// election state.
//
// This implementation does not guarantee that only one client is acting as a
// leader (a.k.a. fencing). A client observes timestamps captured locally to
// infer the state of the leader election. Thus the implementation is tolerant
// to arbitrary clock skew, but is not tolerant to arbitrary clock skew rate.
//
// However the level of tolerance to skew rate can be configured by setting
// RenewDeadline and LeaseDuration appropriately. The tolerance expressed as a
// maximum tolerated ratio of time passed on the fastest node to time passed on
// the slowest node can be approximately achieved with a configuration that sets
// the same ratio of LeaseDuration to RenewDeadline. For example if a user wanted
// to tolerate some nodes progressing forward in time twice as fast as other nodes,
// the user could set LeaseDuration to 60 seconds and RenewDeadline to 30 seconds.
//
// While not required, some method of clock synchronization between nodes in the
// cluster is highly recommended. It's important to keep in mind when configuring
// this client that the tolerance to skew rate varies inversely to master
// availability.
//
// Larger clusters often have a more lenient SLA for API latency. This should be
// taken into account when configuring the client. The rate of leader transitions
// should be monitored and RetryPeriod and LeaseDuration should be increased
// until the rate is stable and acceptably low. It's important to keep in mind
// when configuring this client that the tolerance to API latency varies inversely
// to master availability.
//
// DISCLAIMER: this is an alpha API. This library will likely change significantly
// or even be removed entirely in subsequent releases. Depend on this API at
// your own risk.
package leaderelection

import (
	"fmt"
	"reflect"
	"time"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	rl "k8s.io/kubernetes/pkg/client/leaderelection/resourcelock"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

const (
	JitterFactor         = 1.2
	DefaultLeaseDuration = 15 * time.Second
	DefaultRenewDeadline = 10 * time.Second
	DefaultRetryPeriod   = 2 * time.Second
)

// NewLeadereElector creates a LeaderElector from a LeaderElecitionConfig
func NewLeaderElector(lec LeaderElectionConfig) (*LeaderElector, error) {
	if lec.LeaseDuration <= lec.RenewDeadline {
		return nil, fmt.Errorf("leaseDuration must be greater than renewDeadline")
	}
	if lec.RenewDeadline <= time.Duration(JitterFactor*float64(lec.RetryPeriod)) {
		return nil, fmt.Errorf("renewDeadline must be greater than retryPeriod*JitterFactor")
	}
	if lec.Lock == nil {
		return nil, fmt.Errorf("Lock must not be nil.")
	}
	return &LeaderElector{
		config: lec,
	}, nil
}

type LeaderElectionConfig struct {
	// Lock is the resource that will be used for locking
	Lock rl.Interface

	// LeaseDuration is the duration that non-leader candidates will
	// wait to force acquire leadership. This is measured against time of
	// last observed ack.
	LeaseDuration time.Duration
	// RenewDeadline is the duration that the acting master will retry
	// refreshing leadership before giving up.
	RenewDeadline time.Duration
	// RetryPeriod is the duration the LeaderElector clients should wait
	// between tries of actions.
	RetryPeriod time.Duration

	// Callbacks are callbacks that are triggered during certain lifecycle
	// events of the LeaderElector
	Callbacks LeaderCallbacks
}

// LeaderCallbacks are callbacks that are triggered during certain
// lifecycle events of the LeaderElector. These are invoked asynchronously.
//
// possible future callbacks:
//  * OnChallenge()
type LeaderCallbacks struct {
	// OnStartedLeading is called when a LeaderElector client starts leading
	OnStartedLeading func(stop <-chan struct{})
	// OnStoppedLeading is called when a LeaderElector client stops leading
	OnStoppedLeading func()
	// OnNewLeader is called when the client observes a leader that is
	// not the previously observed leader. This includes the first observed
	// leader when the client starts.
	OnNewLeader func(identity string)
}

// LeaderElector is a leader election client.
//
// possible future methods:
//  * (le *LeaderElector) IsLeader()
//  * (le *LeaderElector) GetLeader()
type LeaderElector struct {
	config LeaderElectionConfig
	// internal bookkeeping
	observedRecord rl.LeaderElectionRecord
	observedTime   time.Time
	// used to implement OnNewLeader(), may lag slightly from the
	// value observedRecord.HolderIdentity if the transition has
	// not yet been reported.
	reportedLeader string
}

// Run starts the leader election loop
func (le *LeaderElector) Run() {
	defer func() {
		runtime.HandleCrash()
		le.config.Callbacks.OnStoppedLeading()
	}()
	le.acquire()
	stop := make(chan struct{})
	go le.config.Callbacks.OnStartedLeading(stop)
	le.renew()
	close(stop)
}

// RunOrDie starts a client with the provided config or panics if the config
// fails to validate.
func RunOrDie(lec LeaderElectionConfig) {
	le, err := NewLeaderElector(lec)
	if err != nil {
		panic(err)
	}
	le.Run()
}

// GetLeader returns the identity of the last observed leader or returns the empty string if
// no leader has yet been observed.
func (le *LeaderElector) GetLeader() string {
	return le.observedRecord.HolderIdentity
}

// IsLeader returns true if the last observed leader was this client else returns false.
func (le *LeaderElector) IsLeader() bool {
	return le.observedRecord.HolderIdentity == le.config.Lock.Identity()
}

// acquire loops calling tryAcquireOrRenew and returns immediately when tryAcquireOrRenew succeeds.
func (le *LeaderElector) acquire() {
	stop := make(chan struct{})
	wait.JitterUntil(func() {
		succeeded := le.tryAcquireOrRenew()
		le.maybeReportTransition()
		desc := le.config.Lock.Describe()
		if !succeeded {
			glog.V(4).Infof("failed to renew lease %v", desc)
			return
		}
		le.config.Lock.RecordEvent("became leader")
		glog.Infof("sucessfully acquired lease %v", desc)
		close(stop)
	}, le.config.RetryPeriod, JitterFactor, true, stop)
}

// renew loops calling tryAcquireOrRenew and returns immediately when tryAcquireOrRenew fails.
func (le *LeaderElector) renew() {
	stop := make(chan struct{})
	wait.Until(func() {
		err := wait.Poll(le.config.RetryPeriod, le.config.RenewDeadline, func() (bool, error) {
			return le.tryAcquireOrRenew(), nil
		})
		le.maybeReportTransition()
		desc := le.config.Lock.Describe()
		if err == nil {
			glog.V(4).Infof("succesfully renewed lease %v", desc)
			return
		}
		le.config.Lock.RecordEvent("stopped leading")
		glog.Infof("failed to renew lease %v", desc)
		close(stop)
	}, 0, stop)
}

// tryAcquireOrRenew tries to acquire a leader lease if it is not already acquired,
// else it tries to renew the lease if it has already been acquired. Returns true
// on success else returns false.
func (le *LeaderElector) tryAcquireOrRenew() bool {
	now := metav1.Now()
	leaderElectionRecord := rl.LeaderElectionRecord{
		HolderIdentity:       le.config.Lock.Identity(),
		LeaseDurationSeconds: int(le.config.LeaseDuration / time.Second),
		RenewTime:            now,
		AcquireTime:          now,
	}

	// 1. obtain or create the ElectionRecord
	oldLeaderElectionRecord, err := le.config.Lock.Get()
	if err != nil {
		if !errors.IsNotFound(err) {
			glog.Errorf("error retrieving resource lock %v: %v", le.config.Lock.Describe(), err)
			return false
		}
		if err = le.config.Lock.Create(leaderElectionRecord); err != nil {
			glog.Errorf("error initially creating leader election record: %v", err)
			return false
		}
		le.observedRecord = leaderElectionRecord
		le.observedTime = time.Now()
		return true
	}

	// 2. Record obtained, check the Identity & Time
	if !reflect.DeepEqual(le.observedRecord, *oldLeaderElectionRecord) {
		le.observedRecord = *oldLeaderElectionRecord
		le.observedTime = time.Now()
	}
	if le.observedTime.Add(le.config.LeaseDuration).After(now.Time) &&
		oldLeaderElectionRecord.HolderIdentity != le.config.Lock.Identity() {
		glog.Infof("lock is held by %v and has not yet expired", oldLeaderElectionRecord.HolderIdentity)
		return false
	}

	// 3. We're going to try to update. The leaderElectionRecord is set to it's default
	// here. Let's correct it before updating.
	if oldLeaderElectionRecord.HolderIdentity == le.config.Lock.Identity() {
		leaderElectionRecord.AcquireTime = oldLeaderElectionRecord.AcquireTime
	} else {
		leaderElectionRecord.LeaderTransitions = oldLeaderElectionRecord.LeaderTransitions + 1
	}

	// update the lock itself
	if err = le.config.Lock.Update(leaderElectionRecord); err != nil {
		glog.Errorf("Failed to update lock: %v", err)
		return false
	}
	le.observedRecord = leaderElectionRecord
	le.observedTime = time.Now()
	return true
}

func (l *LeaderElector) maybeReportTransition() {
	if l.observedRecord.HolderIdentity == l.reportedLeader {
		return
	}
	l.reportedLeader = l.observedRecord.HolderIdentity
	if l.config.Callbacks.OnNewLeader != nil {
		go l.config.Callbacks.OnNewLeader(l.reportedLeader)
	}
}

func DefaultLeaderElectionConfiguration() componentconfig.LeaderElectionConfiguration {
	return componentconfig.LeaderElectionConfiguration{
		LeaderElect:   false,
		LeaseDuration: metav1.Duration{Duration: DefaultLeaseDuration},
		RenewDeadline: metav1.Duration{Duration: DefaultRenewDeadline},
		RetryPeriod:   metav1.Duration{Duration: DefaultRetryPeriod},
	}
}

// BindFlags binds the common LeaderElectionCLIConfig flags to a flagset
func BindFlags(l *componentconfig.LeaderElectionConfiguration, fs *pflag.FlagSet) {
	fs.BoolVar(&l.LeaderElect, "leader-elect", l.LeaderElect, ""+
		"Start a leader election client and gain leadership before "+
		"executing the main loop. Enable this when running replicated "+
		"components for high availability.")
	fs.DurationVar(&l.LeaseDuration.Duration, "leader-elect-lease-duration", l.LeaseDuration.Duration, ""+
		"The duration that non-leader candidates will wait after observing a leadership "+
		"renewal until attempting to acquire leadership of a led but unrenewed leader "+
		"slot. This is effectively the maximum duration that a leader can be stopped "+
		"before it is replaced by another candidate. This is only applicable if leader "+
		"election is enabled.")
	fs.DurationVar(&l.RenewDeadline.Duration, "leader-elect-renew-deadline", l.RenewDeadline.Duration, ""+
		"The interval between attempts by the acting master to renew a leadership slot "+
		"before it stops leading. This must be less than or equal to the lease duration. "+
		"This is only applicable if leader election is enabled.")
	fs.DurationVar(&l.RetryPeriod.Duration, "leader-elect-retry-period", l.RetryPeriod.Duration, ""+
		"The duration the clients should wait between attempting acquisition and renewal "+
		"of a leadership. This is only applicable if leader election is enabled.")
}
