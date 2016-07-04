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

package ha

import (
	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/election"
)

type roleType int

const (
	followerRole roleType = iota
	masterRole
	retiredRole
)

type candidateService struct {
	sched     *SchedulerProcess
	newDriver DriverFactory
	role      roleType
	valid     ValidationFunc
}

type ValidationFunc func(desiredUid, currentUid string)

func NewCandidate(s *SchedulerProcess, f DriverFactory, v ValidationFunc) election.Service {
	return &candidateService{
		sched:     s,
		newDriver: f,
		role:      followerRole,
		valid:     v,
	}
}

func (self *candidateService) Validate(desired, current election.Master) {
	if self.valid != nil {
		self.valid(string(desired), string(current))
	}
}

func (self *candidateService) Start() {
	if self.role == followerRole {
		log.Info("elected as master")
		self.role = masterRole
		self.sched.Elect(self.newDriver)
	}
}

func (self *candidateService) Stop() {
	if self.role == masterRole {
		log.Info("retiring from master")
		self.role = retiredRole
		// order is important here, watchers of a SchedulerProcess will
		// check SchedulerProcess.Failover() once Done() is closed.
		close(self.sched.failover)
		self.sched.End()
	}
}
