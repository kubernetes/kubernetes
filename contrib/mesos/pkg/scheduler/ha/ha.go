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
	"fmt"
	"sync/atomic"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	bindings "github.com/mesos/mesos-go/scheduler"
	"k8s.io/kubernetes/contrib/mesos/pkg/proc"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
)

type DriverFactory func() (bindings.SchedulerDriver, error)

type stageType int32

const (
	initStage stageType = iota
	standbyStage
	masterStage
	finStage
)

func (stage *stageType) transition(from, to stageType) bool {
	return atomic.CompareAndSwapInt32((*int32)(stage), int32(from), int32(to))
}

func (s *stageType) transitionTo(to stageType, unless ...stageType) bool {
	if len(unless) == 0 {
		atomic.StoreInt32((*int32)(s), int32(to))
		return true
	}
	for {
		state := s.get()
		for _, x := range unless {
			if state == x {
				return false
			}
		}
		if s.transition(state, to) {
			return true
		}
	}
}

func (stage *stageType) get() stageType {
	return stageType(atomic.LoadInt32((*int32)(stage)))
}

// execute some action in the deferred context of the process, but only if we
// match the stage of the process at the time the action is executed.
func (stage stageType) Do(p *SchedulerProcess, a proc.Action) <-chan error {
	errOnce := proc.NewErrorOnce(p.fin)
	errOuter := p.Do(proc.Action(func() {
		switch stage {
		case standbyStage:
			//await standby signal or death
			select {
			case <-p.standby:
			case <-p.Done():
			}
		case masterStage:
			//await elected signal or death
			select {
			case <-p.elected:
			case <-p.Done():
			}
		case finStage:
			errOnce.Reportf("scheduler process is dying, dropping action")
			return
		default:
		}
		errOnce.Report(stage.When(p, a))
	}))
	return errOnce.Send(errOuter).Err()
}

// execute some action only if we match the stage of the scheduler process
func (stage stageType) When(p *SchedulerProcess, a proc.Action) (err error) {
	if stage != (&p.stage).get() {
		err = fmt.Errorf("failed to execute deferred action, expected lifecycle stage %v instead of %v", stage, p.stage)
	} else {
		a()
	}
	return
}

type SchedulerProcess struct {
	proc.Process
	bindings.Scheduler
	stage    stageType
	elected  chan struct{} // upon close we've been elected
	failover chan struct{} // closed indicates that we should failover upon End()
	standby  chan struct{}
	fin      chan struct{}
}

func New(framework bindings.Scheduler) *SchedulerProcess {
	p := &SchedulerProcess{
		Process:   proc.New(),
		Scheduler: framework,
		stage:     initStage,
		elected:   make(chan struct{}),
		failover:  make(chan struct{}),
		standby:   make(chan struct{}),
		fin:       make(chan struct{}),
	}
	runtime.On(p.Running(), p.begin)
	return p
}

func (self *SchedulerProcess) begin() {
	if (&self.stage).transition(initStage, standbyStage) {
		close(self.standby)
		log.Infoln("scheduler process entered standby stage")
	} else {
		log.Errorf("failed to transition from init to standby stage")
	}
}

func (self *SchedulerProcess) End() <-chan struct{} {
	if (&self.stage).transitionTo(finStage, finStage) {
		defer close(self.fin)
		log.Infoln("scheduler process entered fin stage")
	}
	return self.Process.End()
}

func (self *SchedulerProcess) Elect(newDriver DriverFactory) {
	errOnce := proc.NewErrorOnce(self.fin)
	proc.OnError(errOnce.Send(standbyStage.Do(self, proc.Action(func() {
		if !(&self.stage).transition(standbyStage, masterStage) {
			log.Errorf("failed to transition from standby to master stage, aborting")
			self.End()
			return
		}
		log.Infoln("scheduler process entered master stage")
		drv, err := newDriver()
		if err != nil {
			log.Errorf("failed to fetch scheduler driver: %v", err)
			self.End()
			return
		}
		log.V(1).Infoln("starting driver...")
		stat, err := drv.Start()
		if stat == mesos.Status_DRIVER_RUNNING && err == nil {
			log.Infoln("driver started successfully and is running")
			close(self.elected)
			go func() {
				defer self.End()
				_, err := drv.Join()
				if err != nil {
					log.Errorf("driver failed with error: %v", err)
				}
				errOnce.Report(err)
			}()
			return
		}
		defer self.End()
		if err != nil {
			log.Errorf("failed to start scheduler driver: %v", err)
		} else {
			log.Errorf("expected RUNNING status, not %v", stat)
		}
	}))).Err(), func(err error) {
		defer self.End()
		log.Errorf("failed to handle election event, aborting: %v", err)
	}, self.fin)
}

func (self *SchedulerProcess) Terminal() <-chan struct{} {
	return self.fin
}

func (self *SchedulerProcess) Elected() <-chan struct{} {
	return self.elected
}

func (self *SchedulerProcess) Failover() <-chan struct{} {
	return self.failover
}

type masterProcess struct {
	*SchedulerProcess
	doer proc.Doer
}

func (self *masterProcess) Done() <-chan struct{} {
	return self.SchedulerProcess.Terminal()
}

func (self *masterProcess) Do(a proc.Action) <-chan error {
	return self.doer.Do(a)
}

// returns a Process instance that will only execute a proc.Action if the scheduler is the elected master
func (self *SchedulerProcess) Master() proc.Process {
	return &masterProcess{
		SchedulerProcess: self,
		doer: proc.DoWith(self, proc.DoerFunc(func(a proc.Action) <-chan error {
			return proc.ErrorChan(masterStage.When(self, a))
		})),
	}
}

func (self *SchedulerProcess) logError(ch <-chan error) {
	self.OnError(ch, func(err error) {
		log.Errorf("failed to execute scheduler action: %v", err)
	})
}

func (self *SchedulerProcess) Registered(drv bindings.SchedulerDriver, fid *mesos.FrameworkID, mi *mesos.MasterInfo) {
	self.logError(self.Master().Do(proc.Action(func() {
		self.Scheduler.Registered(drv, fid, mi)
	})))
}

func (self *SchedulerProcess) Reregistered(drv bindings.SchedulerDriver, mi *mesos.MasterInfo) {
	self.logError(self.Master().Do(proc.Action(func() {
		self.Scheduler.Reregistered(drv, mi)
	})))
}

func (self *SchedulerProcess) Disconnected(drv bindings.SchedulerDriver) {
	self.logError(self.Master().Do(proc.Action(func() {
		self.Scheduler.Disconnected(drv)
	})))
}

func (self *SchedulerProcess) ResourceOffers(drv bindings.SchedulerDriver, off []*mesos.Offer) {
	self.logError(self.Master().Do(proc.Action(func() {
		self.Scheduler.ResourceOffers(drv, off)
	})))
}

func (self *SchedulerProcess) OfferRescinded(drv bindings.SchedulerDriver, oid *mesos.OfferID) {
	self.logError(self.Master().Do(proc.Action(func() {
		self.Scheduler.OfferRescinded(drv, oid)
	})))
}

func (self *SchedulerProcess) StatusUpdate(drv bindings.SchedulerDriver, ts *mesos.TaskStatus) {
	self.logError(self.Master().Do(proc.Action(func() {
		self.Scheduler.StatusUpdate(drv, ts)
	})))
}

func (self *SchedulerProcess) FrameworkMessage(drv bindings.SchedulerDriver, eid *mesos.ExecutorID, sid *mesos.SlaveID, m string) {
	self.logError(self.Master().Do(proc.Action(func() {
		self.Scheduler.FrameworkMessage(drv, eid, sid, m)
	})))
}

func (self *SchedulerProcess) SlaveLost(drv bindings.SchedulerDriver, sid *mesos.SlaveID) {
	self.logError(self.Master().Do(proc.Action(func() {
		self.Scheduler.SlaveLost(drv, sid)
	})))
}

func (self *SchedulerProcess) ExecutorLost(drv bindings.SchedulerDriver, eid *mesos.ExecutorID, sid *mesos.SlaveID, x int) {
	self.logError(self.Master().Do(proc.Action(func() {
		self.Scheduler.ExecutorLost(drv, eid, sid, x)
	})))
}

func (self *SchedulerProcess) Error(drv bindings.SchedulerDriver, msg string) {
	self.Scheduler.Error(drv, msg)
}
