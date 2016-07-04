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

package executor

import (
	"time"

	log "github.com/golang/glog"
	bindings "github.com/mesos/mesos-go/executor"
)

// func that attempts suicide
type jumper func(bindings.ExecutorDriver, <-chan struct{})

type suicideWatcher interface {
	Next(time.Duration, bindings.ExecutorDriver, jumper) suicideWatcher
	Reset(time.Duration) bool
	Stop() bool
}

// TODO(jdef) add metrics for this?
type suicideTimer struct {
	timer *time.Timer
}

func (w *suicideTimer) Next(d time.Duration, driver bindings.ExecutorDriver, f jumper) suicideWatcher {
	return &suicideTimer{
		timer: time.AfterFunc(d, func() {
			log.Warningf("Suicide timeout (%v) expired", d)
			f(driver, nil)
		}),
	}
}

func (w *suicideTimer) Stop() (result bool) {
	if w != nil && w.timer != nil {
		log.Infoln("stopping suicide watch") //TODO(jdef) debug
		result = w.timer.Stop()
	}
	return
}

// return true if the timer was successfully reset
func (w *suicideTimer) Reset(d time.Duration) bool {
	if w != nil && w.timer != nil {
		log.Infoln("resetting suicide watch") //TODO(jdef) debug
		w.timer.Reset(d)
		return true
	}
	return false
}
