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

package taskreconciler

import (
	"fmt"
	"time"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	bindings "github.com/mesos/mesos-go/scheduler"
	"k8s.io/kubernetes/contrib/mesos/pkg/proc"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/errors"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/metrics"
)

type Action func(driver bindings.SchedulerDriver, cancel <-chan struct{}) <-chan error

type TasksReconciler interface {
	RequestExplicit()
	RequestImplicit()
	Run(driver bindings.SchedulerDriver, done <-chan struct{})
}

type tasksReconciler struct {
	proc.Doer
	Action                             Action
	explicit                           chan struct{} // send an empty struct to trigger explicit reconciliation
	implicit                           chan struct{} // send an empty struct to trigger implicit reconciliation
	cooldown                           time.Duration
	explicitReconciliationAbortTimeout time.Duration
}

func New(doer proc.Doer, action Action,
	cooldown, explicitReconciliationAbortTimeout time.Duration, done <-chan struct{}) TasksReconciler {
	return &tasksReconciler{
		Doer:     doer,
		explicit: make(chan struct{}, 1),
		implicit: make(chan struct{}, 1),
		cooldown: cooldown,
		explicitReconciliationAbortTimeout: explicitReconciliationAbortTimeout,
		Action: func(driver bindings.SchedulerDriver, cancel <-chan struct{}) <-chan error {
			// trigged the reconciler action in the doer's execution context,
			// but it could take a while and the scheduler needs to be able to
			// process updates, the callbacks for which ALSO execute in the SAME
			// deferred execution context -- so the action MUST be executed async.
			errOnce := proc.NewErrorOnce(cancel)
			return errOnce.Send(doer.Do(func() {
				// only triggers the action if we're the currently elected,
				// registered master and runs the action async.
				go func() {
					var err <-chan error
					defer errOnce.Send(err)
					err = action(driver, cancel)
				}()
			})).Err()
		},
	}
}

func (r *tasksReconciler) RequestExplicit() {
	select {
	case r.explicit <- struct{}{}: // noop
	default: // request queue full; noop
	}
}

func (r *tasksReconciler) RequestImplicit() {
	select {
	case r.implicit <- struct{}{}: // noop
	default: // request queue full; noop
	}
}

// execute task reconciliation, returns when r.done is closed. intended to run as a goroutine.
// if reconciliation is requested while another is in progress, the in-progress operation will be
// cancelled before the new reconciliation operation begins.
func (r *tasksReconciler) Run(driver bindings.SchedulerDriver, done <-chan struct{}) {
	var cancel, finished chan struct{}
requestLoop:
	for {
		select {
		case <-done:
			return
		default: // proceed
		}
		select {
		case <-r.implicit:
			metrics.ReconciliationRequested.WithLabelValues("implicit").Inc()
			select {
			case <-done:
				return
			case <-r.explicit:
				break // give preference to a pending request for explicit
			default: // continue
				// don't run implicit reconciliation while explicit is ongoing
				if finished != nil {
					select {
					case <-finished: // continue w/ implicit
					default:
						log.Infoln("skipping implicit reconcile because explicit reconcile is ongoing")
						continue requestLoop
					}
				}
				errOnce := proc.NewErrorOnce(done)
				errCh := r.Do(func() {
					var err error
					defer errOnce.Report(err)
					log.Infoln("implicit reconcile tasks")
					metrics.ReconciliationExecuted.WithLabelValues("implicit").Inc()
					if _, err = driver.ReconcileTasks([]*mesos.TaskStatus{}); err != nil {
						log.V(1).Infof("failed to request implicit reconciliation from mesos: %v", err)
					}
				})
				proc.OnError(errOnce.Send(errCh).Err(), func(err error) {
					log.Errorf("failed to run implicit reconciliation: %v", err)
				}, done)
				goto slowdown
			}
		case <-done:
			return
		case <-r.explicit: // continue
			metrics.ReconciliationRequested.WithLabelValues("explicit").Inc()
		}

		if cancel != nil {
			close(cancel)
			cancel = nil

			// play nice and wait for the prior operation to finish, complain
			// if it doesn't
			select {
			case <-done:
				return
			case <-finished: // noop, expected
			case <-time.After(r.explicitReconciliationAbortTimeout): // very unexpected
				log.Error("reconciler action failed to stop upon cancellation")
			}
		}
		// copy 'finished' to 'fin' here in case we end up with simultaneous go-routines,
		// if cancellation takes too long or fails - we don't want to close the same chan
		// more than once
		cancel = make(chan struct{})
		finished = make(chan struct{})
		go func(fin chan struct{}) {
			startedAt := time.Now()
			defer func() {
				metrics.ReconciliationLatency.Observe(metrics.InMicroseconds(time.Since(startedAt)))
			}()

			metrics.ReconciliationExecuted.WithLabelValues("explicit").Inc()
			defer close(fin)
			err := <-r.Action(driver, cancel)
			if err == errors.ReconciliationCancelledErr {
				metrics.ReconciliationCancelled.WithLabelValues("explicit").Inc()
				log.Infoln(err.Error())
			} else if err != nil {
				log.Errorf("reconciler action failed: %v", err)
			}
		}(finished)
	slowdown:
		// don't allow reconciliation to run very frequently, either explicit or implicit
		select {
		case <-done:
			return
		case <-time.After(r.cooldown): // noop
		}
	} // for
}

// MakeComposite invokes the given ReconcilerAction funcs in sequence, aborting the sequence if reconciliation
// is cancelled. if any other errors occur the composite reconciler will attempt to complete the
// sequence, reporting only the last generated error.
func MakeComposite(done <-chan struct{}, actions ...Action) Action {
	if x := len(actions); x == 0 {
		// programming error
		panic("no actions specified for composite reconciler")
	} else if x == 1 {
		return actions[0]
	}
	chained := func(d bindings.SchedulerDriver, c <-chan struct{}, a, b Action) <-chan error {
		ech := a(d, c)
		ch := make(chan error, 1)
		go func() {
			select {
			case <-done:
			case <-c:
			case e := <-ech:
				if e != nil {
					ch <- e
					return
				}
				ech = b(d, c)
				select {
				case <-done:
				case <-c:
				case e := <-ech:
					if e != nil {
						ch <- e
						return
					}
					close(ch)
					return
				}
			}
			ch <- fmt.Errorf("aborting composite reconciler action")
		}()
		return ch
	}
	result := func(d bindings.SchedulerDriver, c <-chan struct{}) <-chan error {
		return chained(d, c, actions[0], actions[1])
	}
	for i := 2; i < len(actions); i++ {
		i := i
		next := func(d bindings.SchedulerDriver, c <-chan struct{}) <-chan error {
			return chained(d, c, Action(result), actions[i])
		}
		result = next
	}
	return Action(result)
}
