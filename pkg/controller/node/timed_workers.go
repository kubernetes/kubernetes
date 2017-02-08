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

package node

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/types"

	"github.com/golang/glog"
)

// WorkArgs keeps arguments that will be passed to tha function executed by the worker.
type WorkArgs struct {
	NamespacedName types.NamespacedName
}

// KeyFromWorkArgs creates a key for the given `WorkArgs`
func (w *WorkArgs) KeyFromWorkArgs() string {
	return w.NamespacedName.String()
}

// NewWorkArgs is a helper function to create new `WorkArgs`
func NewWorkArgs(name, namespace string) *WorkArgs {
	return &WorkArgs{types.NamespacedName{Namespace: namespace, Name: name}}
}

// TimedWorker is a responsible for executing a function no earlier than at FireAt time.
type TimedWorker struct {
	WorkItem  *WorkArgs
	CreatedAt time.Time
	FireAt    time.Time
	Timer     *time.Timer
}

// CreateWorker creates a TimedWorker that will execute `f` not earlier than `fireAt`.
func CreateWorker(args *WorkArgs, createdAt time.Time, fireAt time.Time, f func(args *WorkArgs) error) *TimedWorker {
	delay := fireAt.Sub(time.Now())
	if delay <= 0 {
		go f(args)
		return nil
	}
	timer := time.AfterFunc(delay, func() { f(args) })
	return &TimedWorker{
		WorkItem:  args,
		CreatedAt: createdAt,
		FireAt:    fireAt,
		Timer:     timer,
	}
}

// Cancel cancels the execution of function by the `TimedWorker`
func (w *TimedWorker) Cancel() {
	if w != nil {
		w.Timer.Stop()
	}
}

// TimedWorkerQueue keeps a set of TimedWorkers that still wait for execution.
type TimedWorkerQueue struct {
	sync.Mutex
	workers  map[string]*TimedWorker
	workFunc func(args *WorkArgs) error
}

// CreateWorkerQueue creates a new TimedWorkerQueue for workers that will execute
// given function `f`.
func CreateWorkerQueue(f func(args *WorkArgs) error) *TimedWorkerQueue {
	return &TimedWorkerQueue{
		workers:  make(map[string]*TimedWorker),
		workFunc: f,
	}
}

func (q *TimedWorkerQueue) getWrappedWorkerFunc(key string) func(args *WorkArgs) error {
	return func(args *WorkArgs) error {
		err := q.workFunc(args)
		q.Lock()
		defer q.Unlock()
		if err == nil {
			q.workers[key] = nil
		} else {
			delete(q.workers, key)
		}
		return err
	}
}

// AddWork adds a work to the WorkerQueue which will be executed not earlier than `fireAt`.
func (q *TimedWorkerQueue) AddWork(args *WorkArgs, createdAt time.Time, fireAt time.Time) {
	key := args.KeyFromWorkArgs()

	q.Lock()
	defer q.Unlock()
	if _, exists := q.workers[key]; exists {
		glog.Warningf("Trying to add already existing work for %+v. Skipping.", args)
		return
	}
	worker := CreateWorker(args, createdAt, fireAt, q.getWrappedWorkerFunc(key))
	if worker == nil {
		return
	}
	q.workers[key] = worker
}

// CancelWork removes scheduled function execution from the queue.
func (q *TimedWorkerQueue) CancelWork(key string) {
	q.Lock()
	defer q.Unlock()
	worker, found := q.workers[key]
	if found {
		worker.Cancel()
		delete(q.workers, key)
	}
}

// GetWorkerUnsafe returns a TimedWorker corresponding to the given key.
// Unsafe method - workers have attached goroutines which can fire afater this function is called.
func (q *TimedWorkerQueue) GetWorkerUnsafe(key string) *TimedWorker {
	q.Lock()
	defer q.Unlock()
	return q.workers[key]
}
