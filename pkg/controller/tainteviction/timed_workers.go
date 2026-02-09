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

package tainteviction

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// WorkArgs keeps arguments that will be passed to the function executed by the worker.
type WorkArgs struct {
	// Object is the work item. The UID is only set if it was set when adding the work item.
	Object NamespacedObject
}

// KeyFromWorkArgs creates a key for the given `WorkArgs`.
//
// The key is the same as the NamespacedName of the object in the work item,
// i.e. the UID is ignored. There cannot be two different
// work items with the same NamespacedName and different UIDs.
func (w *WorkArgs) KeyFromWorkArgs() string {
	return w.Object.NamespacedName.String()
}

// NewWorkArgs is a helper function to create new `WorkArgs` without a UID.
func NewWorkArgs(name, namespace string) *WorkArgs {
	return &WorkArgs{
		Object: NamespacedObject{NamespacedName: types.NamespacedName{Namespace: namespace, Name: name}},
	}
}

// TimedWorker is a responsible for executing a function no earlier than at FireAt time.
type TimedWorker struct {
	WorkItem  *WorkArgs
	CreatedAt time.Time
	FireAt    time.Time
	Timer     clock.Timer
	cancelled atomic.Bool
}

// createWorker creates a TimedWorker that will execute `f` not earlier than `fireAt`.
// Returns nil if the work was started immediately and doesn't need a timer.
func createWorker(ctx context.Context, wg *sync.WaitGroup, args *WorkArgs, createdAt time.Time, fireAt time.Time, f func(ctx context.Context, fireAt time.Time, args *WorkArgs) error, clock clock.WithDelayedExecution) *TimedWorker {
	delay := fireAt.Sub(createdAt)
	logger := klog.FromContext(ctx)

	worker := TimedWorker{
		WorkItem:  args,
		CreatedAt: createdAt,
		FireAt:    fireAt,
	}

	// This dance with the cancelled flag is here so that we can be sure that once TimedWorker.Cancel returns,
	// we either never get to any processing, or the thread is already registered with the WaitGroup.
	// Otherwise, the following sequence can happen:
	//   1. TimedWorker.Timer fires and gets to go wrapper().
	//   2. TimedWorker.Cancel is called to stop the timer, but a goroutine is already running.
	//      It's started, but wg.Go hasn't been called yet to register the goroutine.
	//   3. We call wg.Wait, which unblocks, because the inner wg.Go hasn't been called yet.
	//      This causes wg.Wait to unblock after TimedWorker.Cancel is called and still start a goroutine.
	// So in our case we can still get to starting an unregistered goroutine, but it will exit immediately.
	wrapper := func() {
		wg.Go(func() {
			if worker.cancelled.Load() {
				return
			}
			if err := f(ctx, fireAt, args); err != nil {
				logger.Error(err, "TaintEvictionController: timed worker failed")
			}
		})
	}
	if delay <= 0 {
		wrapper()
		return nil
	}
	worker.Timer = clock.AfterFunc(delay, wrapper)
	return &worker
}

// Cancel cancels the execution of function by the `TimedWorker`
func (w *TimedWorker) Cancel() {
	if w != nil {
		// Mark the worker as cancelled.
		// This ensures the worker is either already running or unstarted on return from Cancel.
		w.cancelled.Store(true)
		w.Timer.Stop()
	}
}

// TimedWorkerQueue keeps a set of TimedWorkers that are still wait for execution.
type TimedWorkerQueue struct {
	sync.Mutex
	// map of workers keyed by string returned by 'KeyFromWorkArgs' from the given worker.
	// Entries may be nil if the work didn't need a timer and is already running.
	workers  map[string]*TimedWorker
	workerWG sync.WaitGroup
	workFunc func(ctx context.Context, fireAt time.Time, args *WorkArgs) error
	clock    clock.WithDelayedExecution
}

// CreateWorkerQueue creates a new TimedWorkerQueue for workers that will execute
// given function `f`.
func CreateWorkerQueue(f func(ctx context.Context, fireAt time.Time, args *WorkArgs) error) *TimedWorkerQueue {
	return &TimedWorkerQueue{
		workers:  make(map[string]*TimedWorker),
		workFunc: f,
		clock:    clock.RealClock{},
	}
}

func (q *TimedWorkerQueue) getWrappedWorkerFunc(key string) func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
	return func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
		logger := klog.FromContext(ctx)
		logger.V(4).Info("Firing worker", "item", key, "firedTime", fireAt)
		err := q.workFunc(ctx, fireAt, args)
		q.Lock()
		defer q.Unlock()
		logger.V(4).Info("Worker finished, removing", "item", key, "err", err)
		delete(q.workers, key)
		return err
	}
}

// AddWork adds a work to the WorkerQueue which will be executed not earlier than `fireAt`.
// If replace is false, an existing work item will not get replaced, otherwise it
// gets canceled and the new one is added instead.
func (q *TimedWorkerQueue) AddWork(ctx context.Context, args *WorkArgs, createdAt time.Time, fireAt time.Time) {
	key := args.KeyFromWorkArgs()
	logger := klog.FromContext(ctx)

	q.Lock()
	defer q.Unlock()
	if _, exists := q.workers[key]; exists {
		logger.V(4).Info("Trying to add already existing work, skipping", "item", key, "createTime", createdAt, "firedTime", fireAt)
		return
	}
	logger.V(4).Info("Adding TimedWorkerQueue item and to be fired at firedTime", "item", key, "createTime", createdAt, "firedTime", fireAt)
	worker := createWorker(ctx, &q.workerWG, args, createdAt, fireAt, q.getWrappedWorkerFunc(key), q.clock)
	q.workers[key] = worker
}

// UpdateWork adds or replaces a work item such that it will be executed not earlier than `fireAt`.
// This is a cheap no-op when the old and new fireAt are the same.
func (q *TimedWorkerQueue) UpdateWork(ctx context.Context, args *WorkArgs, createdAt time.Time, fireAt time.Time) {
	key := args.KeyFromWorkArgs()
	logger := klog.FromContext(ctx)

	q.Lock()
	defer q.Unlock()
	if worker, exists := q.workers[key]; exists {
		if worker == nil {
			logger.V(4).Info("Keeping existing work, already in progress", "item", key)
			return
		}
		if worker.FireAt.Compare(fireAt) == 0 {
			logger.V(4).Info("Keeping existing work, same time", "item", key, "createTime", worker.CreatedAt, "firedTime", worker.FireAt)
			return
		}
		logger.V(4).Info("Replacing existing work", "item", key, "createTime", worker.CreatedAt, "firedTime", worker.FireAt)
		worker.Cancel()
	}
	logger.V(4).Info("Adding TimedWorkerQueue item and to be fired at firedTime", "item", key, "createTime", createdAt, "firedTime", fireAt)
	worker := createWorker(ctx, &q.workerWG, args, createdAt, fireAt, q.getWrappedWorkerFunc(key), q.clock)
	q.workers[key] = worker
}

// CancelWork removes scheduled function execution from the queue. Returns true if work was cancelled.
// The key must be the same as the one returned by WorkArgs.KeyFromWorkArgs, i.e.
// the result of NamespacedName.String.
func (q *TimedWorkerQueue) CancelWork(logger klog.Logger, key string) bool {
	q.Lock()
	defer q.Unlock()
	worker, found := q.workers[key]
	result := false
	if found {
		logger.V(4).Info("Cancelling TimedWorkerQueue item", "item", key, "time", time.Now())
		if worker != nil {
			result = true
			worker.Cancel()
		}
		delete(q.workers, key)
	}
	return result
}

// GetWorkerUnsafe returns a TimedWorker corresponding to the given key.
// Unsafe method - workers have attached goroutines which can fire after this function is called.
func (q *TimedWorkerQueue) GetWorkerUnsafe(key string) *TimedWorker {
	q.Lock()
	defer q.Unlock()
	return q.workers[key]
}

// CancelAndWait cancels all workers and waits for all running threads to terminate before returning.
func (q *TimedWorkerQueue) CancelAndWait() {
	// Wait must be called after Unlock, otherwise this hangs.
	defer q.workerWG.Wait()
	q.Lock()
	defer q.Unlock()
	for _, worker := range q.workers {
		worker.Cancel()
	}
	q.workers = make(map[string]*TimedWorker)
}
