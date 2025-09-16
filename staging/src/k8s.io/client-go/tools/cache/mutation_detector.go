/*
Copyright 2016 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"os"
	"reflect"
	"strconv"
	"sync"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
)

var mutationDetectionEnabled = false

func init() {
	mutationDetectionEnabled, _ = strconv.ParseBool(os.Getenv("KUBE_CACHE_MUTATION_DETECTOR"))
}

// MutationDetector is able to monitor objects for mutation within a limited window of time
type MutationDetector interface {
	// AddObject adds the given object to the set being monitored for a while from now
	AddObject(obj interface{})

	// Run starts the monitoring and does not return until the monitoring is stopped.
	Run(stopCh <-chan struct{})
}

// NewCacheMutationDetector creates a new instance for the defaultCacheMutationDetector.
func NewCacheMutationDetector(name string) MutationDetector {
	if !mutationDetectionEnabled {
		return dummyMutationDetector{}
	}
	//nolint:logcheck // This code shouldn't be used in production.
	klog.Warningln("Mutation detector is enabled, this will result in memory leakage.")
	return &defaultCacheMutationDetector{name: name, period: 1 * time.Second, retainDuration: 2 * time.Minute}
}

type dummyMutationDetector struct{}

func (dummyMutationDetector) Run(stopCh <-chan struct{}) {
}
func (dummyMutationDetector) AddObject(obj interface{}) {
}

// defaultCacheMutationDetector gives a way to detect if a cached object has been mutated
// It has a list of cached objects and their copies.  I haven't thought of a way
// to see WHO is mutating it, just that it's getting mutated.
type defaultCacheMutationDetector struct {
	name   string
	period time.Duration

	// compareLock ensures only a single call to CompareObjects runs at a time
	compareObjectsLock sync.Mutex

	// addLock guards addedObjs between AddObject and CompareObjects
	addedObjsLock sync.Mutex
	addedObjs     []cacheObj

	cachedObjs []cacheObj

	retainDuration     time.Duration
	lastRotated        time.Time
	retainedCachedObjs []cacheObj

	// failureFunc is injectable for unit testing.  If you don't have it, the process will panic.
	// This panic is intentional, since turning on this detection indicates you want a strong
	// failure signal.  This failure is effectively a p0 bug and you can't trust process results
	// after a mutation anyway.
	failureFunc func(message string)
}

// cacheObj holds the actual object and a copy
type cacheObj struct {
	cached interface{}
	copied interface{}
}

func (d *defaultCacheMutationDetector) Run(stopCh <-chan struct{}) {
	// we DON'T want protection from panics.  If we're running this code, we want to die
	for {
		if d.lastRotated.IsZero() {
			d.lastRotated = time.Now()
		} else if time.Since(d.lastRotated) > d.retainDuration {
			d.retainedCachedObjs = d.cachedObjs
			d.cachedObjs = nil
			d.lastRotated = time.Now()
		}

		d.CompareObjects()

		select {
		case <-stopCh:
			return
		case <-time.After(d.period):
		}
	}
}

// AddObject makes a deep copy of the object for later comparison.  It only works on runtime.Object
// but that covers the vast majority of our cached objects
func (d *defaultCacheMutationDetector) AddObject(obj interface{}) {
	if _, ok := obj.(DeletedFinalStateUnknown); ok {
		return
	}
	if obj, ok := obj.(runtime.Object); ok {
		copiedObj := obj.DeepCopyObject()

		d.addedObjsLock.Lock()
		defer d.addedObjsLock.Unlock()
		d.addedObjs = append(d.addedObjs, cacheObj{cached: obj, copied: copiedObj})
	}
}

func (d *defaultCacheMutationDetector) CompareObjects() {
	d.compareObjectsLock.Lock()
	defer d.compareObjectsLock.Unlock()

	// move addedObjs into cachedObjs under lock
	// this keeps the critical section small to avoid blocking AddObject while we compare cachedObjs
	d.addedObjsLock.Lock()
	d.cachedObjs = append(d.cachedObjs, d.addedObjs...)
	d.addedObjs = nil
	d.addedObjsLock.Unlock()

	altered := false
	for i, obj := range d.cachedObjs {
		if !reflect.DeepEqual(obj.cached, obj.copied) {
			fmt.Printf("CACHE %s[%d] ALTERED!\n%v\n", d.name, i, diff.ObjectGoPrintSideBySide(obj.cached, obj.copied))
			altered = true
		}
	}
	for i, obj := range d.retainedCachedObjs {
		if !reflect.DeepEqual(obj.cached, obj.copied) {
			fmt.Printf("CACHE %s[%d] ALTERED!\n%v\n", d.name, i, diff.ObjectGoPrintSideBySide(obj.cached, obj.copied))
			altered = true
		}
	}

	if altered {
		msg := fmt.Sprintf("cache %s modified", d.name)
		if d.failureFunc != nil {
			d.failureFunc(msg)
			return
		}
		panic(msg)
	}
}
