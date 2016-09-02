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

package framework

import (
	"fmt"
	"os"
	"reflect"
	"strconv"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
	utildiff "k8s.io/kubernetes/pkg/util/diff"
)

var mutationDetectionEnabled = false

func init() {
	mutationDetectionEnabled, _ = strconv.ParseBool(os.Getenv("KUBE_CACHE_MUTATION_DETECTOR"))
}

type CacheMutationDetector interface {
	AddObject(obj interface{})
	Run(stopCh <-chan struct{})
	// conforms to
	CompareObjects()
}

func NewCacheMutationDetector(name string) CacheMutationDetector {
	if !mutationDetectionEnabled {
		return dummyMutationDetector{}
	}
	return &defaultCacheMutationDetector{name: name, period: 1 * time.Second}
}

type dummyMutationDetector struct{}

func (dummyMutationDetector) Run(stopCh <-chan struct{}) {
}
func (dummyMutationDetector) AddObject(obj interface{}) {
}
func (dummyMutationDetector) CompareObjects() {
}

// cacheMutationDetector gives a way to detect if a cached object has been mutated
// It has a list of cached objects and their copies.  I haven't thought of a way
// to see WHO is mutating it, just that its getting mutated
type defaultCacheMutationDetector struct {
	name   string
	period time.Duration

	lock        sync.Mutex
	cachedObjs  []cacheObj
	failureFunc func(message string)
}

// cacheObj holds the actual object and a copy
type cacheObj struct {
	cached interface{}
	copied interface{}
}

func (d *defaultCacheMutationDetector) Run(stopCh <-chan struct{}) {
	// we DON'T want protection from panics.  If we're running this code, we want to die
	go func() {
		for {
			d.CompareObjects()

			select {
			case <-stopCh:
				return
			case <-time.After(d.period):
			}
		}
	}()
}

// addObj makes a deep copy of the object for later comparison.  It only works on runtime.Object
// but that covers the vast majority of our cached objects
func (d *defaultCacheMutationDetector) AddObject(obj interface{}) {
	if _, ok := obj.(cache.DeletedFinalStateUnknown); ok {
		return
	}
	if _, ok := obj.(runtime.Object); !ok {
		return
	}

	copiedObj, err := api.Scheme.Copy(obj.(runtime.Object))
	if err != nil {
		return
	}

	d.lock.Lock()
	defer d.lock.Unlock()
	d.cachedObjs = append(d.cachedObjs, cacheObj{cached: obj, copied: copiedObj})
}

func (d *defaultCacheMutationDetector) CompareObjects() {
	d.lock.Lock()
	defer d.lock.Unlock()

	altered := false
	for i, obj := range d.cachedObjs {
		if !reflect.DeepEqual(obj.cached, obj.copied) {
			fmt.Printf("CACHE %s[%d] ALTERED!\n%v\n", d.name, i, utildiff.ObjectDiff(obj.cached, obj.copied))
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
