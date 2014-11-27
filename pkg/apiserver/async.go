/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// WorkFunc is used to perform any time consuming work for an api call, after
// the input has been validated. Pass one of these to MakeAsync to create an
// appropriate return value for the Update, Delete, and Create methods.
type WorkFunc func() (result runtime.Object, err error)

// MakeAsync takes a function and executes it, delivering the result in the way required
// by RESTStorage's Update, Delete, and Create methods.
func MakeAsync(fn WorkFunc) <-chan RESTResult {
	channel := make(chan RESTResult)
	go func() {
		defer util.HandleCrash()
		obj, err := fn()
		if err != nil {
			channel <- RESTResult{Object: errToAPIStatus(err)}
		} else {
			channel <- RESTResult{Object: obj}
		}
		// 'close' is used to signal that no further values will
		// be written to the channel. Not strictly necessary, but
		// also won't hurt.
		close(channel)
	}()
	return channel
}

// WorkFunc is used to perform any time consuming work for an api call, after
// the input has been validated. Pass one of these to MakeAsync to create an
// appropriate return value for the Update, Delete, and Create methods.
type WorkResultFunc func() (result RESTResult, err error)

// MakeAsync takes a function and executes it, delivering the result in the way required
// by RESTStorage's Update, Delete, and Create methods.
func MakeAsyncResult(fn WorkResultFunc) <-chan RESTResult {
	channel := make(chan RESTResult)
	go func() {
		defer util.HandleCrash()
		obj, err := fn()
		if err != nil {
			channel <- RESTResult{Object: errToAPIStatus(err)}
		} else {
			channel <- obj
		}
		// 'close' is used to signal that no further values will
		// be written to the channel. Not strictly necessary, but
		// also won't hurt.
		close(channel)
	}()
	return channel
}
