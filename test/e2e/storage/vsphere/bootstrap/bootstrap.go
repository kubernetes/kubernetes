/*
Copyright 2018 The Kubernetes Authors.

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

package bootstrap

import (
	"sync"
)

var once sync.Once
var waiting = make(chan bool)

// Bootstrap takes care of initializing necessary test context for vSphere tests
func Bootstrap() {
	done := make(chan bool)
	go func() {
		once.Do(bootstrapOnce)
		<-waiting
		done <- true
	}()
	<-done
}

func bootstrapOnce() {
	// TBD
	// 1. Read vSphere conf and get VSphere instances
	// 2. Get Node to VSphere mapping
	// 3. Set NodeMapper in vSphere context
	TestContext = Context{}
	close(waiting)
}
