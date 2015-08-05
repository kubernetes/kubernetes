/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package record

import (
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

// FakeRecorder is used as a fake during tests.
type FakeRecorder struct{}

func (f *FakeRecorder) Event(object runtime.Object, reason, message string) {}

func (f *FakeRecorder) Eventf(object runtime.Object, reason, messageFmt string, args ...interface{}) {}

func (f *FakeRecorder) PastEventf(object runtime.Object, timestamp util.Time, reason, messageFmt string, args ...interface{}) {
}
