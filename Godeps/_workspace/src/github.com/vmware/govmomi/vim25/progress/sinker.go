/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package progress

// Sinker defines what is expected of a type that can act as a sink for
// progress reports. The semantics are as follows. If you call Sink(), you are
// responsible for closing the returned channel. Closing this channel means
// that the related task is done, or resulted in error.
type Sinker interface {
	Sink() chan<- Report
}

// SinkFunc defines a function that returns a progress report channel.
type SinkFunc func() chan<- Report

// Sink makes the SinkFunc implement the Sinker interface.
func (fn SinkFunc) Sink() chan<- Report {
	return fn()
}
