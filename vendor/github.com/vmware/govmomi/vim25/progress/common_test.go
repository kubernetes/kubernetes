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

type dummySinker struct {
	ch chan Report
}

func (d dummySinker) Sink() chan<- Report {
	return d.ch
}

type dummyReport struct {
	p float32
	d string
	e error
}

func (p dummyReport) Percentage() float32 {
	return p.p
}

func (p dummyReport) Detail() string {
	return p.d
}

func (p dummyReport) Error() error {
	return p.e
}
