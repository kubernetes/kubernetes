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

// Tee works like Unix tee; it forwards all progress reports it receives to the
// specified sinks
func Tee(s1, s2 Sinker) Sinker {
	fn := func() chan<- Report {
		d1 := s1.Sink()
		d2 := s2.Sink()
		u := make(chan Report)
		go tee(u, d1, d2)
		return u
	}

	return SinkFunc(fn)
}

func tee(u <-chan Report, d1, d2 chan<- Report) {
	defer close(d1)
	defer close(d2)

	for r := range u {
		d1 <- r
		d2 <- r
	}
}
