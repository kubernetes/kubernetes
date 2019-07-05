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

import "fmt"

type prefixedReport struct {
	Report
	prefix string
}

func (r prefixedReport) Detail() string {
	if d := r.Report.Detail(); d != "" {
		return fmt.Sprintf("%s: %s", r.prefix, d)
	}

	return r.prefix
}

func prefixLoop(upstream <-chan Report, downstream chan<- Report, prefix string) {
	defer close(downstream)

	for r := range upstream {
		downstream <- prefixedReport{
			Report: r,
			prefix: prefix,
		}
	}
}

func Prefix(s Sinker, prefix string) Sinker {
	fn := func() chan<- Report {
		upstream := make(chan Report)
		downstream := s.Sink()
		go prefixLoop(upstream, downstream, prefix)
		return upstream
	}

	return SinkFunc(fn)
}
