// Copyright 2014 Prometheus Team
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package procfs provides functions to retrieve system, kernel and process
// metrics from the pseudo-filesystem proc.
//
// Example:
//
//	package main
//
//	import (
//		"fmt"
//		"log"
//
//		"github.com/prometheus/procfs"
//	)
//
//	func main() {
//		p, err := procfs.Self()
//		if err != nil {
//			log.Fatalf("could not get process: %s", err)
//		}
//
//		stat, err := p.Stat()
//		if err != nil {
//			log.Fatalf("could not get process stat: %s", err)
//		}
//
//		fmt.Printf("command:  %s\n", stat.Comm)
//		fmt.Printf("cpu time: %fs\n", stat.CPUTime())
//		fmt.Printf("vsize:    %dB\n", stat.VirtualMemory())
//		fmt.Printf("rss:      %dB\n", stat.ResidentMemory())
//	}
package procfs
