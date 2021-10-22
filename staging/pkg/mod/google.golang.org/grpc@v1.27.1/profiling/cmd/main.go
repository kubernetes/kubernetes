/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Binary cmd is a command-line tool for profiling management. It retrieves and
// processes data from the profiling service.
package main

import (
	"os"

	"google.golang.org/grpc/grpclog"
	ppb "google.golang.org/grpc/profiling/proto"
)

type snapshot struct {
	StreamStats []*ppb.Stat
}

func main() {
	if err := parseArgs(); err != nil {
		grpclog.Errorf("error parsing flags: %v", err)
		os.Exit(1)
	}

	if *flagAddress != "" {
		if err := remoteCommand(); err != nil {
			grpclog.Errorf("error: %v", err)
			os.Exit(1)
		}
	} else {
		if err := localCommand(); err != nil {
			grpclog.Errorf("error: %v", err)
			os.Exit(1)
		}
	}
}
