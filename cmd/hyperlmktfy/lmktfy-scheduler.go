/*
Copyright 2015 Google Inc. All rights reserved.

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

package main

import (
	scheduler "github.com/GoogleCloudPlatform/lmktfy/plugin/cmd/lmktfy-scheduler/app"
)

// NewScheduler creates a new hyperlmktfy Server object that includes the
// description and flags.
func NewScheduler() *Server {
	s := scheduler.NewSchedulerServer()

	hks := Server{
		SimpleUsage: "scheduler",
		Long:        "Implements a LMKTFY scheduler.  This will assign pods to lmktfylets based on capacity and constraints.",
		Run: func(_ *Server, args []string) error {
			return s.Run(args)
		},
	}
	s.AddFlags(hks.Flags())
	return &hks
}
