//go:build freebsd
// +build freebsd

/*
Copyright 2017 The Kubernetes Authors.

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

// Package freebsdtats provides a client to get node and pod level stats on freebsd
package freebsdstats

// NewPerfCounterClient creates a client using perf counters
func NewPerfCounterClient() (Client, error) {
	// Initialize the cache
	return newClient()
}