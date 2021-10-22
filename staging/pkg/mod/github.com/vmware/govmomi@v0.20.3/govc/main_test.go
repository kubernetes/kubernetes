/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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
	"context"
	"flag"
	"testing"

	"github.com/vmware/govmomi/govc/cli"
)

func TestMain(t *testing.T) {
	// Execute flag registration for every command to verify there are no
	// commands with flag name collisions
	for _, cmd := range cli.Commands() {
		fs := flag.NewFlagSet("", flag.ContinueOnError)

		// Use fresh context for every command
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		cmd.Register(ctx, fs)
	}
}
