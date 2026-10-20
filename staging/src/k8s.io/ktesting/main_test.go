/*
Copyright 2024 The Kubernetes Authors.

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

package ktesting

import (
	"flag"
	"fmt"
	"os"
	"testing"

	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	// Bail out early when -help was given as parameter.
	flag.Parse()

	// Must be called *before* creating new goroutines.
	goleakOpts := []goleak.Option{
		goleak.IgnoreCurrent(),
	}

	result := m.Run()

	if err := goleak.Find(goleakOpts...); err != nil {
		fmt.Fprintf(os.Stderr, "leaked Goroutines: %v", err)
		os.Exit(1)
	}

	os.Exit(result)
}
