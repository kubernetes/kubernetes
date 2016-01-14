/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"flag"
	"github.com/onsi/ginkgo/config"
	"testing"
)

var e2e *E2ETests

func init() {
	//Ginkgo default configuration goes below.

	config.DefaultReporterConfig.Verbose = true
	// Turn on EmitSpecProgress to get spec progress (especially on interrupt)
	config.GinkgoConfig.EmitSpecProgress = true
	// Randomize specs as well as suites
	config.GinkgoConfig.RandomizeAllSpecs = true

	// Create new testing suite and parse the flags.
	e2e = NewE2ETests()
	flag.Parse()
}

func TestE2E(t *testing.T) {
	e2e.RunE2ETest(t)
}
