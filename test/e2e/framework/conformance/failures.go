/*
Copyright 2025 The Kubernetes Authors.

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

package architecture

import (
	"fmt"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	gtypes "github.com/onsi/gomega/types"
)

type gomegaFailures struct {
	failures []string
}

var _ gtypes.GomegaTestingT = &gomegaFailures{}

// Helper implements [gtyppes.GomegaTestingT].
func (g *gomegaFailures) Helper() {}

// Fatalf implements [gtypes.GomegaTestingT].
func (g *gomegaFailures) Fatalf(format string, args ...any) {
	g.Add(fmt.Sprintf(format, args...))
}

// Adds one failure.
func (g *gomegaFailures) Add(failure string) {
	if !strings.HasSuffix(failure, "\n") {
		failure += "\n"
	}
	g.failures = append(g.failures, failure)
}

// Check fails via [ginkgo.Fail] if there were any failures.
func (g *gomegaFailures) Check() {
	if len(g.failures) > 0 {
		ginkgo.GinkgoHelper()
		ginkgo.Fail(strings.Join(g.failures, "\n\n"))
	}
}

func (g *gomegaFailures) G() *gomega.WithT {
	return gomega.NewWithT(g)
}
