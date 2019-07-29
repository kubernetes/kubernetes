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

package strategy_test

import (
	. "github.com/onsi/gomega"

	"fmt"
	"path/filepath"
	"strings"

	"sigs.k8s.io/yaml"

	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubectl/pkg/apply"
	"k8s.io/kubectl/pkg/apply/parse"
	"k8s.io/kubectl/pkg/util/openapi"
	tst "k8s.io/kubectl/pkg/util/openapi/testing"
)

const (
	hasConflict = true
	noConflict  = false
)

var fakeResources = tst.NewFakeResources(filepath.Join("..", "..", "..", "test", "data", "openapi", "swagger.json"))

// run parses the openapi and runs the tests
func run(instance apply.Strategy, recorded, local, remote, expected map[string]interface{}) {
	runWith(instance, recorded, local, remote, expected, fakeResources)
}

func runWith(instance apply.Strategy, recorded, local, remote, expected map[string]interface{}, resources openapi.Resources) {
	parseFactory := parse.Factory{Resources: resources}

	parsed, err := parseFactory.CreateElement(recorded, local, remote)
	Expect(err).Should(Not(HaveOccurred()))

	merged, err := parsed.Merge(instance)
	Expect(err).ShouldNot(HaveOccurred())
	Expect(merged.Operation).Should(Equal(apply.SET))
	Expect(merged.MergedResult).Should(Equal(expected), diff.ObjectDiff(merged.MergedResult, expected))
}

// create parses the yaml string into a map[string]interface{}.  Verifies that the string does not have
// any tab characters.
func create(config string) map[string]interface{} {
	result := map[string]interface{}{}

	// The yaml parser will throw an obscure error if there are tabs in the yaml.  Check for this
	Expect(strings.Contains(config, "\t")).To(
		BeFalse(), fmt.Sprintf("Yaml %s cannot contain tabs", config))
	Expect(yaml.Unmarshal([]byte(config), &result)).Should(
		Not(HaveOccurred()), fmt.Sprintf("Could not parse config:\n\n%s\n", config))

	return result
}

func runConflictTest(instance apply.Strategy, recorded, local, remote map[string]interface{}, isConflict bool) {
	parseFactory := parse.Factory{Resources: fakeResources}
	parsed, err := parseFactory.CreateElement(recorded, local, remote)
	Expect(err).Should(Not(HaveOccurred()))

	merged, err := parsed.Merge(instance)
	if isConflict {
		Expect(err).Should(HaveOccurred())
	} else {
		Expect(err).ShouldNot(HaveOccurred())
		Expect(merged.Operation).Should(Equal(apply.SET))
	}
}
