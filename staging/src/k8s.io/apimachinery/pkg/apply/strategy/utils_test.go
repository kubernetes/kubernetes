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

	"k8s.io/apimachinery/pkg/apply"
	"k8s.io/apimachinery/pkg/apply/parse"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kube-openapi/pkg/util/proto"
	tst "k8s.io/kube-openapi/pkg/util/proto/testing"
)

const (
	hasConflict = true
	noConflict  = false
)

var models = buildModelsOrDie(filepath.Join("api", "openapi-spec", "swagger.json"))
var deploymentAppsModel = models.LookupModel("io.k8s.api.apps.v1beta1.Deployment")
var deploymentModel = models.LookupModel("io.k8s.api.extensions.v1beta1.Deployment")

func buildModelsOrDie(path string) proto.Models {
	fake := tst.Fake{Path: path}
	doc, err := fake.OpenAPISchema()
	if err != nil {
		panic(fmt.Errorf("Error while trying to read OpenAPISchema: %v", err))
	}
	m, err := proto.NewOpenAPIData(doc)
	if err != nil {
		panic(fmt.Errorf("Error while trying to parse openapi: %v", err))
	}
	return m
}

// run parses the openapi and runs the tests
func run(instance apply.Strategy, recorded, local, remote, expected map[string]interface{}, schema proto.Schema) {
	parsed, err := parse.CreateElement(recorded, local, remote, schema)
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

func runConflictTest(instance apply.Strategy, recorded, local, remote map[string]interface{}, schema proto.Schema, isConflict bool) {
	parsed, err := parse.CreateElement(recorded, local, remote, schema)
	Expect(err).Should(Not(HaveOccurred()))

	merged, err := parsed.Merge(instance)
	if isConflict {
		Expect(err).Should(HaveOccurred())
	} else {
		Expect(err).ShouldNot(HaveOccurred())
		Expect(merged.Operation).Should(Equal(apply.SET))
	}
}
