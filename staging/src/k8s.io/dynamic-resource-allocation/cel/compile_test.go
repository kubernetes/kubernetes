/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

// We set MaxElements in AttributeType(cel.DeclType) to the larger of these two values
// when DRAListTypeAttributes feature gate is enabled because we cannot
// determine statically whether an attribute will be a scalar or a list at compile time.
// See newCompiler() for more details.
const maxElementsListTypeEnabled = uint64(max(resourceapi.DeviceAttributeMaxValueLength, resourceapi.ResourceSliceMaxAttributeValuesPerDevice))

var testcases = map[string]struct {
	// If nil, the testcase is executed with stored and new expressions.
	// If non-nil, it must point to environment.StoredExpressions or environment.NewExpressions
	// to select that kind of expression.
	envType *environment.Type
	// If nil, the testcase is executed with all combination of features,
	// otherwise only with the given combination.
	features *Features

	expression               string
	driver                   string
	allowMultipleAllocations *bool
	attributes               map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
	capacity                 map[resourceapi.QualifiedName]resourceapi.DeviceCapacity
	expectCompileError       string
	expectMatchError         string
	expectMatch              bool

	// There's no good way to verify that the cost of an expression
	// really is what it should be other than eye-balling it. The
	// cost should not change in the future unless the expression
	// gets changed, so this provides some protection against
	// regressions.
	expectCost uint64
}{
	"true": {
		expression:  "true",
		expectMatch: true,
	},
	"false": {
		expression:  "false",
		expectMatch: false,
	},
	"syntax-error": {
		expression:         "?!",
		expectCompileError: "Syntax error",
	},
	"type-error": {
		expression:         `1`,
		expectCompileError: "must evaluate to bool or the unknown type, not int",
	},
	"runtime-error-lookup-identifier": {
		expression:       `device.attributes["no-such-domain"].noSuchAttr.isGreaterThan(quantity("0"))`,
		expectMatchError: "no such key: noSuchAttr",
		expectCost:       6,
	},
	"runtime-error-lookup-map": {
		expression:       `device.attributes["no-such-domain"]["noSuchAttr"].isGreaterThan(quantity("0"))`,
		expectMatchError: "no such key: noSuchAttr",
		expectCost:       6,
	},
	"domain-check-negative": {
		expression:  `"no-such-domain" in device.attributes`,
		expectMatch: false,
		expectCost:  3,
	},
	"domain-check-positive": {
		expression:  `"dra.example.com" in device.attributes`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"dra.example.com/something": {BoolValue: ptr.To(true)}},
		expectMatch: true,
		expectCost:  3,
	},
	"empty-driver-name": {
		expression:  `device.driver == ""`,
		expectMatch: true,
		expectCost:  2,
	},
	"real-driver-name": {
		expression:  `device.driver == "dra.example.com"`,
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4,
	},
	"driver-name-qualifier": {
		expression:  `device.attributes["dra.example.com"].name`,
		driver:      "dra.example.com",
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {BoolValue: ptr.To(true)}},
		expectMatch: true,
		expectCost:  4,
	},
	"driver-name-qualifier-map-lookup": {
		expression:  `device.attributes["dra.example.com"]["name"]`,
		driver:      "dra.example.com",
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {BoolValue: ptr.To(true)}},
		expectMatch: true,
		expectCost:  4,
	},
	"bind": {
		expression:  `cel.bind(dra, device.attributes["dra.example.com"], dra.name)`,
		driver:      "dra.example.com",
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {BoolValue: ptr.To(true)}},
		expectMatch: true,
		expectCost:  15,
	},
	"qualified-attribute-name": {
		expression:  `device.attributes["other.example.com"].name`,
		driver:      "dra.example.com",
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"other.example.com/name": {BoolValue: ptr.To(true)}},
		expectMatch: true,
		expectCost:  4,
	},
	"bool": {
		expression:  `device.attributes["dra.example.com"].name`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {BoolValue: ptr.To(true)}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4,
	},
	"int": {
		expression:  `device.attributes["dra.example.com"].name > 0`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {IntValue: ptr.To(int64(1))}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  5,
	},
	"string": {
		expression:  `device.attributes["dra.example.com"].name == "fish"`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {StringValue: ptr.To("fish")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  5,
	},
	"macro-compilation-error-new-expressions": {
		features:           &Features{EnableListTypeAttributes: false},
		envType:            new(environment.NewExpressions),
		expression:         `device.attributes["dra.example.com"].ids.exists(id, id == 1)`,
		driver:             "dra.example.com",
		expectCompileError: "compilation failed",
	},
	"macro-exists-on-int": {
		features:         &Features{EnableListTypeAttributes: true},
		expression:       `device.attributes["dra.example.com"].name.exists(x, x > 0)`,
		attributes:       map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {IntValue: new(int64(1))}},
		driver:           "dra.example.com",
		expectMatchError: "got 'types.Int', expected iterable type",
		expectCost:       5 + ((3 + 3) * maxElementsListTypeEnabled /* (cost(loopCondition=="not_strictly_false(!accu)") + cost(loopStep=="accu && (x > 0)")) * maxElementsListTypeEnabled */),
	},
	"macro-exists-on-bool": {
		features:         &Features{EnableListTypeAttributes: true},
		expression:       `device.attributes["dra.example.com"].name.exists(x, x == true)`,
		attributes:       map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {BoolValue: new(true)}},
		driver:           "dra.example.com",
		expectMatchError: "got 'types.Bool', expected iterable type",
		expectCost:       5 + ((3 + 3) * maxElementsListTypeEnabled /* (cost(loopCondition=="not_strictly_false(!accu)") + cost(loopStep=="accu && (x > 0)")) * maxElementsListTypeEnabled */),
	},
	"macro-exists-on-string": {
		features:         &Features{EnableListTypeAttributes: true},
		expression:       `device.attributes["dra.example.com"].name.exists(x, x == "fish")`,
		attributes:       map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {StringValue: new("fish")}},
		driver:           "dra.example.com",
		expectMatchError: "got 'types.String', expected iterable type",
		expectCost:       5 + ((3 + 3) * maxElementsListTypeEnabled /* (cost(loopCondition=="not_strictly_false(!accu)") + cost(loopStep=="accu && (x > 0)")) * maxElementsListTypeEnabled */),
	},
	"list-result-type-causes-match-error": {
		expression:       `device.attributes["dra.example.com"].names`,
		attributes:       map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"names": {StringValues: []string{"fish", "bird"}}},
		driver:           "dra.example.com",
		expectMatchError: "CEL result of type list could not be converted to bool",
		expectCost:       4,
	},
	"includes-function-can-evaluate": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes("fish")`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {StringValues: []string{"fish", "bird"}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"list-of-bool": {
		expression:  `device.attributes["dra.example.com"].names.size() == 2`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"names": {BoolValues: []bool{true, false}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  6,
	},
	"list-of-int": {
		expression:  `device.attributes["dra.example.com"].names.size() == 2`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"names": {IntValues: []int64{1, 2}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  6,
	},
	"list-of-string": {
		expression:  `device.attributes["dra.example.com"].names.size() == 2`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"names": {StringValues: []string{"fish", "bird"}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  6,
	},
	"list-of-semver": {
		expression:  `device.attributes["dra.example.com"].names.size() == 2`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"names": {VersionValues: []string{"1.0.0", "2.0.0"}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  6,
	},
	"macro-on-list-of-int": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].names.exists(x, x > 0)`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"names": {IntValues: []int64{1, 2}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  5 + ((3 + 3) * maxElementsListTypeEnabled /* (cost(loopCondition=="not_strictly_false(!accu)") + cost(loopStep=="accu && (x > 0)")) * maxElementsListTypeEnabled */),
	},
	"macro-on-list-of-string": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].names.all(x, x != "")`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"names": {StringValues: []string{"fish", "bird"}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  5 + ((2 + 2) * maxElementsListTypeEnabled /* (cost(loopCondition=="not_strictly_false(accu)") + cost(loopStep=="accu && (x != "")")) * maxElementsListTypeEnabled */),
	},
	"macro-on-list-of-semver": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].names.all(x, x.isGreaterThan(semver("0.0.1")))`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"names": {VersionValues: []string{"1.0.0", "2.0.0"}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  5 + ((2 + 4) * maxElementsListTypeEnabled /* (cost(loopCondition=="not_strictly_false(accu)") + cost(loopStep=="accu && (x.isGreaterThan(semver("0.0.1"))")) * maxElementsListTypeEnabled */),
	},
	"includes-function-compilation-error": {
		features:           &Features{EnableListTypeAttributes: false},
		envType:            ptr.To(environment.NewExpressions),
		expression:         `device.attributes["dra.example.com"].attr.includes("fish")`,
		driver:             "dra.example.com",
		expectCompileError: "undeclared reference to 'includes'",
	},
	"includes-function-compilation-and-eval-success": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].attr.includes("fish")`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"attr": {StringValue: new("fish")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-bool-scalar-positive": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(true)`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {BoolValue: new(true)}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-bool-scalar-negative": {
		features:    &Features{EnableListTypeAttributes: true},
		envType:     ptr.To(environment.NewExpressions),
		expression:  `device.attributes["dra.example.com"].name.includes(true)`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {BoolValue: new(false)}},
		driver:      "dra.example.com",
		expectMatch: false,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-bool-list-positive": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(true)`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {BoolValues: []bool{true, false}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-bool-list-negative": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(false)`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {BoolValues: []bool{true, true}}},
		driver:      "dra.example.com",
		expectMatch: false,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-int-scalar-positive": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(1)`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {IntValue: new(int64(1))}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-int-scalar-negative": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(1)`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {IntValue: new(int64(2))}},
		driver:      "dra.example.com",
		expectMatch: false,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-int-list-positive": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(1)`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {IntValues: []int64{1, 2}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-int-list-negative": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(3)`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {IntValues: []int64{1, 2}}},
		driver:      "dra.example.com",
		expectMatch: false,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-string-scalar-positive": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes("fish")`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {StringValue: new("fish")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-string-scalar-negative": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes("bird")`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {StringValue: new("fish")}},
		driver:      "dra.example.com",
		expectMatch: false,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-string-list-positive": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes("fish")`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {StringValues: []string{"fish", "bird"}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-string-list-negative": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes("cat")`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {StringValues: []string{"fish", "bird"}}},
		driver:      "dra.example.com",
		expectMatch: false,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-semver-scalar-positive": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(semver("1.0.0"))`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {VersionValue: new("1.0.0")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 48 /* cost of "includes" is max list length */ + 1,
	},
	"includes-function-on-semver-scalar-negative": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(semver("2.0.0"))`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {VersionValue: new("1.0.0")}},
		driver:      "dra.example.com",
		expectMatch: false,
		expectCost:  4 + 48 /* cost of "includes" is max list length */ + 1,
	},
	"includes-function-on-semver-list-positive": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(semver("1.0.0"))`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {VersionValues: []string{"1.0.0", "2.0.0"}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 48 /* cost of "includes" is max list length */ + 1,
	},
	"includes-function-on-semver-list-negative": {
		features:    &Features{EnableListTypeAttributes: true},
		expression:  `device.attributes["dra.example.com"].name.includes(semver("3.0.0"))`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {VersionValues: []string{"1.0.0", "2.0.0"}}},
		driver:      "dra.example.com",
		expectMatch: false,
		expectCost:  4 + 48 /* cost of "includes" is max list length */ + 1,
	},
	"includes-function-on-very-long-list-positive": {
		features:   &Features{EnableListTypeAttributes: true},
		expression: fmt.Sprintf(`device.attributes["dra.example.com"].name.includes("value-%d")`, resourceapi.ResourceSliceMaxAttributeValuesPerDevice),
		attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {StringValues: func() []string {
			values := make([]string, resourceapi.ResourceSliceMaxAttributeValuesPerDevice)
			for i := range values {
				values[i] = fmt.Sprintf("value-%d", i)
			}
			return values
		}()}},
		driver:      "dra.example.com",
		expectMatch: false,
		expectCost:  4 + 48, /* cost of "includes" is max list length */
	},
	"includes-function-on-very-long-list-runtime-error": {
		features:   &Features{EnableListTypeAttributes: true},
		expression: fmt.Sprintf(`device.attributes["dra.example.com"].name.includes("value-%d")`, resourceapi.ResourceSliceMaxAttributeValuesPerDevice+1),
		attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {StringValues: func() []string {
			values := make([]string, resourceapi.ResourceSliceMaxAttributeValuesPerDevice+1)
			for i := range values {
				values[i] = fmt.Sprintf("value-%d", i)
			}
			return values
		}()}},
		driver:           "dra.example.com",
		expectMatchError: fmt.Sprintf("'includes' function cannot be applied to lists longer than %d values", resourceapi.ResourceSliceMaxAttributeValuesPerDevice),
		expectCost:       4 + 48, /* cost of "includes" is max list length */
	},
	"in-operator-on-list": {
		// This case is for documenting purpose to present the difference of call cost estimation
		// between "in" operator and "includes" function.
		// The cost estimation of "includes" is based on resourceapi.ResourceSliceMaxAttributeValuesPerDevice
		// because it's designed for checking whether a value is included in an device attribute list.
		// Instead, the cost estimation of "in" operator is based on maxElementsListTypeEnabled
		// (MaxElements in AttributeType(cel.DeclType)) as this operator is CEL standard one.
		expression:  `1 in device.attributes["dra.example.com"].names`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"names": {IntValues: []int64{1, 2, 3}}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  4 + 64, /* cost of "in" is maxElementsListTypeEnabled*/
	},
	"version": {
		expression:  `device.attributes["dra.example.com"].name.isGreaterThan(semver("0.0.1"))`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {VersionValue: ptr.To("1.0.0")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  6,
	},
	"version_v": {
		// Relaxed parsing with v prefix.
		expression:  `device.attributes["dra.example.com"].name.isGreaterThan(semver("v0.0.1", true)) && isSemver("v1.0.0", true)`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {VersionValue: ptr.To("1.0.0")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  7,
	},
	"macro-exists-on-version": {
		features:         &Features{EnableListTypeAttributes: true},
		expression:       `device.attributes["dra.example.com"].name.exists(x, x.isGreaterThan(semver("0.0.1")))`,
		attributes:       map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"name": {VersionValue: new("1.0.0")}},
		driver:           "dra.example.com",
		expectMatchError: "got 'cel.Semver', expected iterable type",
		expectCost:       5 + ((3 + 4) * maxElementsListTypeEnabled /* (cost(loopCondition=="not_strictly_false(!accu)") + cost(loopStep=="accu && (x.isGreaterThan(semver("0.0.1"))")) * maxElementsListTypeEnabled */),
	},
	"quantity": {
		expression:  `device.capacity["dra.example.com"].name.isGreaterThan(quantity("1Ki"))`,
		capacity:    map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{"name": {Value: resource.MustParse("1Mi")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  6,
	},
	"check-positive": {
		expression:  `"name" in device.capacity["dra.example.com"] && device.capacity["dra.example.com"].name.isGreaterThan(quantity("1Ki"))`,
		capacity:    map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{"name": {Value: resource.MustParse("1Mi")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  10,
	},
	"check-negative": {
		expression:  `!("name" in device.capacity["dra.example.com"]) || device.capacity["dra.example.com"].name.isGreaterThan(quantity("1Ki"))`,
		expectMatch: true,
		expectCost:  11,
	},
	"all": {
		expression: `
device.capacity["dra.example.com"].quantity.isGreaterThan(quantity("1Ki")) &&
device.attributes["dra.example.com"].bool &&
device.attributes["dra.example.com"].int > 0 &&
device.attributes["dra.example.com"].string == "fish" &&
device.attributes["dra.example.com"].version.isGreaterThan(semver("0.0.1")) &&
device.capacity["dra.example.com"]["quantity"].isGreaterThan(quantity("1Ki")) &&
device.attributes["dra.example.com"]["bool"] &&
device.attributes["dra.example.com"]["int"] > 0 &&
device.attributes["dra.example.com"]["string"] == "fish" &&
device.attributes["dra.example.com"]["version"].isGreaterThan(semver("0.0.1"))
`,

		attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"bool":    {BoolValue: ptr.To(true)},
			"int":     {IntValue: ptr.To(int64(1))},
			"string":  {StringValue: ptr.To("fish")},
			"version": {VersionValue: ptr.To("1.0.0")},
		},
		capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"quantity": {Value: resource.MustParse("1Mi")},
		},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  52,
	},
	"many": {
		expression: `device.attributes["dra.example.com"].a && device.attributes["dra.example.com"].b && device.attributes["dra.example.com"].c`,
		attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"a": {BoolValue: ptr.To(true)},
			"b": {BoolValue: ptr.To(true)},
			"c": {BoolValue: ptr.To(true)},
		},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  12,
	},
	"check_attribute_domains": {
		expression:  `device.attributes.exists_one(x, x == "dra.example.com")`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"services": {StringValue: ptr.To("some_example_value")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  164,
	},
	"check_attribute_ids": {
		expression:  `device.attributes["dra.example.com"].exists_one(x, x == "services")`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"services": {StringValue: ptr.To("some_example_value")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  133,
	},
	"split_attribute": {
		expression:  `device.attributes["dra.example.com"].services.split("example").size() >= 2`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"services": {StringValue: ptr.To("some_example_value")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  19,
	},
	"regexp_attribute": {
		expression:  `device.attributes["dra.example.com"].services.matches("[^a]?sym")`,
		attributes:  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"services": {StringValue: ptr.To("asymetric")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  18,
	},
	"check_capacity_domains": {
		expression:  `device.capacity.exists_one(x, x == "dra.example.com")`,
		capacity:    map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{"memory": {Value: resource.MustParse("1Mi")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  164,
	},
	"check_capacity_ids": {
		expression:  `device.capacity["dra.example.com"].exists_one(x, x == "memory")`,
		capacity:    map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{"memory": {Value: resource.MustParse("1Mi")}},
		driver:      "dra.example.com",
		expectMatch: true,
		expectCost:  133,
	},
	"expensive": {
		// The worst-case is based on the maximum number of
		// attributes and the maximum attribute name length.
		// To actually reach that expected cost at runtime, we must
		// have many attributes.
		attributes: func() map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
			attributes := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
			prefix := "dra.example.com/"
			attribute := resourceapi.DeviceAttribute{
				StringValue: ptr.To("abc"),
			}
			for i := 0; i < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; i++ {
				suffix := fmt.Sprintf("-%d", i)
				name := prefix + strings.Repeat("x", resourceapi.DeviceMaxIDLength-len(suffix)) + suffix
				attributes[resourceapi.QualifiedName(name)] = attribute
			}
			return attributes
		}(),
		// From https://github.com/kubernetes/kubernetes/blob/50fc400f178d2078d0ca46aee955ee26375fc437/test/integration/apiserver/cel/validatingadmissionpolicy_test.go#L2150.
		expression:       `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(x, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(y, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z5, int('1'.find('[0-9]*')) < 100)))))))`,
		driver:           "dra.example.com",
		expectMatchError: "actual cost limit exceeded",
		expectCost:       85555551, // Exceed limit!
	},
	"allow_multiple_allocations": {
		features:                 &Features{EnableConsumableCapacity: true},
		expression:               `device.allowMultipleAllocations == true`,
		allowMultipleAllocations: ptr.To(true),
		driver:                   "dra.example.com",
		expectMatch:              true,
		expectCost:               3,
	},
	"allow_multiple_allocations_default": {
		features:                 &Features{EnableConsumableCapacity: true},
		expression:               `device.allowMultipleAllocations == false`,
		allowMultipleAllocations: nil,
		driver:                   "dra.example.com",
		expectMatch:              true,
		expectCost:               3,
	},
	"allow_multiple_allocations_false": {
		features:                 &Features{EnableConsumableCapacity: true},
		expression:               `device.allowMultipleAllocations == false`,
		allowMultipleAllocations: ptr.To(false),
		driver:                   "dra.example.com",
		expectMatch:              true,
		expectCost:               3,
	},
	"allow_multiple_allocations_new": {
		features:                 &Features{EnableConsumableCapacity: true},
		expression:               `device.allowMultipleAllocations == false`,
		allowMultipleAllocations: ptr.To(false),
		driver:                   "dra.example.com",
		expectMatch:              true,
		expectCost:               3,
	},
	"allow_multiple_allocations_enabled": {
		features:                 &Features{EnableConsumableCapacity: true},
		expression:               `device.allowMultipleAllocations == false`,
		allowMultipleAllocations: ptr.To(false),
		driver:                   "dra.example.com",
		expectMatch:              true,
		expectCost:               3,
	},
	"allow_multiple_allocations_disabled": {
		envType:            ptr.To(environment.NewExpressions),
		features:           &Features{EnableConsumableCapacity: false},
		expression:         `device.allowMultipleAllocations == false`,
		driver:             "dra.example.com",
		expectCompileError: `undefined field 'allowMultipleAllocations'`,
	},
}

func TestCEL(t *testing.T) {
	for name, scenario := range testcases {
		run := func(t *testing.T, features Features, envType environment.Type) {
			_, ctx := ktesting.NewTestContext(t)
			result := GetCompiler(features).CompileCELExpression(scenario.expression, Options{EnvType: &envType})
			if scenario.expectCompileError != "" && result.Error == nil {
				t.Fatalf("FAILURE: expected compile error %q, got none", scenario.expectCompileError)
			}
			if result.Error != nil {
				if scenario.expectCompileError == "" {
					t.Fatalf("FAILURE: unexpected compile error: %v", result.Error)
				}
				if !strings.Contains(result.Error.Error(), scenario.expectCompileError) {
					t.Fatalf("FAILURE: expected compile error to contain %q, but got instead: %v", scenario.expectCompileError, result.Error)
				}
				return
			}
			if scenario.expectCompileError != "" {
				t.Fatalf("FAILURE: expected compile error %q, got none", scenario.expectCompileError)
			}
			if expect, actual := scenario.expectCost, result.MaxCost; expect != actual {
				t.Errorf("ERROR: expected CEL cost %d, got %d instead (%.0f%% of limit %d)", expect, actual, float64(actual)*100.0/float64(resourceapi.CELSelectorExpressionMaxCost), resourceapi.CELSelectorExpressionMaxCost)
			}

			match, details, err := result.DeviceMatches(ctx, Device{
				AllowMultipleAllocations: scenario.allowMultipleAllocations, Attributes: scenario.attributes, Capacity: scenario.capacity, Driver: scenario.driver,
			})
			// details.ActualCost can be called for nil details, no need to check.
			actualCost := ptr.Deref(details.ActualCost(), 0)
			if scenario.expectCost > 0 {
				t.Logf("actual cost %d, %d%% of worst-case estimate %d", actualCost, actualCost*100/scenario.expectCost, scenario.expectCost)
			} else {
				t.Logf("actual cost %d, expected zero costs", actualCost)
			}
			if actualCost > result.MaxCost {
				t.Errorf("ERROR: cost estimate %d underestimated the evaluation cost of %d", result.MaxCost, actualCost)
			}

			if err != nil {
				if scenario.expectMatchError == "" {
					t.Fatalf("FAILURE: unexpected evaluation error: %v", err)
				}
				if !strings.Contains(err.Error(), scenario.expectMatchError) {
					t.Fatalf("FAILURE: expected evaluation error to contain %q, but got instead: %v", scenario.expectMatchError, err)
				}
				return
			}
			if scenario.expectMatchError != "" {
				t.Fatalf("FAILURE: expected match error %q, got none", scenario.expectMatchError)
			}
			if match != scenario.expectMatch {
				t.Fatalf("FAILURE: expected result %v, got %v", scenario.expectMatch, match)
			}
		}

		t.Run(name, func(t *testing.T) {
			stateListType := []bool{true, false}
			stateConsumable := []bool{true, false}
			if scenario.features != nil {
				stateConsumable = []bool{scenario.features.EnableConsumableCapacity}
				stateListType = []bool{scenario.features.EnableListTypeAttributes}
			}
			envTypes := []environment.Type{environment.NewExpressions, environment.StoredExpressions}
			if scenario.envType != nil {
				envTypes = []environment.Type{*scenario.envType}
			}

			var features Features
			for _, features.EnableListTypeAttributes = range stateListType {
				t.Run(fmt.Sprintf("list-type-attributes=%v", features.EnableListTypeAttributes), func(t *testing.T) {
					for _, features.EnableConsumableCapacity = range stateConsumable {
						t.Run(fmt.Sprintf("consumable-capacity=%v", features.EnableConsumableCapacity), func(t *testing.T) {
							for _, envType := range envTypes {
								t.Run(string(envType), func(t *testing.T) {
									run(t, features, envType)
								})
							}
						})
					}
				})
			}
		})
	}
}

func TestInterrupt(t *testing.T) {
	for _, name := range []string{"timeout", "deadline", "cancel"} {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			// Adapted from https://github.com/kubernetes/kubernetes/blob/e0859f91b7d269bb7e2f43e23d202ccccaf34c0c/staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/validation_test.go#L3006
			expression := `device.attributes["dra.example.com"].map(key, device.attributes["dra.example.com"][key] * 20).filter(e, e > 50).exists(e, e == 60)`
			result := GetCompiler(Features{}).CompileCELExpression(expression, Options{})
			if result.Error != nil {
				t.Fatalf("unexpected compile error: %v", result.Error)
			}
			switch name {
			case "timeout":
				c, cancel := context.WithTimeout(ctx, time.Nanosecond)
				defer cancel()
				ctx = c
			case "deadline":
				c, cancel := context.WithDeadline(ctx, time.Now())
				defer cancel()
				ctx = c
			case "cancel":
				c, cancel := context.WithCancel(ctx)
				cancel()
				ctx = c
			}
			device := Device{
				Attributes: make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute),
			}
			for i := int64(0); i < 1000; i++ {
				device.Attributes[resourceapi.QualifiedName(fmt.Sprintf("dra.example.com/attr%d", i))] = resourceapi.DeviceAttribute{
					IntValue: ptr.To(i),
				}
			}
			_, _, err := result.DeviceMatches(ctx, device)
			if ctx.Err() != nil {
				if !errors.Is(err, ctx.Err()) {
					t.Fatalf("expected %v, got error: %v", ctx.Err(), err)
				}
			} else {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
			}
		})
	}
}

func BenchmarkDeviceMatches(b *testing.B) {
	for name, scenario := range testcases {
		if scenario.expectCompileError != "" {
			continue
		}

		run := func(b *testing.B, features Features, envType environment.Type) {
			_, ctx := ktesting.NewTestContext(b)
			result := GetCompiler(features).CompileCELExpression(scenario.expression, Options{EnvType: &envType})
			if result.Error != nil {
				b.Fatalf("unexpected compile error: %s", result.Error.Error())
			}

			for i := 0; i < b.N; i++ {
				// It would be nice to measure
				// time/actual_cost, but the time as observed
				// here also includes additional preparations
				// in result.DeviceMatches and thus cannot be
				// used.
				match, _, err := result.DeviceMatches(ctx, Device{
					AllowMultipleAllocations: scenario.allowMultipleAllocations, Attributes: scenario.attributes, Capacity: scenario.capacity, Driver: scenario.driver,
				})
				if err != nil {
					if scenario.expectMatchError == "" {
						b.Fatalf("unexpected evaluation error: %v", err)
					}
					if !strings.Contains(err.Error(), scenario.expectMatchError) {
						b.Fatalf("expected evaluation error to contain %q, but got instead: %v", scenario.expectMatchError, err)
					}
					return
				}
				if scenario.expectMatchError != "" {
					b.Fatalf("expected match error %q, got none", scenario.expectMatchError)
				}
				if match != scenario.expectMatch {
					b.Fatalf("expected result %v, got %v", scenario.expectMatch, match)
				}
			}
		}

		b.Run(name, func(b *testing.B) {
			// Performance shouldn't depend on feature enablement, so don't bother testing all variants.
			run(b, Features{ /* ignored because of stored expression */ }, environment.StoredExpressions)
		})
	}
}
