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

package fuzzer

import (
	"fmt"
	"math/rand"
	"sort"
	"strconv"
	"strings"

	"github.com/google/gofuzz"

	apitesting "k8s.io/apimachinery/pkg/api/apitesting"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
)

func genericFuzzerFuncs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(q *resource.Quantity, c fuzz.Continue) {
			*q = *resource.NewQuantity(c.Int63n(1000), resource.DecimalExponent)
		},
		func(j *int, c fuzz.Continue) {
			*j = int(c.Int31())
		},
		func(j **int, c fuzz.Continue) {
			if c.RandBool() {
				i := int(c.Int31())
				*j = &i
			} else {
				*j = nil
			}
		},
		func(j *runtime.TypeMeta, c fuzz.Continue) {
			// We have to customize the randomization of TypeMetas because their
			// APIVersion and Kind must remain blank in memory.
			j.APIVersion = ""
			j.Kind = ""
		},
		func(j *runtime.Object, c fuzz.Continue) {
			// TODO: uncomment when round trip starts from a versioned object
			if true { //c.RandBool() {
				*j = &runtime.Unknown{
					// We do not set TypeMeta here because it is not carried through a round trip
					Raw:         []byte(`{"apiVersion":"unknown.group/unknown","kind":"Something","someKey":"someValue"}`),
					ContentType: runtime.ContentTypeJSON,
				}
			} else {
				types := []runtime.Object{&metav1.Status{}, &metav1.APIGroup{}}
				t := types[c.Rand.Intn(len(types))]
				c.Fuzz(t)
				*j = t
			}
		},
		func(r *runtime.RawExtension, c fuzz.Continue) {
			// Pick an arbitrary type and fuzz it
			types := []runtime.Object{&metav1.Status{}, &metav1.APIGroup{}}
			obj := types[c.Rand.Intn(len(types))]
			c.Fuzz(obj)

			// Find a codec for converting the object to raw bytes.  This is necessary for the
			// api version and kind to be correctly set be serialization.
			var codec = apitesting.TestCodec(codecs, metav1.SchemeGroupVersion)

			// Convert the object to raw bytes
			bytes, err := runtime.Encode(codec, obj)
			if err != nil {
				panic(fmt.Sprintf("Failed to encode object: %v", err))
			}

			// strip trailing newlines which do not survive roundtrips
			for len(bytes) >= 1 && bytes[len(bytes)-1] == 10 {
				bytes = bytes[:len(bytes)-1]
			}

			// Set the bytes field on the RawExtension
			r.Raw = bytes
		},
	}
}

// taken from gofuzz internals for RandString
type charRange struct {
	first, last rune
}

func (c *charRange) choose(r *rand.Rand) rune {
	count := int64(c.last - c.first + 1)
	ch := c.first + rune(r.Int63n(count))

	return ch
}

// randomLabelPart produces a valid random label value or name-part
// of a label key.
func randomLabelPart(c fuzz.Continue, canBeEmpty bool) string {
	validStartEnd := []charRange{{'0', '9'}, {'a', 'z'}, {'A', 'Z'}}
	validMiddle := []charRange{{'0', '9'}, {'a', 'z'}, {'A', 'Z'},
		{'.', '.'}, {'-', '-'}, {'_', '_'}}

	partLen := c.Rand.Intn(64) // len is [0, 63]
	if !canBeEmpty {
		partLen = c.Rand.Intn(63) + 1 // len is [1, 63]
	}

	runes := make([]rune, partLen)
	if partLen == 0 {
		return string(runes)
	}

	runes[0] = validStartEnd[c.Rand.Intn(len(validStartEnd))].choose(c.Rand)
	for i := range runes[1:] {
		runes[i+1] = validMiddle[c.Rand.Intn(len(validMiddle))].choose(c.Rand)
	}
	runes[len(runes)-1] = validStartEnd[c.Rand.Intn(len(validStartEnd))].choose(c.Rand)

	return string(runes)
}

func randomDNSLabel(c fuzz.Continue) string {
	validStartEnd := []charRange{{'0', '9'}, {'a', 'z'}}
	validMiddle := []charRange{{'0', '9'}, {'a', 'z'}, {'-', '-'}}

	partLen := c.Rand.Intn(63) + 1 // len is [1, 63]
	runes := make([]rune, partLen)

	runes[0] = validStartEnd[c.Rand.Intn(len(validStartEnd))].choose(c.Rand)
	for i := range runes[1:] {
		runes[i+1] = validMiddle[c.Rand.Intn(len(validMiddle))].choose(c.Rand)
	}
	runes[len(runes)-1] = validStartEnd[c.Rand.Intn(len(validStartEnd))].choose(c.Rand)

	return string(runes)
}

func randomLabelKey(c fuzz.Continue) string {
	namePart := randomLabelPart(c, false)
	prefixPart := ""

	usePrefix := c.RandBool()
	if usePrefix {
		// we can fit, with dots, at most 3 labels in the 253 allotted characters
		prefixPartsLen := c.Rand.Intn(2) + 1
		prefixParts := make([]string, prefixPartsLen)
		for i := range prefixParts {
			prefixParts[i] = randomDNSLabel(c)
		}
		prefixPart = strings.Join(prefixParts, ".") + "/"
	}

	return prefixPart + namePart
}

func v1FuzzerFuncs(codecs runtimeserializer.CodecFactory) []interface{} {

	return []interface{}{
		func(j *metav1.TypeMeta, c fuzz.Continue) {
			// We have to customize the randomization of TypeMetas because their
			// APIVersion and Kind must remain blank in memory.
			j.APIVersion = ""
			j.Kind = ""
		},
		func(j *metav1.ObjectMeta, c fuzz.Continue) {
			c.FuzzNoCustom(j)

			j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
			j.UID = types.UID(c.RandString())

			var sec, nsec int64
			c.Fuzz(&sec)
			c.Fuzz(&nsec)
			j.CreationTimestamp = metav1.Unix(sec, nsec).Rfc3339Copy()

			if j.DeletionTimestamp != nil {
				c.Fuzz(&sec)
				c.Fuzz(&nsec)
				t := metav1.Unix(sec, nsec).Rfc3339Copy()
				j.DeletionTimestamp = &t
			}

			if len(j.Labels) == 0 {
				j.Labels = nil
			} else {
				delete(j.Labels, "")
			}
			if len(j.Annotations) == 0 {
				j.Annotations = nil
			} else {
				delete(j.Annotations, "")
			}
			if len(j.OwnerReferences) == 0 {
				j.OwnerReferences = nil
			}
			if len(j.Finalizers) == 0 {
				j.Finalizers = nil
			}
		},
		func(j *metav1.Initializers, c fuzz.Continue) {
			j = nil
		},
		func(j *metav1.ListMeta, c fuzz.Continue) {
			j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
			j.SelfLink = c.RandString()
		},
		func(j *metav1.LabelSelector, c fuzz.Continue) {
			c.FuzzNoCustom(j)
			// we can't have an entirely empty selector, so force
			// use of MatchExpression if necessary
			if len(j.MatchLabels) == 0 && len(j.MatchExpressions) == 0 {
				j.MatchExpressions = make([]metav1.LabelSelectorRequirement, c.Rand.Intn(2)+1)
			}

			if j.MatchLabels != nil {
				fuzzedMatchLabels := make(map[string]string, len(j.MatchLabels))
				for i := 0; i < len(j.MatchLabels); i++ {
					fuzzedMatchLabels[randomLabelKey(c)] = randomLabelPart(c, true)
				}
				j.MatchLabels = fuzzedMatchLabels
			}

			validOperators := []metav1.LabelSelectorOperator{
				metav1.LabelSelectorOpIn,
				metav1.LabelSelectorOpNotIn,
				metav1.LabelSelectorOpExists,
				metav1.LabelSelectorOpDoesNotExist,
			}

			if j.MatchExpressions != nil {
				// NB: the label selector parser code sorts match expressions by key, and sorts the values,
				// so we need to make sure ours are sorted as well here to preserve round-trip comparison.
				// In practice, not sorting doesn't hurt anything...

				for i := range j.MatchExpressions {
					req := metav1.LabelSelectorRequirement{}
					c.Fuzz(&req)
					req.Key = randomLabelKey(c)
					req.Operator = validOperators[c.Rand.Intn(len(validOperators))]
					if req.Operator == metav1.LabelSelectorOpIn || req.Operator == metav1.LabelSelectorOpNotIn {
						if len(req.Values) == 0 {
							// we must have some values here, so randomly choose a short length
							req.Values = make([]string, c.Rand.Intn(2)+1)
						}
						for i := range req.Values {
							req.Values[i] = randomLabelPart(c, true)
						}
						sort.Strings(req.Values)
					} else {
						req.Values = nil
					}
					j.MatchExpressions[i] = req
				}

				sort.Slice(j.MatchExpressions, func(a, b int) bool { return j.MatchExpressions[a].Key < j.MatchExpressions[b].Key })
			}
		},
		func(j *metav1.ManagedFieldsEntry, c fuzz.Continue) {
			c.FuzzNoCustom(j)
			if j.Fields != nil && len(j.Fields.Map) == 0 {
				j.Fields = nil
			}
		},
	}
}

func v1alpha1FuzzerFuncs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(r *metav1beta1.TableRow, c fuzz.Continue) {
			c.Fuzz(&r.Object)
			c.Fuzz(&r.Conditions)
			if len(r.Conditions) == 0 {
				r.Conditions = nil
			}
			n := c.Intn(10)
			if n > 0 {
				r.Cells = make([]interface{}, n)
			}
			for i := range r.Cells {
				t := c.Intn(6)
				switch t {
				case 0:
					r.Cells[i] = c.RandString()
				case 1:
					r.Cells[i] = c.Int63()
				case 2:
					r.Cells[i] = c.RandBool()
				case 3:
					x := map[string]interface{}{}
					for j := c.Intn(10) + 1; j >= 0; j-- {
						x[c.RandString()] = c.RandString()
					}
					r.Cells[i] = x
				case 4:
					x := make([]interface{}, c.Intn(10))
					for i := range x {
						x[i] = c.Int63()
					}
					r.Cells[i] = x
				default:
					r.Cells[i] = nil
				}
			}
		},
	}
}

var Funcs = fuzzer.MergeFuzzerFuncs(
	genericFuzzerFuncs,
	v1FuzzerFuncs,
	v1alpha1FuzzerFuncs,
)
