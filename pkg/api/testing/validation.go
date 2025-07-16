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

package testing

import (
	"bytes"
	"sort"
	"strconv"
	"testing"

	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	runtimetest "k8s.io/apimachinery/pkg/runtime/testing"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// VerifyVersionedValidationEquivalence tests that all versions of an API return equivalent validation errors.
func VerifyVersionedValidationEquivalence(t *testing.T, obj, old k8sruntime.Object) {
	t.Helper()

	// Accumulate errors from all versioned validation, per version.
	all := map[string]field.ErrorList{}
	accumulate := func(t *testing.T, gv string, errs field.ErrorList) {
		all[gv] = errs
	}
	if old == nil {
		runtimetest.RunValidationForEachVersion(t, legacyscheme.Scheme, sets.Set[string]{}, obj, accumulate)
	} else {
		runtimetest.RunUpdateValidationForEachVersion(t, legacyscheme.Scheme, sets.Set[string]{}, obj, old, accumulate)
	}

	// Make a copy so we can modify it.
	other := map[string]field.ErrorList{}
	// Index for nicer output.
	keys := []string{}
	for k, v := range all {
		other[k] = v
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// Compare each lhs to each rhs.
	for _, lk := range keys {
		lv := all[lk]
		// remove lk since to prevent comparison to itself and because this
		// iteration will compare it to any version it has not yet been
		// compared to. e.g. [1, 2, 3] vs. [1, 2, 3] yields:
		//   1 vs. 2
		//   1 vs. 3
		//   2 vs. 3
		delete(other, lk)
		// don't compare to ourself
		for _, rk := range keys {
			rv, found := other[rk]
			if !found {
				continue // done already
			}
			if len(lv) != len(rv) {
				t.Errorf("different error count (%d vs. %d)\n%s: %v\n%s: %v", len(lv), len(rv), lk, fmtErrs(lv), rk, fmtErrs(rv))
				continue
			}
			next := false
			for i := range lv {
				if l, r := lv[i], rv[i]; l.Type != r.Type || l.Detail != r.Detail {
					t.Errorf("different errors\n%s: %v\n%s: %v", lk, fmtErrs(lv), rk, fmtErrs(rv))
					next = true
					break
				}
			}
			if next {
				continue
			}
		}
	}
}

// helper for nicer output
func fmtErrs(errs field.ErrorList) string {
	if len(errs) == 0 {
		return "<no errors>"
	}
	if len(errs) == 1 {
		return strconv.Quote(errs[0].Error())
	}
	buf := bytes.Buffer{}
	for _, e := range errs {
		buf.WriteString("\n")
		buf.WriteString(strconv.Quote(e.Error()))
	}

	return buf.String()
}
