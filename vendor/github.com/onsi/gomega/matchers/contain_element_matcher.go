// untested sections: 2

package matchers

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/matchers/internal/miter"
)

type ContainElementMatcher struct {
	Element any
	Result  []any
}

func (matcher *ContainElementMatcher) Match(actual any) (success bool, err error) {
	if !isArrayOrSlice(actual) && !isMap(actual) && !miter.IsIter(actual) {
		return false, fmt.Errorf("ContainElement matcher expects an array/slice/map/iterator.  Got:\n%s", format.Object(actual, 1))
	}

	var actualT reflect.Type
	var result reflect.Value
	switch numResultArgs := len(matcher.Result); {
	case numResultArgs > 1:
		return false, errors.New("ContainElement matcher expects at most a single optional pointer to store its findings at")
	case numResultArgs == 1:
		// Check the optional result arg to point to a single value/array/slice/map
		// of a type compatible with the actual value.
		if reflect.ValueOf(matcher.Result[0]).Kind() != reflect.Ptr {
			return false, fmt.Errorf("ContainElement matcher expects a non-nil pointer to store its findings at.  Got\n%s",
				format.Object(matcher.Result[0], 1))
		}
		actualT = reflect.TypeOf(actual)
		resultReference := matcher.Result[0]
		result = reflect.ValueOf(resultReference).Elem() // what ResultReference points to, to stash away our findings
		switch result.Kind() {
		case reflect.Array: // result arrays are not supported, as they cannot be dynamically sized.
			if miter.IsIter(actual) {
				_, actualvT := miter.IterKVTypes(actual)
				return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
					reflect.SliceOf(actualvT), result.Type().String())
			}
			return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
				reflect.SliceOf(actualT.Elem()).String(), result.Type().String())

		case reflect.Slice: // result slice
			// can we assign elements in actual to elements in what the result
			// arg points to?
			//   - ✔ actual is an array or slice
			//   - ✔ actual is an iter.Seq producing "v" elements
			//   - ✔ actual is an iter.Seq2 producing "v" elements, ignoring
			//     the "k" elements.
			switch {
			case isArrayOrSlice(actual):
				if !actualT.Elem().AssignableTo(result.Type().Elem()) {
					return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
						actualT.String(), result.Type().String())
				}

			case miter.IsIter(actual):
				_, actualvT := miter.IterKVTypes(actual)
				if !actualvT.AssignableTo(result.Type().Elem()) {
					return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
						actualvT.String(), result.Type().String())
				}

			default: // incompatible result reference
				return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
					reflect.MapOf(actualT.Key(), actualT.Elem()).String(), result.Type().String())
			}

		case reflect.Map: // result map
			// can we assign elements in actual to elements in what the result
			// arg points to?
			//   - ✔ actual is a map
			//   - ✔ actual is an iter.Seq2 (iter.Seq doesn't fit though)
			switch {
			case isMap(actual):
				if !actualT.AssignableTo(result.Type()) {
					return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
						actualT.String(), result.Type().String())
				}

			case miter.IsIter(actual):
				actualkT, actualvT := miter.IterKVTypes(actual)
				if actualkT == nil {
					return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
						reflect.SliceOf(actualvT).String(), result.Type().String())
				}
				if !reflect.MapOf(actualkT, actualvT).AssignableTo(result.Type()) {
					return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
						reflect.MapOf(actualkT, actualvT), result.Type().String())
				}

			default: // incompatible result reference
				return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
					actualT.String(), result.Type().String())
			}

		default:
			// can we assign a (single) element in actual to what the result arg
			// points to?
			switch {
			case miter.IsIter(actual):
				_, actualvT := miter.IterKVTypes(actual)
				if !actualvT.AssignableTo(result.Type()) {
					return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
						actualvT.String(), result.Type().String())
				}
			default:
				if !actualT.Elem().AssignableTo(result.Type()) {
					return false, fmt.Errorf("ContainElement cannot return findings.  Need *%s, got *%s",
						actualT.Elem().String(), result.Type().String())
				}
			}
		}
	}

	// If the supplied matcher isn't an Omega matcher, default to the Equal
	// matcher.
	elemMatcher, elementIsMatcher := matcher.Element.(omegaMatcher)
	if !elementIsMatcher {
		elemMatcher = &EqualMatcher{Expected: matcher.Element}
	}

	value := reflect.ValueOf(actual)

	var getFindings func() reflect.Value // abstracts how the findings are collected and stored
	var lastError error

	if !miter.IsIter(actual) {
		var valueAt func(int) any
		var foundAt func(int)
		// We're dealing with an array/slice/map, so in all cases we can iterate
		// over the elements in actual using indices (that can be considered
		// keys in case of maps).
		if isMap(actual) {
			keys := value.MapKeys()
			valueAt = func(i int) any {
				return value.MapIndex(keys[i]).Interface()
			}
			if result.Kind() != reflect.Invalid {
				fm := reflect.MakeMap(actualT)
				getFindings = func() reflect.Value { return fm }
				foundAt = func(i int) {
					fm.SetMapIndex(keys[i], value.MapIndex(keys[i]))
				}
			}
		} else {
			valueAt = func(i int) any {
				return value.Index(i).Interface()
			}
			if result.Kind() != reflect.Invalid {
				var fsl reflect.Value
				if result.Kind() == reflect.Slice {
					fsl = reflect.MakeSlice(result.Type(), 0, 0)
				} else {
					fsl = reflect.MakeSlice(reflect.SliceOf(result.Type()), 0, 0)
				}
				getFindings = func() reflect.Value { return fsl }
				foundAt = func(i int) {
					fsl = reflect.Append(fsl, value.Index(i))
				}
			}
		}

		for i := 0; i < value.Len(); i++ {
			elem := valueAt(i)
			success, err := elemMatcher.Match(elem)
			if err != nil {
				lastError = err
				continue
			}
			if success {
				if result.Kind() == reflect.Invalid {
					return true, nil
				}
				foundAt(i)
			}
		}
	} else {
		// We're dealing with an iterator as a first-class construct, so things
		// are slightly different: there is no index defined as in case of
		// arrays/slices/maps, just "ooooorder"
		var found func(k, v reflect.Value)
		if result.Kind() != reflect.Invalid {
			if result.Kind() == reflect.Map {
				fm := reflect.MakeMap(result.Type())
				getFindings = func() reflect.Value { return fm }
				found = func(k, v reflect.Value) { fm.SetMapIndex(k, v) }
			} else {
				var fsl reflect.Value
				if result.Kind() == reflect.Slice {
					fsl = reflect.MakeSlice(result.Type(), 0, 0)
				} else {
					fsl = reflect.MakeSlice(reflect.SliceOf(result.Type()), 0, 0)
				}
				getFindings = func() reflect.Value { return fsl }
				found = func(_, v reflect.Value) { fsl = reflect.Append(fsl, v) }
			}
		}

		success := false
		actualkT, _ := miter.IterKVTypes(actual)
		if actualkT == nil {
			miter.IterateV(actual, func(v reflect.Value) bool {
				var err error
				success, err = elemMatcher.Match(v.Interface())
				if err != nil {
					lastError = err
					return true // iterate on...
				}
				if success {
					if result.Kind() == reflect.Invalid {
						return false // a match and no result needed, so we're done
					}
					found(reflect.Value{}, v)
				}
				return true // iterate on...
			})
		} else {
			miter.IterateKV(actual, func(k, v reflect.Value) bool {
				var err error
				success, err = elemMatcher.Match(v.Interface())
				if err != nil {
					lastError = err
					return true // iterate on...
				}
				if success {
					if result.Kind() == reflect.Invalid {
						return false // a match and no result needed, so we're done
					}
					found(k, v)
				}
				return true // iterate on...
			})
		}
		if success && result.Kind() == reflect.Invalid {
			return true, nil
		}
	}

	// when the expectation isn't interested in the findings except for success
	// or non-success, then we're done here and return the last matcher error
	// seen, if any, as well as non-success.
	if result.Kind() == reflect.Invalid {
		return false, lastError
	}

	// pick up any findings the test is interested in as it specified a non-nil
	// result reference. However, the expectation always is that there are at
	// least one or multiple findings. So, if a result is expected, but we had
	// no findings, then this is an error.
	findings := getFindings()
	if findings.Len() == 0 {
		return false, lastError
	}

	// there's just a single finding and the result is neither a slice nor a map
	// (so it's a scalar): pick the one and only finding and return it in the
	// place the reference points to.
	if findings.Len() == 1 && !isArrayOrSlice(result.Interface()) && !isMap(result.Interface()) {
		if isMap(actual) {
			miter := findings.MapRange()
			miter.Next()
			result.Set(miter.Value())
		} else {
			result.Set(findings.Index(0))
		}
		return true, nil
	}

	// at least one or even multiple findings and a the result references a
	// slice or a map, so all we need to do is to store our findings where the
	// reference points to.
	if !findings.Type().AssignableTo(result.Type()) {
		return false, fmt.Errorf("ContainElement cannot return multiple findings.  Need *%s, got *%s",
			findings.Type().String(), result.Type().String())
	}
	result.Set(findings)
	return true, nil
}

func (matcher *ContainElementMatcher) FailureMessage(actual any) (message string) {
	return format.Message(actual, "to contain element matching", matcher.Element)
}

func (matcher *ContainElementMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to contain element matching", matcher.Element)
}
