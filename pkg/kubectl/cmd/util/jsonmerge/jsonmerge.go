package jsonmerge

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/evanphx/json-patch"
	"github.com/golang/glog"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/yaml"
)

// Delta represents a change between two JSON documents.
type Delta struct {
	original []byte
	edit     []byte

	preconditions []PreconditionFunc
}

// PreconditionFunc is a test to verify that an incompatible change
// has occurred before an Apply can be successful.
type PreconditionFunc func(interface{}) bool

// AddPreconditions adds precondition checks to a change which must
// be satisfied before an Apply is considered successful. If a
// precondition returns false, the Apply is failed with
// ErrPreconditionFailed.
func (d *Delta) AddPreconditions(fns ...PreconditionFunc) {
	d.preconditions = append(d.preconditions, fns...)
}

// RequireKeyUnchanged creates a precondition function that fails
// if the provided key is present in the diff (indicating its value
// has changed).
func RequireKeyUnchanged(key string) PreconditionFunc {
	return func(diff interface{}) bool {
		m, ok := diff.(map[string]interface{})
		if !ok {
			return true
		}
		// the presence of key in a diff means that its value has been changed, therefore
		// we should fail the precondition.
		_, ok = m[key]
		return !ok
	}
}

// NewDelta accepts two JSON or YAML documents and calculates the difference
// between them.  It returns a Delta object which can be used to resolve
// conflicts against a third version with a common parent, or an error
// if either document is in error.
func NewDelta(from, to []byte) (*Delta, error) {
	d := &Delta{}
	before, err := yaml.ToJSON(from)
	if err != nil {
		return nil, err
	}
	after, err := yaml.ToJSON(to)
	if err != nil {
		return nil, err
	}
	diff, err := jsonpatch.CreateMergePatch(before, after)
	if err != nil {
		return nil, err
	}
	glog.V(6).Infof("Patch created from:\n%s\n%s\n%s", string(before), string(after), string(diff))
	d.original = before
	d.edit = diff
	return d, nil
}

// Apply attempts to apply the changes described by Delta onto latest,
// returning an error if the changes cannot be applied cleanly.
// IsConflicting will be true if the changes overlap, otherwise a
// generic error will be returned.
func (d *Delta) Apply(latest []byte) ([]byte, error) {
	base, err := yaml.ToJSON(latest)
	if err != nil {
		return nil, err
	}
	changes, err := jsonpatch.CreateMergePatch(d.original, base)
	if err != nil {
		return nil, err
	}
	diff1 := make(map[string]interface{})
	if err := json.Unmarshal(d.edit, &diff1); err != nil {
		return nil, err
	}
	diff2 := make(map[string]interface{})
	if err := json.Unmarshal(changes, &diff2); err != nil {
		return nil, err
	}
	for _, fn := range d.preconditions {
		if !fn(diff1) || !fn(diff2) {
			return nil, ErrPreconditionFailed
		}
	}

	glog.V(6).Infof("Testing for conflict between:\n%s\n%s", string(d.edit), string(changes))
	if hasConflicts(diff1, diff2) {
		return nil, ErrConflict
	}
	return jsonpatch.MergePatch(base, d.edit)
}

// IsConflicting returns true if the provided error indicates a
// conflict exists between the original changes and the applied
// changes.
func IsConflicting(err error) bool {
	return err == ErrConflict
}

// IsPreconditionFailed returns true if the provided error indicates
// a Delta precondition did not succeed.
func IsPreconditionFailed(err error) bool {
	return err == ErrPreconditionFailed
}

var ErrPreconditionFailed = fmt.Errorf("a precondition failed")
var ErrConflict = fmt.Errorf("changes are in conflict")

// hasConflicts returns true if the left and right JSON interface objects overlap with
// different values in any key.  The code will panic if an unrecognized type is passed
// (anything not returned by a JSON decode).  All keys are required to be strings.
func hasConflicts(left, right interface{}) bool {
	switch typedLeft := left.(type) {
	case map[string]interface{}:
		switch typedRight := right.(type) {
		case map[string]interface{}:
			for key, leftValue := range typedLeft {
				if rightValue, ok := typedRight[key]; ok && hasConflicts(leftValue, rightValue) {
					return true
				}
			}
			return false
		default:
			return true
		}
	case []interface{}:
		switch typedRight := right.(type) {
		case []interface{}:
			if len(typedLeft) != len(typedRight) {
				return true
			}
			for i := range typedLeft {
				if hasConflicts(typedLeft[i], typedRight[i]) {
					return true
				}
			}
			return false
		default:
			return true
		}
	case string, float64, bool, int, int64, nil:
		return !reflect.DeepEqual(left, right)
	default:
		panic(fmt.Sprintf("unknown type: %v", reflect.TypeOf(left)))
	}
}
