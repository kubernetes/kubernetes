// Package filters provides helper function to parse and handle command line
// filter, used for example in docker ps or docker images commands.
package filters

import (
	"encoding/json"
	"errors"
	"regexp"
	"strings"

	"github.com/docker/docker/api/types/versions"
)

// Args stores filter arguments as map key:{map key: bool}.
// It contains an aggregation of the map of arguments (which are in the form
// of -f 'key=value') based on the key, and stores values for the same key
// in a map with string keys and boolean values.
// e.g given -f 'label=label1=1' -f 'label=label2=2' -f 'image.name=ubuntu'
// the args will be {"image.name":{"ubuntu":true},"label":{"label1=1":true,"label2=2":true}}
type Args struct {
	fields map[string]map[string]bool
}

// NewArgs initializes a new Args struct.
func NewArgs() Args {
	return Args{fields: map[string]map[string]bool{}}
}

// ParseFlag parses the argument to the filter flag. Like
//
//   `docker ps -f 'created=today' -f 'image.name=ubuntu*'`
//
// If prev map is provided, then it is appended to, and returned. By default a new
// map is created.
func ParseFlag(arg string, prev Args) (Args, error) {
	filters := prev
	if len(arg) == 0 {
		return filters, nil
	}

	if !strings.Contains(arg, "=") {
		return filters, ErrBadFormat
	}

	f := strings.SplitN(arg, "=", 2)

	name := strings.ToLower(strings.TrimSpace(f[0]))
	value := strings.TrimSpace(f[1])

	filters.Add(name, value)

	return filters, nil
}

// ErrBadFormat is an error returned in case of bad format for a filter.
var ErrBadFormat = errors.New("bad format of filter (expected name=value)")

// ToParam packs the Args into a string for easy transport from client to server.
func ToParam(a Args) (string, error) {
	// this way we don't URL encode {}, just empty space
	if a.Len() == 0 {
		return "", nil
	}

	buf, err := json.Marshal(a.fields)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

// ToParamWithVersion packs the Args into a string for easy transport from client to server.
// The generated string will depend on the specified version (corresponding to the API version).
func ToParamWithVersion(version string, a Args) (string, error) {
	// this way we don't URL encode {}, just empty space
	if a.Len() == 0 {
		return "", nil
	}

	// for daemons older than v1.10, filter must be of the form map[string][]string
	var buf []byte
	var err error
	if version != "" && versions.LessThan(version, "1.22") {
		buf, err = json.Marshal(convertArgsToSlice(a.fields))
	} else {
		buf, err = json.Marshal(a.fields)
	}
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

// FromParam unpacks the filter Args.
func FromParam(p string) (Args, error) {
	if len(p) == 0 {
		return NewArgs(), nil
	}

	r := strings.NewReader(p)
	d := json.NewDecoder(r)

	m := map[string]map[string]bool{}
	if err := d.Decode(&m); err != nil {
		r.Seek(0, 0)

		// Allow parsing old arguments in slice format.
		// Because other libraries might be sending them in this format.
		deprecated := map[string][]string{}
		if deprecatedErr := d.Decode(&deprecated); deprecatedErr == nil {
			m = deprecatedArgs(deprecated)
		} else {
			return NewArgs(), err
		}
	}
	return Args{m}, nil
}

// Get returns the list of values associates with a field.
// It returns a slice of strings to keep backwards compatibility with old code.
func (filters Args) Get(field string) []string {
	values := filters.fields[field]
	if values == nil {
		return make([]string, 0)
	}
	slice := make([]string, 0, len(values))
	for key := range values {
		slice = append(slice, key)
	}
	return slice
}

// Add adds a new value to a filter field.
func (filters Args) Add(name, value string) {
	if _, ok := filters.fields[name]; ok {
		filters.fields[name][value] = true
	} else {
		filters.fields[name] = map[string]bool{value: true}
	}
}

// Del removes a value from a filter field.
func (filters Args) Del(name, value string) {
	if _, ok := filters.fields[name]; ok {
		delete(filters.fields[name], value)
		if len(filters.fields[name]) == 0 {
			delete(filters.fields, name)
		}
	}
}

// Len returns the number of fields in the arguments.
func (filters Args) Len() int {
	return len(filters.fields)
}

// MatchKVList returns true if the values for the specified field matches the ones
// from the sources.
// e.g. given Args are {'label': {'label1=1','label2=1'}, 'image.name', {'ubuntu'}},
//      field is 'label' and sources are {'label1': '1', 'label2': '2'}
//      it returns true.
func (filters Args) MatchKVList(field string, sources map[string]string) bool {
	fieldValues := filters.fields[field]

	//do not filter if there is no filter set or cannot determine filter
	if len(fieldValues) == 0 {
		return true
	}

	if len(sources) == 0 {
		return false
	}

	for name2match := range fieldValues {
		testKV := strings.SplitN(name2match, "=", 2)

		v, ok := sources[testKV[0]]
		if !ok {
			return false
		}
		if len(testKV) == 2 && testKV[1] != v {
			return false
		}
	}

	return true
}

// Match returns true if the values for the specified field matches the source string
// e.g. given Args are {'label': {'label1=1','label2=1'}, 'image.name', {'ubuntu'}},
//      field is 'image.name' and source is 'ubuntu'
//      it returns true.
func (filters Args) Match(field, source string) bool {
	if filters.ExactMatch(field, source) {
		return true
	}

	fieldValues := filters.fields[field]
	for name2match := range fieldValues {
		match, err := regexp.MatchString(name2match, source)
		if err != nil {
			continue
		}
		if match {
			return true
		}
	}
	return false
}

// ExactMatch returns true if the source matches exactly one of the filters.
func (filters Args) ExactMatch(field, source string) bool {
	fieldValues, ok := filters.fields[field]
	//do not filter if there is no filter set or cannot determine filter
	if !ok || len(fieldValues) == 0 {
		return true
	}

	// try to match full name value to avoid O(N) regular expression matching
	return fieldValues[source]
}

// UniqueExactMatch returns true if there is only one filter and the source matches exactly this one.
func (filters Args) UniqueExactMatch(field, source string) bool {
	fieldValues := filters.fields[field]
	//do not filter if there is no filter set or cannot determine filter
	if len(fieldValues) == 0 {
		return true
	}
	if len(filters.fields[field]) != 1 {
		return false
	}

	// try to match full name value to avoid O(N) regular expression matching
	return fieldValues[source]
}

// FuzzyMatch returns true if the source matches exactly one of the filters,
// or the source has one of the filters as a prefix.
func (filters Args) FuzzyMatch(field, source string) bool {
	if filters.ExactMatch(field, source) {
		return true
	}

	fieldValues := filters.fields[field]
	for prefix := range fieldValues {
		if strings.HasPrefix(source, prefix) {
			return true
		}
	}
	return false
}

// Include returns true if the name of the field to filter is in the filters.
func (filters Args) Include(field string) bool {
	_, ok := filters.fields[field]
	return ok
}

type invalidFilter string

func (e invalidFilter) Error() string {
	return "Invalid filter '" + string(e) + "'"
}

func (invalidFilter) InvalidParameter() {}

// Validate ensures that all the fields in the filter are valid.
// It returns an error as soon as it finds an invalid field.
func (filters Args) Validate(accepted map[string]bool) error {
	for name := range filters.fields {
		if !accepted[name] {
			return invalidFilter(name)
		}
	}
	return nil
}

// WalkValues iterates over the list of filtered values for a field.
// It stops the iteration if it finds an error and it returns that error.
func (filters Args) WalkValues(field string, op func(value string) error) error {
	if _, ok := filters.fields[field]; !ok {
		return nil
	}
	for v := range filters.fields[field] {
		if err := op(v); err != nil {
			return err
		}
	}
	return nil
}

func deprecatedArgs(d map[string][]string) map[string]map[string]bool {
	m := map[string]map[string]bool{}
	for k, v := range d {
		values := map[string]bool{}
		for _, vv := range v {
			values[vv] = true
		}
		m[k] = values
	}
	return m
}

func convertArgsToSlice(f map[string]map[string]bool) map[string][]string {
	m := map[string][]string{}
	for k, v := range f {
		values := []string{}
		for kk := range v {
			if v[kk] {
				values = append(values, kk)
			}
		}
		m[k] = values
	}
	return m
}
