// Package filters provides helper function to parse and handle command line
// filter, used for example in docker ps or docker images commands.
package filters

import (
	"encoding/json"
	"errors"
	"regexp"
	"strings"
)

// Args stores filter arguments as map key:{array of values}.
// It contains a aggregation of the list of arguments (which are in the form
// of -f 'key=value') based on the key, and store values for the same key
// in an slice.
// e.g given -f 'label=label1=1' -f 'label=label2=2' -f 'image.name=ubuntu'
// the args will be {'label': {'label1=1','label2=2'}, 'image.name', {'ubuntu'}}
type Args map[string][]string

// ParseFlag parses the argument to the filter flag. Like
//
//   `docker ps -f 'created=today' -f 'image.name=ubuntu*'`
//
// If prev map is provided, then it is appended to, and returned. By default a new
// map is created.
func ParseFlag(arg string, prev Args) (Args, error) {
	filters := prev
	if prev == nil {
		filters = Args{}
	}
	if len(arg) == 0 {
		return filters, nil
	}

	if !strings.Contains(arg, "=") {
		return filters, ErrBadFormat
	}

	f := strings.SplitN(arg, "=", 2)
	name := strings.ToLower(strings.TrimSpace(f[0]))
	value := strings.TrimSpace(f[1])
	filters[name] = append(filters[name], value)

	return filters, nil
}

// ErrBadFormat is an error returned in case of bad format for a filter.
var ErrBadFormat = errors.New("bad format of filter (expected name=value)")

// ToParam packs the Args into an string for easy transport from client to server.
func ToParam(a Args) (string, error) {
	// this way we don't URL encode {}, just empty space
	if len(a) == 0 {
		return "", nil
	}

	buf, err := json.Marshal(a)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

// FromParam unpacks the filter Args.
func FromParam(p string) (Args, error) {
	args := Args{}
	if len(p) == 0 {
		return args, nil
	}
	if err := json.NewDecoder(strings.NewReader(p)).Decode(&args); err != nil {
		return nil, err
	}
	return args, nil
}

// MatchKVList returns true if the values for the specified field maches the ones
// from the sources.
// e.g. given Args are {'label': {'label1=1','label2=1'}, 'image.name', {'ubuntu'}},
//      field is 'label' and sources are {'label':{'label1=1','label2=2','label3=3'}}
//      it returns true.
func (filters Args) MatchKVList(field string, sources map[string]string) bool {
	fieldValues := filters[field]

	//do not filter if there is no filter set or cannot determine filter
	if len(fieldValues) == 0 {
		return true
	}

	if sources == nil || len(sources) == 0 {
		return false
	}

outer:
	for _, name2match := range fieldValues {
		testKV := strings.SplitN(name2match, "=", 2)

		for k, v := range sources {
			if len(testKV) == 1 {
				if k == testKV[0] {
					continue outer
				}
			} else if k == testKV[0] && v == testKV[1] {
				continue outer
			}
		}

		return false
	}

	return true
}

// Match returns true if the values for the specified field matches the source string
// e.g. given Args are {'label': {'label1=1','label2=1'}, 'image.name', {'ubuntu'}},
//      field is 'image.name' and source is 'ubuntu'
//      it returns true.
func (filters Args) Match(field, source string) bool {
	fieldValues := filters[field]

	//do not filter if there is no filter set or cannot determine filter
	if len(fieldValues) == 0 {
		return true
	}
	for _, name2match := range fieldValues {
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
