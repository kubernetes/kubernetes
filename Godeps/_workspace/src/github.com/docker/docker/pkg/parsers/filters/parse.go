package filters

import (
	"encoding/json"
	"errors"
	"regexp"
	"strings"
)

type Args map[string][]string

// Parse the argument to the filter flag. Like
//
//   `docker ps -f 'created=today' -f 'image.name=ubuntu*'`
//
// If prev map is provided, then it is appended to, and returned. By default a new
// map is created.
func ParseFlag(arg string, prev Args) (Args, error) {
	var filters Args = prev
	if prev == nil {
		filters = Args{}
	}
	if len(arg) == 0 {
		return filters, nil
	}

	if !strings.Contains(arg, "=") {
		return filters, ErrorBadFormat
	}

	f := strings.SplitN(arg, "=", 2)
	name := strings.ToLower(strings.TrimSpace(f[0]))
	value := strings.TrimSpace(f[1])
	filters[name] = append(filters[name], value)

	return filters, nil
}

var ErrorBadFormat = errors.New("bad format of filter (expected name=value)")

// packs the Args into an string for easy transport from client to server
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

// unpacks the filter Args
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
