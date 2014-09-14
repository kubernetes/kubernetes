package ec2test

import (
	"fmt"
	"net/url"
	"strings"
)

// filter holds an ec2 filter.  A filter maps an attribute to a set of
// possible values for that attribute. For an item to pass through the
// filter, every attribute of the item mentioned in the filter must match
// at least one of its given values.
type filter map[string][]string

// newFilter creates a new filter from the Filter fields in the url form.
//
// The filtering is specified through a map of name=>values, where the
// name is a well-defined key identifying the data to be matched,
// and the list of values holds the possible values the filtered
// item can take for the key to be included in the
// result set. For example:
//
//   Filter.1.Name=instance-type
//   Filter.1.Value.1=m1.small
//   Filter.1.Value.2=m1.large
//
func newFilter(form url.Values) filter {
	// TODO return an error if the fields are not well formed?
	names := make(map[int]string)
	values := make(map[int][]string)
	maxId := 0
	for name, fvalues := range form {
		var rest string
		var id int
		if x, _ := fmt.Sscanf(name, "Filter.%d.%s", &id, &rest); x != 2 {
			continue
		}
		if id > maxId {
			maxId = id
		}
		if rest == "Name" {
			names[id] = fvalues[0]
			continue
		}
		if !strings.HasPrefix(rest, "Value.") {
			continue
		}
		values[id] = append(values[id], fvalues[0])
	}

	f := make(filter)
	for id, name := range names {
		f[name] = values[id]
	}
	return f
}

func notDigit(r rune) bool {
	return r < '0' || r > '9'
}

// filterable represents an object that can be passed through a filter.
type filterable interface {
	// matchAttr returns true if given attribute of the
	// object matches value. It returns an error if the
	// attribute is not recognised or the value is malformed.
	matchAttr(attr, value string) (bool, error)
}

// ok returns true if x passes through the filter.
func (f filter) ok(x filterable) (bool, error) {
next:
	for a, vs := range f {
		for _, v := range vs {
			if ok, err := x.matchAttr(a, v); ok {
				continue next
			} else if err != nil {
				return false, fmt.Errorf("bad attribute or value %q=%q for type %T: %v", a, v, x, err)
			}
		}
		return false, nil
	}
	return true, nil
}
