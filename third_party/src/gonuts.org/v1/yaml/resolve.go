package yaml

import (
	"math"
	"strconv"
	"strings"
)

// TODO: merge, timestamps, base 60 floats, omap.

type resolveMapItem struct {
	value interface{}
	tag   string
}

var resolveTable = make([]byte, 256)
var resolveMap = make(map[string]resolveMapItem)

func init() {
	t := resolveTable
	t[int('+')] = 'S' // Sign
	t[int('-')] = 'S'
	for _, c := range "0123456789" {
		t[int(c)] = 'D' // Digit
	}
	for _, c := range "yYnNtTfFoO~" {
		t[int(c)] = 'M' // In map
	}
	t[int('.')] = '.' // Float (potentially in map)
	t[int('<')] = '<' // Merge

	var resolveMapList = []struct {
		v   interface{}
		tag string
		l   []string
	}{
		{true, "!!bool", []string{"y", "Y", "yes", "Yes", "YES"}},
		{true, "!!bool", []string{"true", "True", "TRUE"}},
		{true, "!!bool", []string{"on", "On", "ON"}},
		{false, "!!bool", []string{"n", "N", "no", "No", "NO"}},
		{false, "!!bool", []string{"false", "False", "FALSE"}},
		{false, "!!bool", []string{"off", "Off", "OFF"}},
		{nil, "!!null", []string{"~", "null", "Null", "NULL"}},
		{math.NaN(), "!!float", []string{".nan", ".NaN", ".NAN"}},
		{math.Inf(+1), "!!float", []string{".inf", ".Inf", ".INF"}},
		{math.Inf(+1), "!!float", []string{"+.inf", "+.Inf", "+.INF"}},
		{math.Inf(-1), "!!float", []string{"-.inf", "-.Inf", "-.INF"}},
		{"<<", "!!merge", []string{"<<"}},
	}

	m := resolveMap
	for _, item := range resolveMapList {
		for _, s := range item.l {
			m[s] = resolveMapItem{item.v, item.tag}
		}
	}
}

const longTagPrefix = "tag:yaml.org,2002:"

func shortTag(tag string) string {
	if strings.HasPrefix(tag, longTagPrefix) {
		return "!!" + tag[len(longTagPrefix):]
	}
	return tag
}

func resolvableTag(tag string) bool {
	switch tag {
	case "", "!!str", "!!bool", "!!int", "!!float", "!!null":
		return true
	}
	return false
}

func resolve(tag string, in string) (rtag string, out interface{}) {
	tag = shortTag(tag)
	if !resolvableTag(tag) {
		return tag, in
	}

	defer func() {
		if tag != "" && tag != rtag {
			panic("Can't decode " + rtag + " '" + in + "' as a " + tag)
		}
	}()

	if in == "" {
		return "!!null", nil
	}

	c := resolveTable[in[0]]
	if c == 0 {
		// It's a string for sure. Nothing to do.
		return "!!str", in
	}

	// Handle things we can lookup in a map.
	if item, ok := resolveMap[in]; ok {
		return item.tag, item.value
	}

	switch c {
	case 'M':
		// We've already checked the map above.

	case '.':
		// Not in the map, so maybe a normal float.
		floatv, err := strconv.ParseFloat(in, 64)
		if err == nil {
			return "!!float", floatv
		}
	// XXX Handle base 60 floats here (WTF!)

	case 'D', 'S':
		// Int, float, or timestamp.
		plain := strings.Replace(in, "_", "", -1)
		intv, err := strconv.ParseInt(plain, 0, 64)
		if err == nil {
			if intv == int64(int(intv)) {
				return "!!int", int(intv)
			} else {
				return "!!int", intv
			}
		}
		floatv, err := strconv.ParseFloat(plain, 64)
		if err == nil {
			return "!!float", floatv
		}
		if strings.HasPrefix(plain, "0b") {
			intv, err := strconv.ParseInt(plain[2:], 2, 64)
			if err == nil {
				return "!!int", int(intv)
			}
		} else if strings.HasPrefix(plain, "-0b") {
			intv, err := strconv.ParseInt(plain[3:], 2, 64)
			if err == nil {
				return "!!int", -int(intv)
			}
		}
	// XXX Handle timestamps here.

	default:
		panic("resolveTable item not yet handled: " +
			string([]byte{c}) + " (with " + in + ")")
	}
	return "!!str", in
}
