package parseutil

import (
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/mitchellh/mapstructure"
)

func ParseDurationSecond(in interface{}) (time.Duration, error) {
	var dur time.Duration
	jsonIn, ok := in.(json.Number)
	if ok {
		in = jsonIn.String()
	}
	switch in.(type) {
	case string:
		inp := in.(string)
		var err error
		// Look for a suffix otherwise its a plain second value
		if strings.HasSuffix(inp, "s") || strings.HasSuffix(inp, "m") || strings.HasSuffix(inp, "h") {
			dur, err = time.ParseDuration(inp)
			if err != nil {
				return dur, err
			}
		} else {
			// Plain integer
			secs, err := strconv.ParseInt(inp, 10, 64)
			if err != nil {
				return dur, err
			}
			dur = time.Duration(secs) * time.Second
		}
	case int:
		dur = time.Duration(in.(int)) * time.Second
	case int32:
		dur = time.Duration(in.(int32)) * time.Second
	case int64:
		dur = time.Duration(in.(int64)) * time.Second
	case uint:
		dur = time.Duration(in.(uint)) * time.Second
	case uint32:
		dur = time.Duration(in.(uint32)) * time.Second
	case uint64:
		dur = time.Duration(in.(uint64)) * time.Second
	default:
		return 0, errors.New("could not parse duration from input")
	}

	return dur, nil
}

func ParseBool(in interface{}) (bool, error) {
	var result bool
	if err := mapstructure.WeakDecode(in, &result); err != nil {
		return false, err
	}
	return result, nil
}
