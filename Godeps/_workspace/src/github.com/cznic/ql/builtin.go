// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"fmt"
	"log"
	"math/rand"
	"reflect"
	"strings"
	"time"
)

//TODO agg bigint, bigrat, time, duration

var builtin = map[string]struct {
	f           func([]interface{}, map[interface{}]interface{}) (interface{}, error)
	minArgs     int
	maxArgs     int
	isStatic    bool
	isAggregate bool
}{
	"__testBlob":   {builtinTestBlob, 1, 1, true, false},
	"__testString": {builtinTestString, 1, 1, true, false},
	"avg":          {builtinAvg, 1, 1, false, true},
	"complex":      {builtinComplex, 2, 2, true, false},
	"contains":     {builtinContains, 2, 2, true, false},
	"count":        {builtinCount, 0, 1, false, true},
	"date":         {builtinDate, 8, 8, true, false},
	"day":          {builtinDay, 1, 1, true, false},
	"formatTime":   {builtinFormatTime, 2, 2, true, false},
	"hasPrefix":    {builtinHasPrefix, 2, 2, true, false},
	"hasSuffix":    {builtinHasSuffix, 2, 2, true, false},
	"hour":         {builtinHour, 1, 1, true, false},
	"hours":        {builtinHours, 1, 1, true, false},
	"id":           {builtinID, 0, 1, false, false},
	"imag":         {builtinImag, 1, 1, true, false},
	"len":          {builtinLen, 1, 1, true, false},
	"max":          {builtinMax, 1, 1, false, true},
	"min":          {builtinMin, 1, 1, false, true},
	"minute":       {builtinMinute, 1, 1, true, false},
	"minutes":      {builtinMinutes, 1, 1, true, false},
	"month":        {builtinMonth, 1, 1, true, false},
	"nanosecond":   {builtinNanosecond, 1, 1, true, false},
	"nanoseconds":  {builtinNanoseconds, 1, 1, true, false},
	"now":          {builtinNow, 0, 0, false, false},
	"parseTime":    {builtinParseTime, 2, 2, true, false},
	"real":         {builtinReal, 1, 1, true, false},
	"second":       {builtinSecond, 1, 1, true, false},
	"seconds":      {builtinSeconds, 1, 1, true, false},
	"since":        {builtinSince, 1, 1, false, false},
	"sum":          {builtinSum, 1, 1, false, true},
	"timeIn":       {builtinTimeIn, 2, 2, true, false},
	"weekday":      {builtinWeekday, 1, 1, true, false},
	"year":         {builtinYear, 1, 1, true, false},
	"yearDay":      {builtinYearday, 1, 1, true, false},
}

func badNArgs(min int, s string, arg []interface{}) error {
	a := []string{}
	for _, v := range arg {
		a = append(a, fmt.Sprintf("%v", v))
	}
	switch len(arg) < min {
	case true:
		return fmt.Errorf("missing argument to %s(%s)", s, strings.Join(a, ", "))
	default: //case false:
		return fmt.Errorf("too many arguments to %s(%s)", s, strings.Join(a, ", "))
	}
}

func invArg(arg interface{}, s string) error {
	return fmt.Errorf("invalid argument %v (type %T) for %s", arg, arg, s)
}

func builtinTestBlob(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	n, err := intExpr(arg[0])
	if err != nil {
		return nil, err
	}

	rng := rand.New(rand.NewSource(n))
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(rng.Int())
	}
	return b, nil
}

func builtinTestString(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	n, err := intExpr(arg[0])
	if err != nil {
		return nil, err
	}

	rng := rand.New(rand.NewSource(n))
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(rng.Int())
	}
	return string(b), nil
}

func builtinAvg(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	type avg struct {
		sum interface{}
		n   uint64
	}

	if _, ok := ctx["$agg0"]; ok {
		return
	}

	fn := ctx["$fn"]
	if _, ok := ctx["$agg"]; ok {
		data, ok := ctx[fn].(avg)
		if !ok {
			return
		}

		switch x := data.sum.(type) {
		case complex64:
			return complex64(complex128(x) / complex(float64(data.n), 0)), nil
		case complex128:
			return complex64(complex128(x) / complex(float64(data.n), 0)), nil
		case float32:
			return float32(float64(x) / float64(data.n)), nil
		case float64:
			return float64(x) / float64(data.n), nil
		case int8:
			return int8(int64(x) / int64(data.n)), nil
		case int16:
			return int16(int64(x) / int64(data.n)), nil
		case int32:
			return int32(int64(x) / int64(data.n)), nil
		case int64:
			return int64(int64(x) / int64(data.n)), nil
		case uint8:
			return uint8(uint64(x) / data.n), nil
		case uint16:
			return uint16(uint64(x) / data.n), nil
		case uint32:
			return uint32(uint64(x) / data.n), nil
		case uint64:
			return uint64(uint64(x) / data.n), nil
		}

	}

	data, _ := ctx[fn].(avg)
	y := arg[0]
	if y == nil {
		return
	}

	switch x := data.sum.(type) {
	case nil:
		switch y := y.(type) {
		case float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64:
			data = avg{y, 0}
		default:
			return nil, fmt.Errorf("avg: cannot accept %v (value if type %T)", y, y)
		}
	case complex64:
		data.sum = x + y.(complex64)
	case complex128:
		data.sum = x + y.(complex128)
	case float32:
		data.sum = x + y.(float32)
	case float64:
		data.sum = x + y.(float64)
	case int8:
		data.sum = x + y.(int8)
	case int16:
		data.sum = x + y.(int16)
	case int32:
		data.sum = x + y.(int32)
	case int64:
		data.sum = x + y.(int64)
	case uint8:
		data.sum = x + y.(uint8)
	case uint16:
		data.sum = x + y.(uint16)
	case uint32:
		data.sum = x + y.(uint32)
	case uint64:
		data.sum = x + y.(uint64)
	}
	data.n++
	ctx[fn] = data
	return
}

func builtinComplex(arg []interface{}, _ map[interface{}]interface{}) (v interface{}, err error) {
	re, im := arg[0], arg[1]
	if re == nil || im == nil {
		return nil, nil
	}

	re, im = coerce(re, im)
	if reflect.TypeOf(re) != reflect.TypeOf(im) {
		return nil, fmt.Errorf("complex(%T(%#v), %T(%#v)): invalid types", re, re, im, im)
	}

	switch re := re.(type) {
	case idealFloat:
		return idealComplex(complex(float64(re), float64(im.(idealFloat)))), nil
	case idealInt:
		return idealComplex(complex(float64(re), float64(im.(idealInt)))), nil
	case idealRune:
		return idealComplex(complex(float64(re), float64(im.(idealRune)))), nil
	case idealUint:
		return idealComplex(complex(float64(re), float64(im.(idealUint)))), nil
	case float32:
		return complex(float32(re), im.(float32)), nil
	case float64:
		return complex(float64(re), im.(float64)), nil
	case int8:
		return complex(float64(re), float64(im.(int8))), nil
	case int16:
		return complex(float64(re), float64(im.(int16))), nil
	case int32:
		return complex(float64(re), float64(im.(int32))), nil
	case int64:
		return complex(float64(re), float64(im.(int64))), nil
	case uint8:
		return complex(float64(re), float64(im.(uint8))), nil
	case uint16:
		return complex(float64(re), float64(im.(uint16))), nil
	case uint32:
		return complex(float64(re), float64(im.(uint32))), nil
	case uint64:
		return complex(float64(re), float64(im.(uint64))), nil
	default:
		return nil, invArg(re, "complex")
	}
}

func builtinContains(arg []interface{}, _ map[interface{}]interface{}) (v interface{}, err error) {
	switch s := arg[0].(type) {
	case nil:
		return nil, nil
	case string:
		switch chars := arg[1].(type) {
		case nil:
			return nil, nil
		case string:
			return strings.Contains(s, chars), nil
		default:
			return nil, invArg(chars, "string")
		}
	default:
		return nil, invArg(s, "string")
	}
}

func builtinCount(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	if _, ok := ctx["$agg0"]; ok {
		return int64(0), nil
	}

	fn := ctx["$fn"]
	if _, ok := ctx["$agg"]; ok {
		return ctx[fn].(int64), nil
	}

	n, _ := ctx[fn].(int64)
	switch len(arg) {
	case 0:
		n++
	case 1:
		if arg[0] != nil {
			n++
		}
	default:
		log.Panic("internal error 067")
	}
	ctx[fn] = n
	return
}

func builtinDate(arg []interface{}, _ map[interface{}]interface{}) (v interface{}, err error) {
	for i, v := range arg {
		switch i {
		case 7:
			switch x := v.(type) {
			case string:
			default:
				return nil, invArg(x, "date")
			}
		default:
			switch x := v.(type) {
			case int64:
			case idealInt:
				arg[i] = int64(x)
			default:
				return nil, invArg(x, "date")
			}
		}
	}

	sloc := arg[7].(string)
	loc := time.Local
	switch sloc {
	case "local":
	default:
		loc, err = time.LoadLocation(sloc)
		if err != nil {
			return
		}
	}

	return time.Date(
		int(arg[0].(int64)),
		time.Month(arg[1].(int64)),
		int(arg[2].(int64)),
		int(arg[3].(int64)),
		int(arg[4].(int64)),
		int(arg[5].(int64)),
		int(arg[6].(int64)),
		loc,
	), nil
}

func builtinLen(arg []interface{}, _ map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case string:
		return int64(len(x)), nil
	default:
		return nil, invArg(x, "len")
	}
}

func builtinDay(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		return int64(x.Day()), nil
	default:
		return nil, invArg(x, "day")
	}
}

func builtinFormatTime(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		switch y := arg[1].(type) {
		case nil:
			return nil, nil
		case string:
			return x.Format(y), nil
		default:
			return nil, invArg(y, "formatTime")
		}
	default:
		return nil, invArg(x, "formatTime")
	}
}

func builtinHasPrefix(arg []interface{}, _ map[interface{}]interface{}) (v interface{}, err error) {
	switch s := arg[0].(type) {
	case nil:
		return nil, nil
	case string:
		switch prefix := arg[1].(type) {
		case nil:
			return nil, nil
		case string:
			return strings.HasPrefix(s, prefix), nil
		default:
			return nil, invArg(prefix, "string")
		}
	default:
		return nil, invArg(s, "string")
	}
}

func builtinHasSuffix(arg []interface{}, _ map[interface{}]interface{}) (v interface{}, err error) {
	switch s := arg[0].(type) {
	case nil:
		return nil, nil
	case string:
		switch suffix := arg[1].(type) {
		case nil:
			return nil, nil
		case string:
			return strings.HasSuffix(s, suffix), nil
		default:
			return nil, invArg(suffix, "string")
		}
	default:
		return nil, invArg(s, "string")
	}
}

func builtinHour(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		return int64(x.Hour()), nil
	default:
		return nil, invArg(x, "hour")
	}
}

func builtinHours(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Duration:
		return x.Hours(), nil
	default:
		return nil, invArg(x, "hours")
	}
}

func builtinID(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := ctx["$id"].(type) {
	case map[string]interface{}:
		if len(arg) == 0 {
			return nil, nil
		}

		tab := arg[0].(*ident)
		id, ok := x[tab.s]
		if !ok {
			return nil, fmt.Errorf("value not available: id(%s)", tab)
		}

		if _, ok := id.(int64); ok {
			return id, nil
		}

		return nil, fmt.Errorf("value not available: id(%s)", tab)
	case int64:
		return x, nil
	default:
		panic("internal error 072")
	}
}

func builtinImag(arg []interface{}, _ map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case idealComplex:
		return imag(x), nil
	case complex64:
		return imag(x), nil
	case complex128:
		return imag(x), nil
	default:
		return nil, invArg(x, "imag")
	}
}

func builtinMax(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	if _, ok := ctx["$agg0"]; ok {
		return
	}

	fn := ctx["$fn"]
	if _, ok := ctx["$agg"]; ok {
		if v, ok = ctx[fn]; ok {
			return
		}

		return nil, nil
	}

	max := ctx[fn]
	y := arg[0]
	if y == nil {
		return
	}
	switch x := max.(type) {
	case nil:
		switch y := y.(type) {
		case float32, float64, string, int8, int16, int32, int64, uint8, uint16, uint32, uint64, time.Time:
			max = y
		default:
			return nil, fmt.Errorf("max: cannot accept %v (value if type %T)", y, y)
		}
	case float32:
		if y := y.(float32); y > x {
			max = y
		}
	case float64:
		if y := y.(float64); y > x {
			max = y
		}
	case string:
		if y := y.(string); y > x {
			max = y
		}
	case int8:
		if y := y.(int8); y > x {
			max = y
		}
	case int16:
		if y := y.(int16); y > x {
			max = y
		}
	case int32:
		if y := y.(int32); y > x {
			max = y
		}
	case int64:
		if y := y.(int64); y > x {
			max = y
		}
	case uint8:
		if y := y.(uint8); y > x {
			max = y
		}
	case uint16:
		if y := y.(uint16); y > x {
			max = y
		}
	case uint32:
		if y := y.(uint32); y > x {
			max = y
		}
	case uint64:
		if y := y.(uint64); y > x {
			max = y
		}
	case time.Time:
		if y := y.(time.Time); y.After(x) {
			max = y
		}
	}
	ctx[fn] = max
	return
}

func builtinMin(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	if _, ok := ctx["$agg0"]; ok {
		return
	}

	fn := ctx["$fn"]
	if _, ok := ctx["$agg"]; ok {
		if v, ok = ctx[fn]; ok {
			return
		}

		return nil, nil
	}

	min := ctx[fn]
	y := arg[0]
	if y == nil {
		return
	}
	switch x := min.(type) {
	case nil:
		switch y := y.(type) {
		case float32, float64, string, int8, int16, int32, int64, uint8, uint16, uint32, uint64, time.Time:
			min = y
		default:
			return nil, fmt.Errorf("min: cannot accept %v (value if type %T)", y, y)
		}
	case float32:
		if y := y.(float32); y < x {
			min = y
		}
	case float64:
		if y := y.(float64); y < x {
			min = y
		}
	case string:
		if y := y.(string); y < x {
			min = y
		}
	case int8:
		if y := y.(int8); y < x {
			min = y
		}
	case int16:
		if y := y.(int16); y < x {
			min = y
		}
	case int32:
		if y := y.(int32); y < x {
			min = y
		}
	case int64:
		if y := y.(int64); y < x {
			min = y
		}
	case uint8:
		if y := y.(uint8); y < x {
			min = y
		}
	case uint16:
		if y := y.(uint16); y < x {
			min = y
		}
	case uint32:
		if y := y.(uint32); y < x {
			min = y
		}
	case uint64:
		if y := y.(uint64); y < x {
			min = y
		}
	case time.Time:
		if y := y.(time.Time); y.Before(x) {
			min = y
		}
	}
	ctx[fn] = min
	return
}

func builtinMinute(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		return int64(x.Minute()), nil
	default:
		return nil, invArg(x, "minute")
	}
}

func builtinMinutes(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Duration:
		return x.Minutes(), nil
	default:
		return nil, invArg(x, "minutes")
	}
}

func builtinMonth(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		return int64(x.Month()), nil
	default:
		return nil, invArg(x, "month")
	}
}

func builtinNanosecond(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		return int64(x.Nanosecond()), nil
	default:
		return nil, invArg(x, "nanosecond")
	}
}

func builtinNanoseconds(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Duration:
		return x.Nanoseconds(), nil
	default:
		return nil, invArg(x, "nanoseconds")
	}
}

func builtinNow(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	return time.Now(), nil
}

func builtinParseTime(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	var a [2]string
	for i, v := range arg {
		switch x := v.(type) {
		case nil:
			return nil, nil
		case string:
			a[i] = x
		default:
			return nil, invArg(x, "parseTime")
		}
	}

	t, err := time.Parse(a[0], a[1])
	if err != nil {
		return nil, err
	}

	ls := t.Location().String()
	if ls == "UTC" {
		return t, nil
	}

	l, err := time.LoadLocation(ls)
	if err != nil {
		return t, nil
	}

	return time.ParseInLocation(a[0], a[1], l)
}

func builtinReal(arg []interface{}, _ map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case idealComplex:
		return real(x), nil
	case complex64:
		return real(x), nil
	case complex128:
		return real(x), nil
	default:
		return nil, invArg(x, "real")
	}
}

func builtinSecond(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		return int64(x.Second()), nil
	default:
		return nil, invArg(x, "second")
	}
}

func builtinSeconds(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Duration:
		return x.Seconds(), nil
	default:
		return nil, invArg(x, "seconds")
	}
}

func builtinSince(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		return time.Since(x), nil
	default:
		return nil, invArg(x, "since")
	}
}

func builtinSum(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	if _, ok := ctx["$agg0"]; ok {
		return
	}

	fn := ctx["$fn"]
	if _, ok := ctx["$agg"]; ok {
		if v, ok = ctx[fn]; ok {
			return
		}

		return nil, nil
	}

	sum := ctx[fn]
	y := arg[0]
	if y == nil {
		return
	}
	switch x := sum.(type) {
	case nil:
		switch y := y.(type) {
		case complex64, complex128, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64:
			sum = y
		default:
			return nil, fmt.Errorf("sum: cannot accept %v (value if type %T)", y, y)
		}
	case complex64:
		sum = x + y.(complex64)
	case complex128:
		sum = x + y.(complex128)
	case float32:
		sum = x + y.(float32)
	case float64:
		sum = x + y.(float64)
	case int8:
		sum = x + y.(int8)
	case int16:
		sum = x + y.(int16)
	case int32:
		sum = x + y.(int32)
	case int64:
		sum = x + y.(int64)
	case uint8:
		sum = x + y.(uint8)
	case uint16:
		sum = x + y.(uint16)
	case uint32:
		sum = x + y.(uint32)
	case uint64:
		sum = x + y.(uint64)
	}
	ctx[fn] = sum
	return
}

func builtinTimeIn(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		switch y := arg[1].(type) {
		case nil:
			return nil, nil
		case string:
			loc := time.Local
			switch y {
			case "local":
			default:
				loc, err = time.LoadLocation(y)
				if err != nil {
					return
				}
			}

			return x.In(loc), nil
		default:
			return nil, invArg(x, "timeIn")
		}
	default:
		return nil, invArg(x, "timeIn")
	}
}

func builtinWeekday(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		return int64(x.Weekday()), nil
	default:
		return nil, invArg(x, "weekday")
	}
}

func builtinYear(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		return int64(x.Year()), nil
	default:
		return nil, invArg(x, "year")
	}
}

func builtinYearday(arg []interface{}, ctx map[interface{}]interface{}) (v interface{}, err error) {
	switch x := arg[0].(type) {
	case nil:
		return nil, nil
	case time.Time:
		return int64(x.YearDay()), nil
	default:
		return nil, invArg(x, "yearDay")
	}
}
