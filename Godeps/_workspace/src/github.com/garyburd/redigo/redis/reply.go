// Copyright 2012 Gary Burd
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package redis

import (
	"errors"
	"fmt"
	"strconv"
)

// ErrNil indicates that a reply value is nil.
var ErrNil = errors.New("redigo: nil returned")

// Int is a helper that converts a command reply to an integer. If err is not
// equal to nil, then Int returns 0, err. Otherwise, Int converts the
// reply to an int as follows:
//
//  Reply type    Result
//  integer       int(reply), nil
//  bulk string   parsed reply, nil
//  nil           0, ErrNil
//  other         0, error
func Int(reply interface{}, err error) (int, error) {
	if err != nil {
		return 0, err
	}
	switch reply := reply.(type) {
	case int64:
		x := int(reply)
		if int64(x) != reply {
			return 0, strconv.ErrRange
		}
		return x, nil
	case []byte:
		n, err := strconv.ParseInt(string(reply), 10, 0)
		return int(n), err
	case nil:
		return 0, ErrNil
	case Error:
		return 0, reply
	}
	return 0, fmt.Errorf("redigo: unexpected type for Int, got type %T", reply)
}

// Int64 is a helper that converts a command reply to 64 bit integer. If err is
// not equal to nil, then Int returns 0, err. Otherwise, Int64 converts the
// reply to an int64 as follows:
//
//  Reply type    Result
//  integer       reply, nil
//  bulk string   parsed reply, nil
//  nil           0, ErrNil
//  other         0, error
func Int64(reply interface{}, err error) (int64, error) {
	if err != nil {
		return 0, err
	}
	switch reply := reply.(type) {
	case int64:
		return reply, nil
	case []byte:
		n, err := strconv.ParseInt(string(reply), 10, 64)
		return n, err
	case nil:
		return 0, ErrNil
	case Error:
		return 0, reply
	}
	return 0, fmt.Errorf("redigo: unexpected type for Int64, got type %T", reply)
}

var errNegativeInt = errors.New("redigo: unexpected value for Uint64")

// Uint64 is a helper that converts a command reply to 64 bit integer. If err is
// not equal to nil, then Int returns 0, err. Otherwise, Int64 converts the
// reply to an int64 as follows:
//
//  Reply type    Result
//  integer       reply, nil
//  bulk string   parsed reply, nil
//  nil           0, ErrNil
//  other         0, error
func Uint64(reply interface{}, err error) (uint64, error) {
	if err != nil {
		return 0, err
	}
	switch reply := reply.(type) {
	case int64:
		if reply < 0 {
			return 0, errNegativeInt
		}
		return uint64(reply), nil
	case []byte:
		n, err := strconv.ParseUint(string(reply), 10, 64)
		return n, err
	case nil:
		return 0, ErrNil
	case Error:
		return 0, reply
	}
	return 0, fmt.Errorf("redigo: unexpected type for Uint64, got type %T", reply)
}

// Float64 is a helper that converts a command reply to 64 bit float. If err is
// not equal to nil, then Float64 returns 0, err. Otherwise, Float64 converts
// the reply to an int as follows:
//
//  Reply type    Result
//  bulk string   parsed reply, nil
//  nil           0, ErrNil
//  other         0, error
func Float64(reply interface{}, err error) (float64, error) {
	if err != nil {
		return 0, err
	}
	switch reply := reply.(type) {
	case []byte:
		n, err := strconv.ParseFloat(string(reply), 64)
		return n, err
	case nil:
		return 0, ErrNil
	case Error:
		return 0, reply
	}
	return 0, fmt.Errorf("redigo: unexpected type for Float64, got type %T", reply)
}

// String is a helper that converts a command reply to a string. If err is not
// equal to nil, then String returns "", err. Otherwise String converts the
// reply to a string as follows:
//
//  Reply type      Result
//  bulk string     string(reply), nil
//  simple string   reply, nil
//  nil             "",  ErrNil
//  other           "",  error
func String(reply interface{}, err error) (string, error) {
	if err != nil {
		return "", err
	}
	switch reply := reply.(type) {
	case []byte:
		return string(reply), nil
	case string:
		return reply, nil
	case nil:
		return "", ErrNil
	case Error:
		return "", reply
	}
	return "", fmt.Errorf("redigo: unexpected type for String, got type %T", reply)
}

// Bytes is a helper that converts a command reply to a slice of bytes. If err
// is not equal to nil, then Bytes returns nil, err. Otherwise Bytes converts
// the reply to a slice of bytes as follows:
//
//  Reply type      Result
//  bulk string     reply, nil
//  simple string   []byte(reply), nil
//  nil             nil, ErrNil
//  other           nil, error
func Bytes(reply interface{}, err error) ([]byte, error) {
	if err != nil {
		return nil, err
	}
	switch reply := reply.(type) {
	case []byte:
		return reply, nil
	case string:
		return []byte(reply), nil
	case nil:
		return nil, ErrNil
	case Error:
		return nil, reply
	}
	return nil, fmt.Errorf("redigo: unexpected type for Bytes, got type %T", reply)
}

// Bool is a helper that converts a command reply to a boolean. If err is not
// equal to nil, then Bool returns false, err. Otherwise Bool converts the
// reply to boolean as follows:
//
//  Reply type      Result
//  integer         value != 0, nil
//  bulk string     strconv.ParseBool(reply)
//  nil             false, ErrNil
//  other           false, error
func Bool(reply interface{}, err error) (bool, error) {
	if err != nil {
		return false, err
	}
	switch reply := reply.(type) {
	case int64:
		return reply != 0, nil
	case []byte:
		return strconv.ParseBool(string(reply))
	case nil:
		return false, ErrNil
	case Error:
		return false, reply
	}
	return false, fmt.Errorf("redigo: unexpected type for Bool, got type %T", reply)
}

// MultiBulk is deprecated. Use Values.
func MultiBulk(reply interface{}, err error) ([]interface{}, error) { return Values(reply, err) }

// Values is a helper that converts an array command reply to a []interface{}.
// If err is not equal to nil, then Values returns nil, err. Otherwise, Values
// converts the reply as follows:
//
//  Reply type      Result
//  array           reply, nil
//  nil             nil, ErrNil
//  other           nil, error
func Values(reply interface{}, err error) ([]interface{}, error) {
	if err != nil {
		return nil, err
	}
	switch reply := reply.(type) {
	case []interface{}:
		return reply, nil
	case nil:
		return nil, ErrNil
	case Error:
		return nil, reply
	}
	return nil, fmt.Errorf("redigo: unexpected type for Values, got type %T", reply)
}

// Strings is a helper that converts an array command reply to a []string. If
// err is not equal to nil, then Strings returns nil, err. Nil array items are
// converted to "" in the output slice. Strings returns an error if an array
// item is not a bulk string or nil.
func Strings(reply interface{}, err error) ([]string, error) {
	if err != nil {
		return nil, err
	}
	switch reply := reply.(type) {
	case []interface{}:
		result := make([]string, len(reply))
		for i := range reply {
			if reply[i] == nil {
				continue
			}
			p, ok := reply[i].([]byte)
			if !ok {
				return nil, fmt.Errorf("redigo: unexpected element type for Strings, got type %T", reply[i])
			}
			result[i] = string(p)
		}
		return result, nil
	case nil:
		return nil, ErrNil
	case Error:
		return nil, reply
	}
	return nil, fmt.Errorf("redigo: unexpected type for Strings, got type %T", reply)
}

// Ints is a helper that converts an array command reply to a []int. If
// err is not equal to nil, then Ints returns nil, err.
func Ints(reply interface{}, err error) ([]int, error) {
	var ints []int
	if reply == nil {
		return ints, ErrNil
	}
	values, err := Values(reply, err)
	if err != nil {
		return ints, err
	}
	if err := ScanSlice(values, &ints); err != nil {
		return ints, err
	}
	return ints, nil
}

// StringMap is a helper that converts an array of strings (alternating key, value)
// into a map[string]string. The HGETALL and CONFIG GET commands return replies in this format.
// Requires an even number of values in result.
func StringMap(result interface{}, err error) (map[string]string, error) {
	values, err := Values(result, err)
	if err != nil {
		return nil, err
	}
	if len(values)%2 != 0 {
		return nil, errors.New("redigo: StringMap expects even number of values result")
	}
	m := make(map[string]string, len(values)/2)
	for i := 0; i < len(values); i += 2 {
		key, okKey := values[i].([]byte)
		value, okValue := values[i+1].([]byte)
		if !okKey || !okValue {
			return nil, errors.New("redigo: ScanMap key not a bulk string value")
		}
		m[string(key)] = string(value)
	}
	return m, nil
}
