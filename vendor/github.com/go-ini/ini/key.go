// Copyright 2014 Unknwon
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

package ini

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Key represents a key under a section.
type Key struct {
	s               *Section
	name            string
	value           string
	isAutoIncrement bool
	isBooleanType   bool

	isShadow bool
	shadows  []*Key

	Comment string
}

// newKey simply return a key object with given values.
func newKey(s *Section, name, val string) *Key {
	return &Key{
		s:     s,
		name:  name,
		value: val,
	}
}

func (k *Key) addShadow(val string) error {
	if k.isShadow {
		return errors.New("cannot add shadow to another shadow key")
	} else if k.isAutoIncrement || k.isBooleanType {
		return errors.New("cannot add shadow to auto-increment or boolean key")
	}

	shadow := newKey(k.s, k.name, val)
	shadow.isShadow = true
	k.shadows = append(k.shadows, shadow)
	return nil
}

// AddShadow adds a new shadow key to itself.
func (k *Key) AddShadow(val string) error {
	if !k.s.f.options.AllowShadows {
		return errors.New("shadow key is not allowed")
	}
	return k.addShadow(val)
}

// ValueMapper represents a mapping function for values, e.g. os.ExpandEnv
type ValueMapper func(string) string

// Name returns name of key.
func (k *Key) Name() string {
	return k.name
}

// Value returns raw value of key for performance purpose.
func (k *Key) Value() string {
	return k.value
}

// ValueWithShadows returns raw values of key and its shadows if any.
func (k *Key) ValueWithShadows() []string {
	if len(k.shadows) == 0 {
		return []string{k.value}
	}
	vals := make([]string, len(k.shadows)+1)
	vals[0] = k.value
	for i := range k.shadows {
		vals[i+1] = k.shadows[i].value
	}
	return vals
}

// transformValue takes a raw value and transforms to its final string.
func (k *Key) transformValue(val string) string {
	if k.s.f.ValueMapper != nil {
		val = k.s.f.ValueMapper(val)
	}

	// Fail-fast if no indicate char found for recursive value
	if !strings.Contains(val, "%") {
		return val
	}
	for i := 0; i < _DEPTH_VALUES; i++ {
		vr := varPattern.FindString(val)
		if len(vr) == 0 {
			break
		}

		// Take off leading '%(' and trailing ')s'.
		noption := strings.TrimLeft(vr, "%(")
		noption = strings.TrimRight(noption, ")s")

		// Search in the same section.
		nk, err := k.s.GetKey(noption)
		if err != nil {
			// Search again in default section.
			nk, _ = k.s.f.Section("").GetKey(noption)
		}

		// Substitute by new value and take off leading '%(' and trailing ')s'.
		val = strings.Replace(val, vr, nk.value, -1)
	}
	return val
}

// String returns string representation of value.
func (k *Key) String() string {
	return k.transformValue(k.value)
}

// Validate accepts a validate function which can
// return modifed result as key value.
func (k *Key) Validate(fn func(string) string) string {
	return fn(k.String())
}

// parseBool returns the boolean value represented by the string.
//
// It accepts 1, t, T, TRUE, true, True, YES, yes, Yes, y, ON, on, On,
// 0, f, F, FALSE, false, False, NO, no, No, n, OFF, off, Off.
// Any other value returns an error.
func parseBool(str string) (value bool, err error) {
	switch str {
	case "1", "t", "T", "true", "TRUE", "True", "YES", "yes", "Yes", "y", "ON", "on", "On":
		return true, nil
	case "0", "f", "F", "false", "FALSE", "False", "NO", "no", "No", "n", "OFF", "off", "Off":
		return false, nil
	}
	return false, fmt.Errorf("parsing \"%s\": invalid syntax", str)
}

// Bool returns bool type value.
func (k *Key) Bool() (bool, error) {
	return parseBool(k.String())
}

// Float64 returns float64 type value.
func (k *Key) Float64() (float64, error) {
	return strconv.ParseFloat(k.String(), 64)
}

// Int returns int type value.
func (k *Key) Int() (int, error) {
	return strconv.Atoi(k.String())
}

// Int64 returns int64 type value.
func (k *Key) Int64() (int64, error) {
	return strconv.ParseInt(k.String(), 10, 64)
}

// Uint returns uint type valued.
func (k *Key) Uint() (uint, error) {
	u, e := strconv.ParseUint(k.String(), 10, 64)
	return uint(u), e
}

// Uint64 returns uint64 type value.
func (k *Key) Uint64() (uint64, error) {
	return strconv.ParseUint(k.String(), 10, 64)
}

// Duration returns time.Duration type value.
func (k *Key) Duration() (time.Duration, error) {
	return time.ParseDuration(k.String())
}

// TimeFormat parses with given format and returns time.Time type value.
func (k *Key) TimeFormat(format string) (time.Time, error) {
	return time.Parse(format, k.String())
}

// Time parses with RFC3339 format and returns time.Time type value.
func (k *Key) Time() (time.Time, error) {
	return k.TimeFormat(time.RFC3339)
}

// MustString returns default value if key value is empty.
func (k *Key) MustString(defaultVal string) string {
	val := k.String()
	if len(val) == 0 {
		k.value = defaultVal
		return defaultVal
	}
	return val
}

// MustBool always returns value without error,
// it returns false if error occurs.
func (k *Key) MustBool(defaultVal ...bool) bool {
	val, err := k.Bool()
	if len(defaultVal) > 0 && err != nil {
		k.value = strconv.FormatBool(defaultVal[0])
		return defaultVal[0]
	}
	return val
}

// MustFloat64 always returns value without error,
// it returns 0.0 if error occurs.
func (k *Key) MustFloat64(defaultVal ...float64) float64 {
	val, err := k.Float64()
	if len(defaultVal) > 0 && err != nil {
		k.value = strconv.FormatFloat(defaultVal[0], 'f', -1, 64)
		return defaultVal[0]
	}
	return val
}

// MustInt always returns value without error,
// it returns 0 if error occurs.
func (k *Key) MustInt(defaultVal ...int) int {
	val, err := k.Int()
	if len(defaultVal) > 0 && err != nil {
		k.value = strconv.FormatInt(int64(defaultVal[0]), 10)
		return defaultVal[0]
	}
	return val
}

// MustInt64 always returns value without error,
// it returns 0 if error occurs.
func (k *Key) MustInt64(defaultVal ...int64) int64 {
	val, err := k.Int64()
	if len(defaultVal) > 0 && err != nil {
		k.value = strconv.FormatInt(defaultVal[0], 10)
		return defaultVal[0]
	}
	return val
}

// MustUint always returns value without error,
// it returns 0 if error occurs.
func (k *Key) MustUint(defaultVal ...uint) uint {
	val, err := k.Uint()
	if len(defaultVal) > 0 && err != nil {
		k.value = strconv.FormatUint(uint64(defaultVal[0]), 10)
		return defaultVal[0]
	}
	return val
}

// MustUint64 always returns value without error,
// it returns 0 if error occurs.
func (k *Key) MustUint64(defaultVal ...uint64) uint64 {
	val, err := k.Uint64()
	if len(defaultVal) > 0 && err != nil {
		k.value = strconv.FormatUint(defaultVal[0], 10)
		return defaultVal[0]
	}
	return val
}

// MustDuration always returns value without error,
// it returns zero value if error occurs.
func (k *Key) MustDuration(defaultVal ...time.Duration) time.Duration {
	val, err := k.Duration()
	if len(defaultVal) > 0 && err != nil {
		k.value = defaultVal[0].String()
		return defaultVal[0]
	}
	return val
}

// MustTimeFormat always parses with given format and returns value without error,
// it returns zero value if error occurs.
func (k *Key) MustTimeFormat(format string, defaultVal ...time.Time) time.Time {
	val, err := k.TimeFormat(format)
	if len(defaultVal) > 0 && err != nil {
		k.value = defaultVal[0].Format(format)
		return defaultVal[0]
	}
	return val
}

// MustTime always parses with RFC3339 format and returns value without error,
// it returns zero value if error occurs.
func (k *Key) MustTime(defaultVal ...time.Time) time.Time {
	return k.MustTimeFormat(time.RFC3339, defaultVal...)
}

// In always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) In(defaultVal string, candidates []string) string {
	val := k.String()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InFloat64 always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InFloat64(defaultVal float64, candidates []float64) float64 {
	val := k.MustFloat64()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InInt always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InInt(defaultVal int, candidates []int) int {
	val := k.MustInt()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InInt64 always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InInt64(defaultVal int64, candidates []int64) int64 {
	val := k.MustInt64()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InUint always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InUint(defaultVal uint, candidates []uint) uint {
	val := k.MustUint()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InUint64 always returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InUint64(defaultVal uint64, candidates []uint64) uint64 {
	val := k.MustUint64()
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InTimeFormat always parses with given format and returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InTimeFormat(format string, defaultVal time.Time, candidates []time.Time) time.Time {
	val := k.MustTimeFormat(format)
	for _, cand := range candidates {
		if val == cand {
			return val
		}
	}
	return defaultVal
}

// InTime always parses with RFC3339 format and returns value without error,
// it returns default value if error occurs or doesn't fit into candidates.
func (k *Key) InTime(defaultVal time.Time, candidates []time.Time) time.Time {
	return k.InTimeFormat(time.RFC3339, defaultVal, candidates)
}

// RangeFloat64 checks if value is in given range inclusively,
// and returns default value if it's not.
func (k *Key) RangeFloat64(defaultVal, min, max float64) float64 {
	val := k.MustFloat64()
	if val < min || val > max {
		return defaultVal
	}
	return val
}

// RangeInt checks if value is in given range inclusively,
// and returns default value if it's not.
func (k *Key) RangeInt(defaultVal, min, max int) int {
	val := k.MustInt()
	if val < min || val > max {
		return defaultVal
	}
	return val
}

// RangeInt64 checks if value is in given range inclusively,
// and returns default value if it's not.
func (k *Key) RangeInt64(defaultVal, min, max int64) int64 {
	val := k.MustInt64()
	if val < min || val > max {
		return defaultVal
	}
	return val
}

// RangeTimeFormat checks if value with given format is in given range inclusively,
// and returns default value if it's not.
func (k *Key) RangeTimeFormat(format string, defaultVal, min, max time.Time) time.Time {
	val := k.MustTimeFormat(format)
	if val.Unix() < min.Unix() || val.Unix() > max.Unix() {
		return defaultVal
	}
	return val
}

// RangeTime checks if value with RFC3339 format is in given range inclusively,
// and returns default value if it's not.
func (k *Key) RangeTime(defaultVal, min, max time.Time) time.Time {
	return k.RangeTimeFormat(time.RFC3339, defaultVal, min, max)
}

// Strings returns list of string divided by given delimiter.
func (k *Key) Strings(delim string) []string {
	str := k.String()
	if len(str) == 0 {
		return []string{}
	}

	vals := strings.Split(str, delim)
	for i := range vals {
		// vals[i] = k.transformValue(strings.TrimSpace(vals[i]))
		vals[i] = strings.TrimSpace(vals[i])
	}
	return vals
}

// StringsWithShadows returns list of string divided by given delimiter.
// Shadows will also be appended if any.
func (k *Key) StringsWithShadows(delim string) []string {
	vals := k.ValueWithShadows()
	results := make([]string, 0, len(vals)*2)
	for i := range vals {
		if len(vals) == 0 {
			continue
		}

		results = append(results, strings.Split(vals[i], delim)...)
	}

	for i := range results {
		results[i] = k.transformValue(strings.TrimSpace(results[i]))
	}
	return results
}

// Float64s returns list of float64 divided by given delimiter. Any invalid input will be treated as zero value.
func (k *Key) Float64s(delim string) []float64 {
	vals, _ := k.getFloat64s(delim, true, false)
	return vals
}

// Ints returns list of int divided by given delimiter. Any invalid input will be treated as zero value.
func (k *Key) Ints(delim string) []int {
	vals, _ := k.parseInts(k.Strings(delim), true, false)
	return vals
}

// Int64s returns list of int64 divided by given delimiter. Any invalid input will be treated as zero value.
func (k *Key) Int64s(delim string) []int64 {
	vals, _ := k.parseInt64s(k.Strings(delim), true, false)
	return vals
}

// Uints returns list of uint divided by given delimiter. Any invalid input will be treated as zero value.
func (k *Key) Uints(delim string) []uint {
	vals, _ := k.getUints(delim, true, false)
	return vals
}

// Uint64s returns list of uint64 divided by given delimiter. Any invalid input will be treated as zero value.
func (k *Key) Uint64s(delim string) []uint64 {
	vals, _ := k.getUint64s(delim, true, false)
	return vals
}

// TimesFormat parses with given format and returns list of time.Time divided by given delimiter.
// Any invalid input will be treated as zero value (0001-01-01 00:00:00 +0000 UTC).
func (k *Key) TimesFormat(format, delim string) []time.Time {
	vals, _ := k.getTimesFormat(format, delim, true, false)
	return vals
}

// Times parses with RFC3339 format and returns list of time.Time divided by given delimiter.
// Any invalid input will be treated as zero value (0001-01-01 00:00:00 +0000 UTC).
func (k *Key) Times(delim string) []time.Time {
	return k.TimesFormat(time.RFC3339, delim)
}

// ValidFloat64s returns list of float64 divided by given delimiter. If some value is not float, then
// it will not be included to result list.
func (k *Key) ValidFloat64s(delim string) []float64 {
	vals, _ := k.getFloat64s(delim, false, false)
	return vals
}

// ValidInts returns list of int divided by given delimiter. If some value is not integer, then it will
// not be included to result list.
func (k *Key) ValidInts(delim string) []int {
	vals, _ := k.parseInts(k.Strings(delim), false, false)
	return vals
}

// ValidInt64s returns list of int64 divided by given delimiter. If some value is not 64-bit integer,
// then it will not be included to result list.
func (k *Key) ValidInt64s(delim string) []int64 {
	vals, _ := k.parseInt64s(k.Strings(delim), false, false)
	return vals
}

// ValidUints returns list of uint divided by given delimiter. If some value is not unsigned integer,
// then it will not be included to result list.
func (k *Key) ValidUints(delim string) []uint {
	vals, _ := k.getUints(delim, false, false)
	return vals
}

// ValidUint64s returns list of uint64 divided by given delimiter. If some value is not 64-bit unsigned
// integer, then it will not be included to result list.
func (k *Key) ValidUint64s(delim string) []uint64 {
	vals, _ := k.getUint64s(delim, false, false)
	return vals
}

// ValidTimesFormat parses with given format and returns list of time.Time divided by given delimiter.
func (k *Key) ValidTimesFormat(format, delim string) []time.Time {
	vals, _ := k.getTimesFormat(format, delim, false, false)
	return vals
}

// ValidTimes parses with RFC3339 format and returns list of time.Time divided by given delimiter.
func (k *Key) ValidTimes(delim string) []time.Time {
	return k.ValidTimesFormat(time.RFC3339, delim)
}

// StrictFloat64s returns list of float64 divided by given delimiter or error on first invalid input.
func (k *Key) StrictFloat64s(delim string) ([]float64, error) {
	return k.getFloat64s(delim, false, true)
}

// StrictInts returns list of int divided by given delimiter or error on first invalid input.
func (k *Key) StrictInts(delim string) ([]int, error) {
	return k.parseInts(k.Strings(delim), false, true)
}

// StrictInt64s returns list of int64 divided by given delimiter or error on first invalid input.
func (k *Key) StrictInt64s(delim string) ([]int64, error) {
	return k.parseInt64s(k.Strings(delim), false, true)
}

// StrictUints returns list of uint divided by given delimiter or error on first invalid input.
func (k *Key) StrictUints(delim string) ([]uint, error) {
	return k.getUints(delim, false, true)
}

// StrictUint64s returns list of uint64 divided by given delimiter or error on first invalid input.
func (k *Key) StrictUint64s(delim string) ([]uint64, error) {
	return k.getUint64s(delim, false, true)
}

// StrictTimesFormat parses with given format and returns list of time.Time divided by given delimiter
// or error on first invalid input.
func (k *Key) StrictTimesFormat(format, delim string) ([]time.Time, error) {
	return k.getTimesFormat(format, delim, false, true)
}

// StrictTimes parses with RFC3339 format and returns list of time.Time divided by given delimiter
// or error on first invalid input.
func (k *Key) StrictTimes(delim string) ([]time.Time, error) {
	return k.StrictTimesFormat(time.RFC3339, delim)
}

// getFloat64s returns list of float64 divided by given delimiter.
func (k *Key) getFloat64s(delim string, addInvalid, returnOnInvalid bool) ([]float64, error) {
	strs := k.Strings(delim)
	vals := make([]float64, 0, len(strs))
	for _, str := range strs {
		val, err := strconv.ParseFloat(str, 64)
		if err != nil && returnOnInvalid {
			return nil, err
		}
		if err == nil || addInvalid {
			vals = append(vals, val)
		}
	}
	return vals, nil
}

// parseInts transforms strings to ints.
func (k *Key) parseInts(strs []string, addInvalid, returnOnInvalid bool) ([]int, error) {
	vals := make([]int, 0, len(strs))
	for _, str := range strs {
		val, err := strconv.Atoi(str)
		if err != nil && returnOnInvalid {
			return nil, err
		}
		if err == nil || addInvalid {
			vals = append(vals, val)
		}
	}
	return vals, nil
}

// parseInt64s transforms strings to int64s.
func (k *Key) parseInt64s(strs []string, addInvalid, returnOnInvalid bool) ([]int64, error) {
	vals := make([]int64, 0, len(strs))
	for _, str := range strs {
		val, err := strconv.ParseInt(str, 10, 64)
		if err != nil && returnOnInvalid {
			return nil, err
		}
		if err == nil || addInvalid {
			vals = append(vals, val)
		}
	}
	return vals, nil
}

// getUints returns list of uint divided by given delimiter.
func (k *Key) getUints(delim string, addInvalid, returnOnInvalid bool) ([]uint, error) {
	strs := k.Strings(delim)
	vals := make([]uint, 0, len(strs))
	for _, str := range strs {
		val, err := strconv.ParseUint(str, 10, 0)
		if err != nil && returnOnInvalid {
			return nil, err
		}
		if err == nil || addInvalid {
			vals = append(vals, uint(val))
		}
	}
	return vals, nil
}

// getUint64s returns list of uint64 divided by given delimiter.
func (k *Key) getUint64s(delim string, addInvalid, returnOnInvalid bool) ([]uint64, error) {
	strs := k.Strings(delim)
	vals := make([]uint64, 0, len(strs))
	for _, str := range strs {
		val, err := strconv.ParseUint(str, 10, 64)
		if err != nil && returnOnInvalid {
			return nil, err
		}
		if err == nil || addInvalid {
			vals = append(vals, val)
		}
	}
	return vals, nil
}

// getTimesFormat parses with given format and returns list of time.Time divided by given delimiter.
func (k *Key) getTimesFormat(format, delim string, addInvalid, returnOnInvalid bool) ([]time.Time, error) {
	strs := k.Strings(delim)
	vals := make([]time.Time, 0, len(strs))
	for _, str := range strs {
		val, err := time.Parse(format, str)
		if err != nil && returnOnInvalid {
			return nil, err
		}
		if err == nil || addInvalid {
			vals = append(vals, val)
		}
	}
	return vals, nil
}

// SetValue changes key value.
func (k *Key) SetValue(v string) {
	if k.s.f.BlockMode {
		k.s.f.lock.Lock()
		defer k.s.f.lock.Unlock()
	}

	k.value = v
	k.s.keysHash[k.name] = v
}
