/*
Copyright 2013 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package types provides various common types.
package types // import "go4.org/types"

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"sync"
	"time"
)

var null_b = []byte("null")

// NopCloser is an io.Closer that does nothing.
var NopCloser io.Closer = CloseFunc(func() error { return nil })

// EmptyBody is a ReadCloser that returns EOF on Read and does nothing
// on Close.
var EmptyBody io.ReadCloser = ioutil.NopCloser(strings.NewReader(""))

// Time3339 is a time.Time which encodes to and from JSON
// as an RFC 3339 time in UTC.
type Time3339 time.Time

var (
	_ json.Marshaler   = Time3339{}
	_ json.Unmarshaler = (*Time3339)(nil)
)

func (t Time3339) String() string {
	return time.Time(t).UTC().Format(time.RFC3339Nano)
}

func (t Time3339) MarshalJSON() ([]byte, error) {
	if t.Time().IsZero() {
		return null_b, nil
	}
	return json.Marshal(t.String())
}

func (t *Time3339) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, null_b) {
		*t = Time3339{}
		return nil
	}
	if len(b) < 2 || b[0] != '"' || b[len(b)-1] != '"' {
		return fmt.Errorf("types: failed to unmarshal non-string value %q as an RFC 3339 time", b)
	}
	s := string(b[1 : len(b)-1])
	if s == "" {
		*t = Time3339{}
		return nil
	}
	tm, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		if strings.HasPrefix(s, "0000-00-00T00:00:00") {
			*t = Time3339{}
			return nil
		}
		return err
	}
	*t = Time3339(tm)
	return nil
}

// ParseTime3339OrZero parses a string in RFC3339 format. If it's invalid,
// the zero time value is returned instead.
func ParseTime3339OrZero(v string) Time3339 {
	t, err := time.Parse(time.RFC3339Nano, v)
	if err != nil {
		return Time3339{}
	}
	return Time3339(t)
}

func ParseTime3339OrNil(v string) *Time3339 {
	t, err := time.Parse(time.RFC3339Nano, v)
	if err != nil {
		return nil
	}
	tm := Time3339(t)
	return &tm
}

// Time returns the time as a time.Time with slightly less stutter
// than a manual conversion.
func (t Time3339) Time() time.Time {
	return time.Time(t)
}

// IsZero returns whether the time is Go zero or Unix zero.
func (t *Time3339) IsAnyZero() bool {
	return t == nil || time.Time(*t).IsZero() || time.Time(*t).Unix() == 0
}

// ByTime sorts times.
type ByTime []time.Time

func (s ByTime) Len() int           { return len(s) }
func (s ByTime) Less(i, j int) bool { return s[i].Before(s[j]) }
func (s ByTime) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// NewOnceCloser returns a Closer wrapping c which only calls Close on c
// once. Subsequent calls to Close return nil.
func NewOnceCloser(c io.Closer) io.Closer {
	return &onceCloser{c: c}
}

type onceCloser struct {
	mu sync.Mutex
	c  io.Closer
}

func (c *onceCloser) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.c == nil {
		return nil
	}
	err := c.c.Close()
	c.c = nil
	return err
}

// CloseFunc implements io.Closer with a function.
type CloseFunc func() error

func (fn CloseFunc) Close() error { return fn() }
