// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Package time provides for custom types to translate time from JSON and other formats
// into time.Time objects.
package time

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Unix provides a type that can marshal and unmarshal a string representation
// of the unix epoch into a time.Time object.
type Unix struct {
	T time.Time
}

// MarshalJSON implements encoding/json.MarshalJSON().
func (u Unix) MarshalJSON() ([]byte, error) {
	if u.T.IsZero() {
		return []byte(""), nil
	}
	return []byte(fmt.Sprintf("%q", strconv.FormatInt(u.T.Unix(), 10))), nil
}

// UnmarshalJSON implements encoding/json.UnmarshalJSON().
func (u *Unix) UnmarshalJSON(b []byte) error {
	i, err := strconv.Atoi(strings.Trim(string(b), `"`))
	if err != nil {
		return fmt.Errorf("unix time(%s) could not be converted from string to int: %w", string(b), err)
	}
	u.T = time.Unix(int64(i), 0)
	return nil
}

// DurationTime provides a type that can marshal and unmarshal a string representation
// of a duration from now into a time.Time object.
// Note: I'm not sure this is the best way to do this. What happens is we get a field
// called "expires_in" that represents the seconds from now that this expires. We
// turn that into a time we call .ExpiresOn. But maybe we should be recording
// when the token was received at .TokenRecieved and .ExpiresIn should remain as a duration.
// Then we could have a method called ExpiresOn().  Honestly, the whole thing is
// bad because the server doesn't return a concrete time. I think this is
// cleaner, but its not great either.
type DurationTime struct {
	T time.Time
}

// MarshalJSON implements encoding/json.MarshalJSON().
func (d DurationTime) MarshalJSON() ([]byte, error) {
	if d.T.IsZero() {
		return []byte(""), nil
	}

	dt := time.Until(d.T)
	return []byte(fmt.Sprintf("%d", int64(dt*time.Second))), nil
}

// UnmarshalJSON implements encoding/json.UnmarshalJSON().
func (d *DurationTime) UnmarshalJSON(b []byte) error {
	i, err := strconv.Atoi(strings.Trim(string(b), `"`))
	if err != nil {
		return fmt.Errorf("unix time(%s) could not be converted from string to int: %w", string(b), err)
	}
	d.T = time.Now().Add(time.Duration(i) * time.Second)
	return nil
}
