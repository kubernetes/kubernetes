// Copyright (C) 2013-2018 by Maxim Bublis <b@codemonkey.ru>
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package uuid

import (
	"bytes"
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

// Value implements the driver.Valuer interface.
func (u UUID) Value() (driver.Value, error) {
	return u.String(), nil
}

// Scan implements the sql.Scanner interface.
// A 16-byte slice will be handled by UnmarshalBinary, while
// a longer byte slice or a string will be handled by UnmarshalText.
func (u *UUID) Scan(src interface{}) error {
	switch src := src.(type) {
	case UUID: // support gorm convert from UUID to NullUUID
		*u = src
		return nil

	case []byte:
		if len(src) == Size {
			return u.UnmarshalBinary(src)
		}
		return u.UnmarshalText(src)

	case string:
		return u.UnmarshalText([]byte(src))
	}

	return fmt.Errorf("uuid: cannot convert %T to UUID", src)
}

// NullUUID can be used with the standard sql package to represent a
// UUID value that can be NULL in the database.
type NullUUID struct {
	UUID  UUID
	Valid bool
}

// Value implements the driver.Valuer interface.
func (u NullUUID) Value() (driver.Value, error) {
	if !u.Valid {
		return nil, nil
	}
	// Delegate to UUID Value function
	return u.UUID.Value()
}

// Scan implements the sql.Scanner interface.
func (u *NullUUID) Scan(src interface{}) error {
	if src == nil {
		u.UUID, u.Valid = Nil, false
		return nil
	}

	// Delegate to UUID Scan function
	u.Valid = true
	return u.UUID.Scan(src)
}

// MarshalJSON marshals the NullUUID as null or the nested UUID
func (u NullUUID) MarshalJSON() ([]byte, error) {
	if !u.Valid {
		return json.Marshal(nil)
	}

	return json.Marshal(u.UUID)
}

// UnmarshalJSON unmarshals a NullUUID
func (u *NullUUID) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		u.UUID, u.Valid = Nil, false
		return nil
	}

	if err := json.Unmarshal(b, &u.UUID); err != nil {
		return err
	}

	u.Valid = true

	return nil
}
