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
	"database/sql"
	"database/sql/driver"
	"fmt"
)

var _ driver.Valuer = UUID{}
var _ sql.Scanner = (*UUID)(nil)

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
		uu, err := FromString(src)
		*u = uu
		return err
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

var nullJSON = []byte("null")

// MarshalJSON marshals the NullUUID as null or the nested UUID
func (u NullUUID) MarshalJSON() ([]byte, error) {
	if !u.Valid {
		return nullJSON, nil
	}
	var buf [38]byte
	buf[0] = '"'
	encodeCanonical(buf[1:37], u.UUID)
	buf[37] = '"'
	return buf[:], nil
}

// UnmarshalJSON unmarshals a NullUUID
func (u *NullUUID) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		u.UUID, u.Valid = Nil, false
		return nil
	}
	if n := len(b); n >= 2 && b[0] == '"' {
		b = b[1 : n-1]
	}
	err := u.UUID.UnmarshalText(b)
	u.Valid = (err == nil)
	return err
}
