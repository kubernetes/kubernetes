// Copyright 2021 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	"bytes"
	"database/sql/driver"
	"encoding/json"
	"fmt"
)

var jsonNull = []byte("null")

// NullUUID represents a UUID that may be null.
// NullUUID implements the SQL driver.Scanner interface so
// it can be used as a scan destination:
//
//  var u uuid.NullUUID
//  err := db.QueryRow("SELECT name FROM foo WHERE id=?", id).Scan(&u)
//  ...
//  if u.Valid {
//     // use u.UUID
//  } else {
//     // NULL value
//  }
//
type NullUUID struct {
	UUID  UUID
	Valid bool // Valid is true if UUID is not NULL
}

// Scan implements the SQL driver.Scanner interface.
func (nu *NullUUID) Scan(value interface{}) error {
	if value == nil {
		nu.UUID, nu.Valid = Nil, false
		return nil
	}

	err := nu.UUID.Scan(value)
	if err != nil {
		nu.Valid = false
		return err
	}

	nu.Valid = true
	return nil
}

// Value implements the driver Valuer interface.
func (nu NullUUID) Value() (driver.Value, error) {
	if !nu.Valid {
		return nil, nil
	}
	// Delegate to UUID Value function
	return nu.UUID.Value()
}

// MarshalBinary implements encoding.BinaryMarshaler.
func (nu NullUUID) MarshalBinary() ([]byte, error) {
	if nu.Valid {
		return nu.UUID[:], nil
	}

	return []byte(nil), nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
func (nu *NullUUID) UnmarshalBinary(data []byte) error {
	if len(data) != 16 {
		return fmt.Errorf("invalid UUID (got %d bytes)", len(data))
	}
	copy(nu.UUID[:], data)
	nu.Valid = true
	return nil
}

// MarshalText implements encoding.TextMarshaler.
func (nu NullUUID) MarshalText() ([]byte, error) {
	if nu.Valid {
		return nu.UUID.MarshalText()
	}

	return jsonNull, nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (nu *NullUUID) UnmarshalText(data []byte) error {
	id, err := ParseBytes(data)
	if err != nil {
		nu.Valid = false
		return err
	}
	nu.UUID = id
	nu.Valid = true
	return nil
}

// MarshalJSON implements json.Marshaler.
func (nu NullUUID) MarshalJSON() ([]byte, error) {
	if nu.Valid {
		return json.Marshal(nu.UUID)
	}

	return jsonNull, nil
}

// UnmarshalJSON implements json.Unmarshaler.
func (nu *NullUUID) UnmarshalJSON(data []byte) error {
	if bytes.Equal(data, jsonNull) {
		*nu = NullUUID{}
		return nil // valid null UUID
	}
	err := json.Unmarshal(data, &nu.UUID)
	nu.Valid = err == nil
	return err
}
