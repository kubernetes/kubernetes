package semver

import (
	"database/sql/driver"
	"fmt"
)

// Scan implements the database/sql.Scanner interface.
func (v *Version) Scan(src interface{}) (err error) {
	var str string
	switch src := src.(type) {
	case string:
		str = src
	case []byte:
		str = string(src)
	default:
		return fmt.Errorf("version.Scan: cannot convert %T to string", src)
	}

	if t, err := Parse(str); err == nil {
		*v = t
	}

	return
}

// Value implements the database/sql/driver.Valuer interface.
func (v Version) Value() (driver.Value, error) {
	return v.String(), nil
}
