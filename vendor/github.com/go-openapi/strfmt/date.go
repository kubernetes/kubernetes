// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package strfmt

import (
	"database/sql/driver"
	"fmt"
	"regexp"
	"time"

	"github.com/mailru/easyjson/jlexer"
	"github.com/mailru/easyjson/jwriter"
)

func init() {
	d := Date{}
	Default.Add("date", &d, IsDate)
}

// IsDate returns true when the string is a valid date
func IsDate(str string) bool {
	matches := rxDate.FindAllStringSubmatch(str, -1)
	if len(matches) == 0 || len(matches[0]) == 0 {
		return false
	}
	m := matches[0]
	return !(m[2] < "01" || m[2] > "12" || m[3] < "01" || m[3] > "31")
}

const (
	// RFC3339FullDate represents a full-date as specified by RFC3339
	// See: http://goo.gl/xXOvVd
	RFC3339FullDate = "2006-01-02"
	// DatePattern pattern to match for the date format from http://tools.ietf.org/html/rfc3339#section-5.6
	DatePattern = `^([0-9]{4})-([0-9]{2})-([0-9]{2})`
)

var (
	rxDate = regexp.MustCompile(DatePattern)
)

// Date represents a date from the API
//
// swagger:strfmt date
type Date time.Time

// String converts this date into a string
func (d Date) String() string {
	return time.Time(d).Format(RFC3339FullDate)
}

// UnmarshalText parses a text representation into a date type
func (d *Date) UnmarshalText(text []byte) error {
	if len(text) == 0 {
		return nil
	}
	dd, err := time.Parse(RFC3339FullDate, string(text))
	if err != nil {
		return err
	}
	*d = Date(dd)
	return nil
}

// MarshalText serializes this date type to string
func (d Date) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

// Scan scans a Date value from database driver type.
func (d *Date) Scan(raw interface{}) error {
	switch v := raw.(type) {
	case []byte:
		return d.UnmarshalText(v)
	case string:
		return d.UnmarshalText([]byte(v))
	case time.Time:
		*d = Date(v)
		return nil
	case nil:
		*d = Date{}
		return nil
	default:
		return fmt.Errorf("cannot sql.Scan() strfmt.Date from: %#v", v)
	}
}

// Value converts Date to a primitive value ready to written to a database.
func (d Date) Value() (driver.Value, error) {
	return driver.Value(d), nil
}

func (t Date) MarshalJSON() ([]byte, error) {
	var w jwriter.Writer
	t.MarshalEasyJSON(&w)
	return w.BuildBytes()
}

func (t Date) MarshalEasyJSON(w *jwriter.Writer) {
	w.String(time.Time(t).Format(RFC3339FullDate))
}

func (t *Date) UnmarshalJSON(data []byte) error {
	l := jlexer.Lexer{Data: data}
	t.UnmarshalEasyJSON(&l)
	return l.Error()
}

func (t *Date) UnmarshalEasyJSON(in *jlexer.Lexer) {
	if data := in.String(); in.Ok() {
		tt, err := time.Parse(RFC3339FullDate, data)
		if err != nil {
			in.AddError(err)
			return
		}
		*t = Date(tt)
	}
}
