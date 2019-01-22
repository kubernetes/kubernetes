// Copyright 2012 Kamil Kisiel. All rights reserved.
// Use of this source code is governed by the MIT
// license which can be found in the LICENSE file.

/*
Package sqlstruct provides some convenience functions for using structs with
the Go standard library's database/sql package.

The package matches struct field names to SQL query column names. A field can
also specify a matching column with "sql" tag, if it's different from field
name.  Unexported fields or fields marked with `sql:"-"` are ignored, just like
with "encoding/json" package.

For example:

    type T struct {
        F1 string
        F2 string `sql:"field2"`
        F3 string `sql:"-"`
    }

    rows, err := db.Query(fmt.Sprintf("SELECT %s FROM tablename", sqlstruct.Columns(T{})))
    ...

    for rows.Next() {
    	var t T
        err = sqlstruct.Scan(&t, rows)
        ...
    }

    err = rows.Err() // get any errors encountered during iteration

Aliased tables in a SQL statement may be scanned into a specific structure identified
by the same alias, using the ColumnsAliased and ScanAliased functions:

    type User struct {
        Id int `sql:"id"`
        Username string `sql:"username"`
        Email string `sql:"address"`
        Name string `sql:"name"`
        HomeAddress *Address `sql:"-"`
    }

    type Address struct {
        Id int `sql:"id"`
        City string `sql:"city"`
        Street string `sql:"address"`
    }

    ...

    var user User
    var address Address
    sql := `
SELECT %s, %s FROM users AS u
INNER JOIN address AS a ON a.id = u.address_id
WHERE u.username = ?
`
    sql = fmt.Sprintf(sql, sqlstruct.ColumnsAliased(*user, "u"), sqlstruct.ColumnsAliased(*address, "a"))
    rows, err := db.Query(sql, "gedi")
    if err != nil {
        log.Fatal(err)
    }
    defer rows.Close()
    if rows.Next() {
        err = sqlstruct.ScanAliased(&user, rows, "u")
        if err != nil {
            log.Fatal(err)
        }
        err = sqlstruct.ScanAliased(&address, rows, "a")
        if err != nil {
            log.Fatal(err)
        }
        user.HomeAddress = address
    }
    fmt.Printf("%+v", *user)
    // output: "{Id:1 Username:gedi Email:gediminas.morkevicius@gmail.com Name:Gedas HomeAddress:0xc21001f570}"
    fmt.Printf("%+v", *user.HomeAddress)
    // output: "{Id:2 City:Vilnius Street:Plento 34}"

*/
package sqlstruct

import (
	"bytes"
	"database/sql"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
)

// NameMapper is the function used to convert struct fields which do not have sql tags
// into database column names.
//
// The default mapper converts field names to lower case. If instead you would prefer
// field names converted to snake case, simply assign sqlstruct.ToSnakeCase to the variable:
//
//		sqlstruct.NameMapper = sqlstruct.ToSnakeCase
//
// Alternatively for a custom mapping, any func(string) string can be used instead.
var NameMapper func(string) string = strings.ToLower

// A cache of fieldInfos to save reflecting every time. Inspried by encoding/xml
var finfos map[reflect.Type]fieldInfo
var finfoLock sync.RWMutex

// TagName is the name of the tag to use on struct fields
var TagName = "sql"

// fieldInfo is a mapping of field tag values to their indices
type fieldInfo map[string][]int

func init() {
	finfos = make(map[reflect.Type]fieldInfo)
}

// Rows defines the interface of types that are scannable with the Scan function.
// It is implemented by the sql.Rows type from the standard library
type Rows interface {
	Scan(...interface{}) error
	Columns() ([]string, error)
}

// getFieldInfo creates a fieldInfo for the provided type. Fields that are not tagged
// with the "sql" tag and unexported fields are not included.
func getFieldInfo(typ reflect.Type) fieldInfo {
	finfoLock.RLock()
	finfo, ok := finfos[typ]
	finfoLock.RUnlock()
	if ok {
		return finfo
	}

	finfo = make(fieldInfo)

	n := typ.NumField()
	for i := 0; i < n; i++ {
		f := typ.Field(i)
		tag := f.Tag.Get(TagName)

		// Skip unexported fields or fields marked with "-"
		if f.PkgPath != "" || tag == "-" {
			continue
		}

		// Handle embedded structs
		if f.Anonymous && f.Type.Kind() == reflect.Struct {
			for k, v := range getFieldInfo(f.Type) {
				finfo[k] = append([]int{i}, v...)
			}
			continue
		}

		// Use field name for untagged fields
		if tag == "" {
			tag = f.Name
		}
		tag = NameMapper(tag)

		finfo[tag] = []int{i}
	}

	finfoLock.Lock()
	finfos[typ] = finfo
	finfoLock.Unlock()

	return finfo
}

// Scan scans the next row from rows in to a struct pointed to by dest. The struct type
// should have exported fields tagged with the "sql" tag. Columns from row which are not
// mapped to any struct fields are ignored. Struct fields which have no matching column
// in the result set are left unchanged.
func Scan(dest interface{}, rows Rows) error {
	return doScan(dest, rows, "")
}

// ScanAliased works like scan, except that it expects the results in the query to be
// prefixed by the given alias.
//
// For example, if scanning to a field named "name" with an alias of "user" it will
// expect to find the result in a column named "user_name".
//
// See ColumnAliased for a convenient way to generate these queries.
func ScanAliased(dest interface{}, rows Rows, alias string) error {
	return doScan(dest, rows, alias)
}

// Columns returns a string containing a sorted, comma-separated list of column names as
// defined by the type s. s must be a struct that has exported fields tagged with the "sql" tag.
func Columns(s interface{}) string {
	return strings.Join(cols(s), ", ")
}

// ColumnsAliased works like Columns except it prefixes the resulting column name with the
// given alias.
//
// For each field in the given struct it will generate a statement like:
//    alias.field AS alias_field
//
// It is intended to be used in conjunction with the ScanAliased function.
func ColumnsAliased(s interface{}, alias string) string {
	names := cols(s)
	aliased := make([]string, 0, len(names))
	for _, n := range names {
		aliased = append(aliased, alias+"."+n+" AS "+alias+"_"+n)
	}
	return strings.Join(aliased, ", ")
}

func cols(s interface{}) []string {
	v := reflect.ValueOf(s)
	fields := getFieldInfo(v.Type())

	names := make([]string, 0, len(fields))
	for f := range fields {
		names = append(names, f)
	}

	sort.Strings(names)
	return names
}

func doScan(dest interface{}, rows Rows, alias string) error {
	destv := reflect.ValueOf(dest)
	typ := destv.Type()

	if typ.Kind() != reflect.Ptr || typ.Elem().Kind() != reflect.Struct {
		panic(fmt.Errorf("dest must be pointer to struct; got %T", destv))
	}
	fieldInfo := getFieldInfo(typ.Elem())

	elem := destv.Elem()
	var values []interface{}

	cols, err := rows.Columns()
	if err != nil {
		return err
	}

	for _, name := range cols {
		if len(alias) > 0 {
			name = strings.Replace(name, alias+"_", "", 1)
		}
		idx, ok := fieldInfo[strings.ToLower(name)]
		var v interface{}
		if !ok {
			// There is no field mapped to this column so we discard it
			v = &sql.RawBytes{}
		} else {
			v = elem.FieldByIndex(idx).Addr().Interface()
		}
		values = append(values, v)
	}

	return rows.Scan(values...)
}

// ToSnakeCase converts a string to snake case, words separated with underscores.
// It's intended to be used with NameMapper to map struct field names to snake case database fields.
func ToSnakeCase(src string) string {
	thisUpper := false
	prevUpper := false

	buf := bytes.NewBufferString("")
	for i, v := range src {
		if v >= 'A' && v <= 'Z' {
			thisUpper = true
		} else {
			thisUpper = false
		}
		if i > 0 && thisUpper && !prevUpper {
			buf.WriteRune('_')
		}
		prevUpper = thisUpper
		buf.WriteRune(v)
	}
	return strings.ToLower(buf.String())
}
