// Copyright (c) 2014 ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"regexp"
	"runtime/debug"
	"strings"
	"testing"
)

var (
	oN = flag.Int("N", 0, "")
	oM = flag.Int("M", 0, "")
)

var testdata []string

func init() {
	tests, err := ioutil.ReadFile("testdata.ql")
	if err != nil {
		log.Panic(err)
	}

	a := bytes.Split(tests, []byte("\n-- "))
	pre := []byte("-- ")
	pres := []byte("S ")
	for _, v := range a[1:] {
		switch {
		case bytes.HasPrefix(v, pres):
			v = append(pre, v...)
			v = append([]byte(sample), v...)
		default:
			v = append(pre, v...)
		}
		testdata = append(testdata, string(v))
	}
}

func typeof(v interface{}) (r int) { //NTYPE
	switch v.(type) {
	case bool:
		return qBool
	case complex64:
		return qComplex64
	case complex128:
		return qComplex128
	case float32:
		return qFloat32
	case float64:
		return qFloat64
	case int8:
		return qInt8
	case int16:
		return qInt16
	case int32:
		return qInt32
	case int64:
		return qInt64
	case string:
		return qString
	case uint8:
		return qUint8
	case uint16:
		return qUint16
	case uint32:
		return qUint32
	case uint64:
		return qUint64
	}
	return
}

func stypeof(nm string, val interface{}) string {
	if t := typeof(val); t != 0 {
		return fmt.Sprintf("%c%s", t, nm)
	}

	switch val.(type) {
	case idealComplex:
		return fmt.Sprintf("c%s", nm)
	case idealFloat:
		return fmt.Sprintf("f%s", nm)
	case idealInt:
		return fmt.Sprintf("l%s", nm)
	case idealRune:
		return fmt.Sprintf("k%s", nm)
	case idealUint:
		return fmt.Sprintf("x%s", nm)
	default:
		return fmt.Sprintf("?%s", nm)
	}
}

func dumpCols(cols []*col) string {
	a := []string{}
	for _, col := range cols {
		a = append(a, fmt.Sprintf("%d:%s %s", col.index, col.name, typeStr(col.typ)))
	}
	return strings.Join(a, ",")
}

func dumpFlds(flds []*fld) string {
	a := []string{}
	for _, fld := range flds {
		a = append(a, fmt.Sprintf("%s AS %s", fld.expr, fld.name))
	}
	return strings.Join(a, ",")
}

func recSetDump(rs Recordset) (s string, err error) {
	var state int
	var a []string
	var flds []*fld
	rs2 := rs.(recordset)
	if err = rs2.do(rs2.ctx, false, func(_ interface{}, rec []interface{}) (bool, error) {
		switch state {
		case 0:
			flds = rec[0].([]*fld)
			state++
		case 1:
			for i, v := range flds {
				a = append(a, stypeof(v.name, rec[i]))
			}
			a = []string{strings.Join(a, ", ")}
			state++
			fallthrough
		default:
			if err = expand(rec); err != nil {
				return false, err
			}

			a = append(a, fmt.Sprintf("%v", rec))
		}
		return true, nil
	}); err != nil {
		return
	}

	if state == 1 {
		for _, v := range flds {
			a = append(a, stypeof(v.name, nil))
		}
		a = []string{strings.Join(a, ", ")}
	}
	return strings.Join(a, "\n"), nil
}

// http://en.wikipedia.org/wiki/Join_(SQL)#Sample_tables
const sample = `
     BEGIN TRANSACTION;
		CREATE TABLE department (
			DepartmentID   int,
			DepartmentName string,
		);

		INSERT INTO department VALUES
			(31, "Sales"),
			(33, "Engineering"),
			(34, "Clerical"),
			(35, "Marketing"),
		;

		CREATE TABLE employee (
			LastName     string,
			DepartmentID int,
		);

		INSERT INTO employee VALUES
			("Rafferty", 31),
			("Jones", 33),
			("Heisenberg", 33),
			("Robinson", 34),
			("Smith", 34),
			("John", NULL),
		;
     COMMIT;
`

// Test provides a testing facility for alternative storage implementations.
// The storef should return freshly created and empty storage. Removing the
// store from the system is the responsibility of the caller. The test only
// guarantees not to panic on recoverable errors and return an error instead.
// Test errors are not returned but reported to t.
func test(t *testing.T, s testDB) (panicked error) {
	defer func() {
		if e := recover(); e != nil {
			switch x := e.(type) {
			case error:
				panicked = x
			default:
				panicked = fmt.Errorf("%v", e)
			}
		}
		if panicked != nil {
			t.Errorf("PANIC: %v\n%s", panicked, debug.Stack())
		}
	}()

	db, err := s.setup()
	if err != nil {
		t.Error(err)
		return
	}

	if err = s.mark(); err != nil {
		t.Error(err)
		return
	}

	defer func() {
		if err = s.teardown(); err != nil {
			t.Error(err)
		}
	}()

	chk := func(test int, err error, expErr string, re *regexp.Regexp) (ok bool) {
		s := err.Error()
		if re == nil {
			t.Error("FAIL: ", test, s)
			return false
		}

		if !re.MatchString(s) {
			t.Error("FAIL: ", test, "error doesn't match:", s, "expected", expErr)
			return false
		}

		return true
	}

	max := len(testdata)
	if n := *oM; n != 0 && n < max {
		max = n
	}
	for itest, test := range testdata[*oN:max] {
		//dbg("---------------------------------------- itest %d", itest)
		var re *regexp.Regexp
		a := strings.Split(test+"|", "|")
		q, rset := a[0], strings.TrimSpace(a[1])
		var expErr string
		if len(a) < 3 {
			t.Error(itest, "internal error 066")
			return
		}

		if expErr = a[2]; expErr != "" {
			re = regexp.MustCompile("(?i:" + strings.TrimSpace(expErr) + ")")
		}

		q = strings.Replace(q, "&or;", "|", -1)
		q = strings.Replace(q, "&oror;", "||", -1)
		list, err := Compile(q)
		if err != nil {
			if !chk(itest, err, expErr, re) {
				return
			}

			continue
		}

		tctx := NewRWCtx()
		if !func() (ok bool) {
			defer func() {
				nfo, err := db.Info()
				if err != nil {
					panic(err)
				}

				for _, tab := range nfo.Tables {
					if _, _, err = db.Run(NewRWCtx(), fmt.Sprintf(`
						BEGIN TRANSACTION;
							DROP table %s;
						COMMIT;
						`,
						tab.Name)); err != nil {
						panic(err)
					}
				}
			}()

			if err = s.mark(); err != nil {
				t.Error(err)
				return
			}

			rs, _, err := db.Execute(tctx, list, int64(30))
			if err != nil {
				return chk(itest, err, expErr, re)
			}

			if rs == nil {
				t.Errorf("FAIL: %d: expected non nil Recordset or error %q", itest, expErr)
				return
			}

			g, err := recSetDump(rs[len(rs)-1])
			if err != nil {
				return chk(itest, err, expErr, re)
			}

			if expErr != "" {
				t.Errorf("FAIL: %d: expected error %q", itest, expErr)
				return
			}

			a = strings.Split(rset, "\n")
			for i, v := range a {
				a[i] = strings.TrimSpace(v)
			}
			e := strings.Join(a, "\n")
			if g != e {
				t.Errorf("FAIL: test # %d\n%s\n---- g\n%s\n---- e\n%s\n----", itest, q, g, e)
				return
			}

			return true
		}() {
			return
		}
	}
	return
}
