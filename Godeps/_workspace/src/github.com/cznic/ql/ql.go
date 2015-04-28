// Copyright 2014 The ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//LATER profile mem
//LATER profile cpu
//LATER coverage

//MAYBE CROSSJOIN (explicit form), LEFT JOIN, INNER JOIN, OUTER JOIN equivalents.

package ql

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"math/big"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cznic/strutil"
)

// NOTE: all rset implementations must be safe for concurrent use by multiple
// goroutines.  If the do method requires any execution domain local data, they
// must be held out of the implementing instance.
var (
	_ rset = (*crossJoinRset)(nil)
	_ rset = (*distinctRset)(nil)
	_ rset = (*groupByRset)(nil)
	_ rset = (*limitRset)(nil)
	_ rset = (*offsetRset)(nil)
	_ rset = (*orderByRset)(nil)
	_ rset = (*selectRset)(nil)
	_ rset = (*selectStmt)(nil)
	_ rset = (*tableRset)(nil)
	_ rset = (*whereRset)(nil)

	isTesting bool // enables test hook: select from an index
)

const (
	noNames = iota
	returnNames
	onlyNames
)

// List represents a group of compiled statements.
type List struct {
	l      []stmt
	params int
}

// String implements fmt.Stringer
func (l List) String() string {
	var b bytes.Buffer
	f := strutil.IndentFormatter(&b, "\t")
	for _, s := range l.l {
		switch s.(type) {
		case beginTransactionStmt:
			f.Format("%s\n%i", s)
		case commitStmt, rollbackStmt:
			f.Format("%u%s\n", s)
		default:
			f.Format("%s\n", s)
		}
	}
	return b.String()
}

type rset interface {
	do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) error
}

type recordset struct {
	ctx *execCtx
	rset
	tx *TCtx
}

func (r recordset) Do(names bool, f func(data []interface{}) (more bool, err error)) (err error) {
	nm := noNames
	if names {
		nm = returnNames
	}
	return r.ctx.db.do(r, nm, f)
}

func (r recordset) Fields() (names []string, err error) {
	err = r.ctx.db.do(
		r,
		onlyNames,
		func(data []interface{}) (more bool, err error) {
			for _, v := range data {
				s, ok := v.(string)
				if !ok {
					return false, fmt.Errorf("got %T(%v), expected string (RecordSet.Fields)", v, v)
				}
				names = append(names, s)
			}
			return false, nil
		},
	)
	return
}

func (r recordset) FirstRow() (row []interface{}, err error) {
	rows, err := r.Rows(1, 0)
	if err != nil {
		return nil, err
	}

	if len(rows) != 0 {
		return rows[0], nil
	}

	return nil, nil
}

func (r recordset) Rows(limit, offset int) (rows [][]interface{}, err error) {
	if err := r.Do(false, func(row []interface{}) (bool, error) {
		if offset > 0 {
			offset--
			return true, nil
		}

		switch {
		case limit < 0:
			rows = append(rows, row)
			return true, nil
		case limit == 0:
			return false, nil
		default: // limit > 0
			rows = append(rows, row)
			limit--
			return limit > 0, nil
		}
	}); err != nil {
		return nil, err
	}

	return rows, nil
}

type groupByRset struct {
	colNames []string
	src      rset
}

func (r *groupByRset) do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	t, err := ctx.db.store.CreateTemp(true)
	if err != nil {
		return
	}

	defer func() {
		if derr := t.Drop(); derr != nil && err == nil {
			err = derr
		}
	}()

	var flds []*fld
	var gcols []*col
	var cols []*col
	ok := false
	k := make([]interface{}, len(r.colNames)) //LATER optimize when len(r.cols) == 0
	if err = r.src.do(ctx, onlyNames, func(rid interface{}, in []interface{}) (more bool, err error) {
		if ok {
			infer(in, &cols)
			for i, c := range gcols {
				k[i] = in[c.index]
			}
			h0, err := t.Get(k)
			if err != nil {
				return false, err
			}

			var h int64
			if len(h0) != 0 {
				h, _ = h0[0].(int64)
			}
			nh, err := t.Create(append([]interface{}{h, nil}, in...)...)
			if err != nil {
				return false, err
			}

			for i, c := range gcols {
				k[i] = in[c.index]
			}
			err = t.Set(k, []interface{}{nh})
			if err != nil {
				return false, err
			}

			return true, nil
		}

		ok = true
		flds = in[0].([]*fld)
		for _, c := range r.colNames {
			i := findFldIndex(flds, c)
			if i < 0 {
				return false, fmt.Errorf("unknown column %s", c)
			}

			gcols = append(gcols, &col{name: c, index: i})
		}
		return !onlyNames, nil
	}); err != nil {
		return
	}

	if onlyNames {
		_, err := f(nil, []interface{}{flds})
		return err
	}

	it, err := t.SeekFirst()
	if err != nil {
		return noEOF(err)
	}

	for i, v := range flds {
		cols[i].name = v.name
		cols[i].index = i
	}

	var data []interface{}
	var more bool
	for more, err = f(nil, []interface{}{t, cols}); more && err == nil; more, err = f(nil, data) {
		_, data, err = it.Next()
		if err != nil {
			return noEOF(err)
		}
	}
	return err
}

// TCtx represents transaction context. It enables to execute multiple
// statement lists in the same context. The same context guarantees the state
// of the DB cannot change in between the separated executions.
//
// LastInsertID
//
// LastInsertID is updated by INSERT INTO statements. The value considers
// performed ROLLBACK statements, if any, even though roll backed IDs are not
// reused. QL clients should treat the field as read only.
//
// RowsAffected
//
// RowsAffected is updated by INSERT INTO, DELETE FROM and UPDATE statements.
// The value does not (yet) consider any ROLLBACK statements involved.  QL
// clients should treat the field as read only.
type TCtx struct {
	LastInsertID int64
	RowsAffected int64
}

// NewRWCtx returns a new read/write transaction context.  NewRWCtx is safe for
// concurrent use by multiple goroutines, every one of them will get a new,
// unique conext.
func NewRWCtx() *TCtx { return &TCtx{} }

// Recordset is a result of a select statment. It can call a user function for
// every row (record) in the set using the Do method.
//
// Recordsets can be safely reused. Evaluation of the rows is performed lazily.
// Every invocation of Do will see the current, potentially actualized data.
//
// Do
//
// Do will call f for every row (record) in the Recordset.
//
// If f returns more == false or err != nil then f will not be called for any
// remaining rows in the set and the err value is returned from Do.
//
// If names == true then f is firstly called with a virtual row
// consisting of field (column) names of the RecordSet.
//
// Do is executed in a read only context and performs a RLock of the
// database.
//
// Do is safe for concurrent use by multiple goroutines.
//
// Fields
//
// The only reliable way, in the general case, how to get field names of a
// recordset is to execute the Do method with the names parameter set to true.
// Any SELECT can return different fields on different runs, provided the
// columns of some of the underlying tables involved were altered in between
// and the query sports the SELECT * form.  Then the fields are not really
// known until the first query result row materializes.  The problem is that
// some queries can be costly even before that first row is computed.  If only
// the field names is what is required in some situation then executing such
// costly query could be prohibitively expensive.
//
// The Fields method provides an alternative. It computes the recordset fields
// while ignoring table data, WHERE clauses, predicates and without evaluating
// any expressions nor any functions.
//
// The result of Fields can be obviously imprecise if tables are altered before
// running Do later. In exchange, calling Fields is cheap - compared to
// actually computing a first row of a query having, say cross joins on n
// relations (1^n is always 1, n ∈ N).
//
// FirstRow
//
// FirstRow will return the first row of the RecordSet or an error, if any. If
// the Recordset has no rows the result is (nil, nil).
//
// Rows
//
// Rows will return rows in Recordset or an error, if any. The semantics of
// limit and offset are the same as of the LIMIT and OFFSET clauses of the
// SELECT statement. To get all rows pass limit < 0. If there are no rows to
// return the result is (nil, nil).
type Recordset interface {
	Do(names bool, f func(data []interface{}) (more bool, err error)) error
	Fields() (names []string, err error)
	FirstRow() (row []interface{}, err error)
	Rows(limit, offset int) (rows [][]interface{}, err error)
}

type assignment struct {
	colName string
	expr    expression
}

func (a *assignment) String() string {
	return fmt.Sprintf("%s=%s", a.colName, a.expr)
}

type distinctRset struct {
	src rset
}

func (r *distinctRset) do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	t, err := ctx.db.store.CreateTemp(true)
	if err != nil {
		return
	}

	defer func() {
		if derr := t.Drop(); derr != nil && err == nil {
			err = derr
		}
	}()

	var flds []*fld
	ok := false
	if err = r.src.do(ctx, onlyNames, func(id interface{}, in []interface{}) (more bool, err error) {
		if ok {
			if err = t.Set(in, nil); err != nil {
				return false, err
			}

			return true, nil
		}

		flds = in[0].([]*fld)
		ok = true
		return true && !onlyNames, nil
	}); err != nil {
		return
	}

	if onlyNames {
		_, err := f(nil, []interface{}{flds})
		return noEOF(err)
	}

	it, err := t.SeekFirst()
	if err != nil {
		return noEOF(err)
	}

	var data []interface{}
	var more bool
	for more, err = f(nil, []interface{}{flds}); more && err == nil; more, err = f(nil, data) {
		data, _, err = it.Next()
		if err != nil {
			return noEOF(err)
		}
	}
	return err
}

type orderByRset struct {
	asc bool
	by  []expression
	src rset
}

func (r *orderByRset) String() string {
	a := make([]string, len(r.by))
	for i, v := range r.by {
		a[i] = v.String()
	}
	s := strings.Join(a, ", ")
	if !r.asc {
		s += " DESC"
	}
	return s
}

func (r *orderByRset) do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	t, err := ctx.db.store.CreateTemp(r.asc)
	if err != nil {
		return
	}

	defer func() {
		if derr := t.Drop(); derr != nil && err == nil {
			err = derr
		}
	}()

	m := map[interface{}]interface{}{}
	var flds []*fld
	ok := false
	k := make([]interface{}, len(r.by)+1)
	id := int64(-1)
	if err = r.src.do(ctx, onlyNames, func(rid interface{}, in []interface{}) (more bool, err error) {
		id++
		if ok {
			for i, fld := range flds {
				if nm := fld.name; nm != "" {
					m[nm] = in[i]
				}
			}
			m["$id"] = rid
			for i, expr := range r.by {
				val, err := expr.eval(ctx, m, ctx.arg)
				if err != nil {
					return false, err
				}

				if val != nil {
					val, ordered, err := isOrderedType(val)
					if err != nil {
						return false, err
					}

					if !ordered {
						return false, fmt.Errorf("cannot order by %v (type %T)", val, val)

					}
				}

				k[i] = val
			}
			k[len(r.by)] = id
			if err = t.Set(k, in); err != nil {
				return false, err
			}

			return true, nil
		}

		ok = true
		flds = in[0].([]*fld)
		return true && !onlyNames, nil
	}); err != nil {
		return
	}

	if onlyNames {
		_, err = f(nil, []interface{}{flds})
		return noEOF(err)
	}

	it, err := t.SeekFirst()
	if err != nil {
		if err != io.EOF {
			return err
		}

		_, err = f(nil, []interface{}{flds})
		return err
	}

	var data []interface{}
	var more bool
	for more, err = f(nil, []interface{}{flds}); more && err == nil; more, err = f(nil, data) {
		_, data, err = it.Next()
		if err != nil {
			return noEOF(err)
		}
	}
	return
}

var nowhere = &whereRset{}

type whereRset struct {
	expr expression
	src  rset
}

func (r *whereRset) doIndexedBool(t *table, en indexIterator, v bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	m, err := f(nil, []interface{}{t.flds()})
	if !m || err != nil {
		return
	}

	for {
		k, h, err := en.Next()
		if err != nil {
			return noEOF(err)
		}

		switch x := k.(type) {
		case nil:
			panic("internal error 052") // nil should sort before true
		case bool:
			if x != v {
				return nil
			}
		}

		if _, err := tableRset("").doOne(t, h, f); err != nil {
			return err
		}
	}
}

func (r *whereRset) tryBinOp(execCtx *execCtx, t *table, id *ident, v value, op int, f func(id interface{}, data []interface{}) (more bool, err error)) (bool, error) {
	c := findCol(t.cols0, id.s)
	if c == nil {
		return false, fmt.Errorf("undefined column: %s", id.s)
	}

	xCol := t.indices[c.index+1]
	if xCol == nil { // no index for this column
		return false, nil
	}

	data := []interface{}{v.val}
	cc := *c
	cc.index = 0
	if err := typeCheck(data, []*col{&cc}); err != nil {
		return true, err
	}

	v.val = data[0]
	ex := &binaryOperation{op, nil, v}
	switch op {
	case '<', le:
		v.val = false // first value collating after nil
		fallthrough
	case eq, ge:
		m, err := f(nil, []interface{}{t.flds()})
		if !m || err != nil {
			return true, err
		}

		en, _, err := xCol.x.Seek(v.val)
		if err != nil {
			return true, noEOF(err)
		}

		for {
			k, h, err := en.Next()
			if k == nil {
				return true, nil
			}

			if err != nil {
				return true, noEOF(err)
			}

			ex.l = value{k}
			eval, err := ex.eval(execCtx, nil, nil)
			if err != nil {
				return true, err
			}

			if !eval.(bool) {
				return true, nil
			}

			if _, err := tableRset("").doOne(t, h, f); err != nil {
				return true, err
			}
		}
	case '>':
		m, err := f(nil, []interface{}{t.flds()})
		if !m || err != nil {
			return true, err
		}

		en, err := xCol.x.SeekLast()
		if err != nil {
			return true, noEOF(err)
		}

		for {
			k, h, err := en.Prev()
			if k == nil {
				return true, nil
			}

			if err != nil {
				return true, noEOF(err)
			}

			ex.l = value{k}
			eval, err := ex.eval(execCtx, nil, nil)
			if err != nil {
				return true, err
			}

			if !eval.(bool) {
				return true, nil
			}

			if _, err := tableRset("").doOne(t, h, f); err != nil {
				return true, err
			}
		}
	default:
		panic("internal error 053")
	}
}

func (r *whereRset) tryBinOpID(execCtx *execCtx, t *table, v value, op int, f func(id interface{}, data []interface{}) (more bool, err error)) (bool, error) {
	xCol := t.indices[0]
	if xCol == nil { // no index for id()
		return false, nil
	}

	data := []interface{}{v.val}
	if err := typeCheck(data, []*col{&col{typ: qInt64}}); err != nil {
		return true, err
	}

	v.val = data[0]
	ex := &binaryOperation{op, nil, v}
	switch op {
	case '<', le:
		v.val = int64(1)
		fallthrough
	case eq, ge:
		m, err := f(nil, []interface{}{t.flds()})
		if !m || err != nil {
			return true, err
		}

		en, _, err := xCol.x.Seek(v.val)
		if err != nil {
			return true, noEOF(err)
		}

		for {
			k, h, err := en.Next()
			if k == nil {
				return true, nil
			}

			if err != nil {
				return true, noEOF(err)
			}

			ex.l = value{k}
			eval, err := ex.eval(execCtx, nil, nil)
			if err != nil {
				return true, err
			}

			if !eval.(bool) {
				return true, nil
			}

			if _, err := tableRset("").doOne(t, h, f); err != nil {
				return true, err
			}
		}
	case '>':
		m, err := f(nil, []interface{}{t.flds()})
		if !m || err != nil {
			return true, err
		}

		en, err := xCol.x.SeekLast()
		if err != nil {
			return true, noEOF(err)
		}

		for {
			k, h, err := en.Prev()
			if k == nil {
				return true, nil
			}

			if err != nil {
				return true, noEOF(err)
			}

			ex.l = value{k}
			eval, err := ex.eval(execCtx, nil, nil)
			if err != nil {
				return true, err
			}

			if !eval.(bool) {
				return true, nil
			}

			if _, err := tableRset("").doOne(t, h, f); err != nil {
				return true, err
			}
		}
	default:
		panic("internal error 071")
	}
}

func (r *whereRset) tryUseIndex(ctx *execCtx, f func(id interface{}, data []interface{}) (more bool, err error)) (bool, error) {
	//TODO(indices) support IS [NOT] NULL
	c, ok := r.src.(*crossJoinRset)
	if !ok {
		return false, nil
	}

	tabName, ok := c.isSingleTable()
	if !ok || isSystemName[tabName] {
		return false, nil
	}

	t := ctx.db.root.tables[tabName]
	if t == nil {
		return true, fmt.Errorf("table %s does not exist", tabName)
	}

	if !t.hasIndices() {
		return false, nil
	}

	//LATER WHERE column1 boolOp column2 ...
	//LATER WHERE !column (rewritable as: column == false)
	switch ex := r.expr.(type) {
	case *unaryOperation: // WHERE !column
		if ex.op != '!' {
			return false, nil
		}

		switch operand := ex.v.(type) {
		case *ident:
			c := findCol(t.cols0, operand.s)
			if c == nil { // no such column
				return false, fmt.Errorf("unknown column %s", ex)
			}

			if c.typ != qBool { // not a bool column
				return false, nil
			}

			xCol := t.indices[c.index+1]
			if xCol == nil { // column isn't indexed
				return false, nil
			}

			en, _, err := xCol.x.Seek(false)
			if err != nil {
				return false, noEOF(err)
			}

			return true, r.doIndexedBool(t, en, false, f)
		default:
			return false, nil
		}
	case *ident: // WHERE column
		c := findCol(t.cols0, ex.s)
		if c == nil { // no such column
			return false, fmt.Errorf("unknown column %s", ex)
		}

		if c.typ != qBool { // not a bool column
			return false, nil
		}

		xCol := t.indices[c.index+1]
		if xCol == nil { // column isn't indexed
			return false, nil
		}

		en, _, err := xCol.x.Seek(true)
		if err != nil {
			return false, noEOF(err)
		}

		return true, r.doIndexedBool(t, en, true, f)
	case *binaryOperation:
		//DONE handle id()
		var invOp int
		switch ex.op {
		case '<':
			invOp = '>'
		case le:
			invOp = ge
		case eq:
			invOp = eq
		case '>':
			invOp = '<'
		case ge:
			invOp = le
		default:
			return false, nil
		}

		switch lhs := ex.l.(type) {
		case *call:
			if !(lhs.f == "id" && len(lhs.arg) == 0) {
				return false, nil
			}

			switch rhs := ex.r.(type) {
			case parameter:
				v, err := rhs.eval(ctx, nil, ctx.arg)
				if err != nil {
					return false, err
				}

				return r.tryBinOpID(ctx, t, value{v}, ex.op, f)
			case value:
				return r.tryBinOpID(ctx, t, rhs, ex.op, f)
			default:
				return false, nil
			}
		case *ident:
			switch rhs := ex.r.(type) {
			case parameter:
				v, err := rhs.eval(ctx, nil, ctx.arg)
				if err != nil {
					return false, err
				}

				return r.tryBinOp(ctx, t, lhs, value{v}, ex.op, f)
			case value:
				return r.tryBinOp(ctx, t, lhs, rhs, ex.op, f)
			default:
				return false, nil
			}
		case parameter:
			switch rhs := ex.r.(type) {
			case *call:
				if !(rhs.f == "id" && len(rhs.arg) == 0) {
					return false, nil
				}

				v, err := lhs.eval(ctx, nil, ctx.arg)
				if err != nil {
					return false, err
				}

				return r.tryBinOpID(ctx, t, value{v}, invOp, f)
			case *ident:
				v, err := lhs.eval(ctx, nil, ctx.arg)
				if err != nil {
					return false, err
				}

				return r.tryBinOp(ctx, t, rhs, value{v}, invOp, f)
			default:
				return false, nil
			}
		case value:
			switch rhs := ex.r.(type) {
			case *call:
				if !(rhs.f == "id" && len(rhs.arg) == 0) {
					return false, nil
				}

				return r.tryBinOpID(ctx, t, lhs, invOp, f)
			case *ident:
				return r.tryBinOp(ctx, t, rhs, lhs, invOp, f)
			default:
				return false, nil
			}
		default:
			return false, nil
		}
	default:
		return false, nil
	}
}

func (r *whereRset) do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	//dbg("====")
	if !onlyNames {
		if ok, err := r.tryUseIndex(ctx, f); ok || err != nil {
			//dbg("ok %t, err %v", ok, err)
			return err
		}
	}

	//dbg("not using indices")
	m := map[interface{}]interface{}{}
	var flds []*fld
	ok := false
	return r.src.do(ctx, onlyNames, func(rid interface{}, in []interface{}) (more bool, err error) {
		if ok {
			for i, fld := range flds {
				if nm := fld.name; nm != "" {
					m[nm] = in[i]
				}
			}
			m["$id"] = rid
			val, err := r.expr.eval(ctx, m, ctx.arg)
			if err != nil {
				return false, err
			}

			if val == nil {
				return true, nil
			}

			x, ok := val.(bool)
			if !ok {
				return false, fmt.Errorf("invalid WHERE expression %s (value of type %T)", val, val)
			}

			if !x {
				return true, nil
			}

			return f(rid, in)
		}

		flds = in[0].([]*fld)
		ok = true
		m, err := f(nil, in)
		return m && !onlyNames, err
	})
}

type offsetRset struct {
	expr expression
	src  rset
}

func (r *offsetRset) do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	m := map[interface{}]interface{}{}
	var flds []*fld
	var ok, eval bool
	var off uint64
	return r.src.do(ctx, onlyNames, func(rid interface{}, in []interface{}) (more bool, err error) {
		if ok {
			if !eval {
				for i, fld := range flds {
					if nm := fld.name; nm != "" {
						m[nm] = in[i]
					}
				}
				m["$id"] = rid
				val, err := r.expr.eval(ctx, m, ctx.arg)
				if err != nil {
					return false, err
				}

				if val == nil {
					return true, nil
				}

				if off, err = limOffExpr(val); err != nil {
					return false, err
				}

				eval = true
			}
			if off > 0 {
				off--
				return true, nil
			}

			return f(rid, in)
		}

		flds = in[0].([]*fld)
		ok = true
		m, err := f(nil, in)
		return m && !onlyNames, err
	})
}

type limitRset struct {
	expr expression
	src  rset
}

func (r *limitRset) do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	m := map[interface{}]interface{}{}
	var flds []*fld
	var ok, eval bool
	var lim uint64
	return r.src.do(ctx, onlyNames, func(rid interface{}, in []interface{}) (more bool, err error) {
		if ok {
			if !eval {
				for i, fld := range flds {
					if nm := fld.name; nm != "" {
						m[nm] = in[i]
					}
				}
				m["$id"] = rid
				val, err := r.expr.eval(ctx, m, ctx.arg)
				if err != nil {
					return false, err
				}

				if val == nil {
					return true, nil
				}

				if lim, err = limOffExpr(val); err != nil {
					return false, err
				}

				eval = true
			}
			switch lim {
			case 0:
				return false, nil
			default:
				lim--
				return f(rid, in)
			}
		}

		flds = in[0].([]*fld)
		ok = true
		m, err := f(nil, in)
		return m && !onlyNames, err
	})
}

type selectRset struct {
	flds []*fld
	src  rset
}

func (r *selectRset) doGroup(grp *groupByRset, ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	if onlyNames {
		if len(r.flds) != 0 {
			_, err := f(nil, []interface{}{r.flds})
			return err
		}

		return grp.do(ctx, true, f)
	}

	var t temp
	var cols []*col
	out := make([]interface{}, len(r.flds))
	ok := false
	rows := 0
	if err = r.src.do(ctx, onlyNames, func(rid interface{}, in []interface{}) (more bool, err error) {
		if ok {
			h := in[0].(int64)
			m := map[interface{}]interface{}{}
			for h != 0 {
				in, err = t.Read(nil, h, cols...)
				if err != nil {
					return false, err
				}

				rec := in[2:]
				for i, c := range cols {
					if nm := c.name; nm != "" {
						m[nm] = rec[i]
					}
				}
				m["$id"] = rid
				for _, fld := range r.flds {
					if _, err = fld.expr.eval(ctx, m, ctx.arg); err != nil {
						return false, err
					}
				}

				h = in[0].(int64)
			}
			m["$agg"] = true
			for i, fld := range r.flds {
				if out[i], err = fld.expr.eval(ctx, m, ctx.arg); err != nil {
					return false, err
				}
			}
			rows++
			return f(nil, out)
		}

		ok = true
		rows++
		t = in[0].(temp)
		cols = in[1].([]*col)
		if len(r.flds) == 0 {
			r.flds = make([]*fld, len(cols))
			for i, v := range cols {
				r.flds[i] = &fld{expr: &ident{v.name}, name: v.name}
			}
			out = make([]interface{}, len(r.flds))
		}
		m, err := f(nil, []interface{}{r.flds})
		return m && !onlyNames, err
	}); err != nil || onlyNames {
		return
	}

	switch rows {
	case 0:
		more, err := f(nil, []interface{}{r.flds})
		if !more || err != nil {
			return err
		}

		fallthrough
	case 1:
		m := map[interface{}]interface{}{"$agg0": true} // aggregate empty record set
		for i, fld := range r.flds {
			if out[i], err = fld.expr.eval(ctx, m, ctx.arg); err != nil {
				return
			}
		}
		_, err = f(nil, out)
	}
	return
}

func (r *selectRset) do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	if grp, ok := r.src.(*groupByRset); ok {
		return r.doGroup(grp, ctx, onlyNames, f)
	}

	if len(r.flds) == 0 {
		return r.src.do(ctx, onlyNames, f)
	}

	if onlyNames {
		_, err := f(nil, []interface{}{r.flds})
		return err
	}

	var flds []*fld
	m := map[interface{}]interface{}{}
	ok := false
	return r.src.do(ctx, onlyNames, func(rid interface{}, in []interface{}) (more bool, err error) {
		if ok {
			for i, fld := range flds {
				if nm := fld.name; nm != "" {
					m[nm] = in[i]
				}
			}
			m["$id"] = rid
			out := make([]interface{}, len(r.flds))
			for i, fld := range r.flds {
				if out[i], err = fld.expr.eval(ctx, m, ctx.arg); err != nil {
					return false, err
				}
			}
			m, err := f(rid, out)
			return m, err

		}

		ok = true
		flds = in[0].([]*fld)
		m, err := f(nil, []interface{}{r.flds})
		return m && !onlyNames, err
	})
}

type tableRset string

func (r tableRset) doIndex(x *indexedCol, ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	flds := []*fld{&fld{name: x.name}}
	m, err := f(nil, []interface{}{flds})
	if onlyNames {
		return err
	}

	if !m || err != nil {
		return
	}

	en, _, err := x.x.Seek(nil)
	if err != nil {
		return err
	}

	var id int64
	rec := []interface{}{nil}
	for {
		k, _, err := en.Next()
		if err != nil {
			return noEOF(err)
		}

		id++
		rec[0] = k
		m, err := f(id, rec)
		if !m || err != nil {
			return err
		}
	}
}

func (tableRset) doOne(t *table, h int64, f func(id interface{}, data []interface{}) (more bool, err error)) ( /* next handle */ int64, error) {
	cols := t.cols
	ncols := len(cols)
	rec, err := t.store.Read(nil, h, cols...)
	if err != nil {
		return -1, err
	}

	h = rec[0].(int64)
	if n := ncols + 2 - len(rec); n > 0 {
		rec = append(rec, make([]interface{}, n)...)
	}

	for i, c := range cols {
		if x := c.index; 2+x < len(rec) {
			rec[2+i] = rec[2+x]
			continue
		}

		rec[2+i] = nil //DONE +test (#571)
	}
	m, err := f(rec[1], rec[2:2+ncols]) // 0:next, 1:id
	if !m || err != nil {
		return -1, err
	}

	return h, nil
}

func (r tableRset) doSysTable(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	flds := []*fld{&fld{name: "Name"}, &fld{name: "Schema"}}
	m, err := f(nil, []interface{}{flds})
	if onlyNames {
		return err
	}

	if !m || err != nil {
		return
	}

	rec := make([]interface{}, 2)
	di, err := ctx.db.info()
	if err != nil {
		return err
	}

	var id int64
	for _, ti := range di.Tables {
		rec[0] = ti.Name
		a := []string{}
		for _, ci := range ti.Columns {
			a = append(a, fmt.Sprintf("%s %s", ci.Name, ci.Type))
		}
		rec[1] = fmt.Sprintf("CREATE TABLE %s (%s);", ti.Name, strings.Join(a, ", "))
		id++
		m, err := f(id, rec)
		if !m || err != nil {
			return err
		}
	}
	return
}

func (r tableRset) doSysColumn(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	flds := []*fld{&fld{name: "TableName"}, &fld{name: "Ordinal"}, &fld{name: "Name"}, &fld{name: "Type"}}
	m, err := f(nil, []interface{}{flds})
	if onlyNames {
		return err
	}

	if !m || err != nil {
		return
	}

	rec := make([]interface{}, 4)
	di, err := ctx.db.info()
	if err != nil {
		return err
	}

	var id int64
	for _, ti := range di.Tables {
		rec[0] = ti.Name
		var ix int64
		for _, ci := range ti.Columns {
			ix++
			rec[1] = ix
			rec[2] = ci.Name
			rec[3] = ci.Type.String()
			id++
			m, err := f(id, rec)
			if !m || err != nil {
				return err
			}
		}
	}
	return
}

func (r tableRset) doSysIndex(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	flds := []*fld{&fld{name: "TableName"}, &fld{name: "ColumnName"}, &fld{name: "Name"}, &fld{name: "IsUnique"}}
	m, err := f(nil, []interface{}{flds})
	if onlyNames {
		return err
	}

	if !m || err != nil {
		return
	}

	rec := make([]interface{}, 4)
	di, err := ctx.db.info()
	if err != nil {
		return err
	}

	var id int64
	for _, xi := range di.Indices {
		rec[0] = xi.Table
		rec[1] = xi.Column
		rec[2] = xi.Name
		rec[3] = xi.Unique
		id++
		m, err := f(id, rec)
		if !m || err != nil {
			return err
		}
	}
	return
}

func (r tableRset) do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	switch r {
	case "__Table":
		return r.doSysTable(ctx, onlyNames, f)
	case "__Column":
		return r.doSysColumn(ctx, onlyNames, f)
	case "__Index":
		return r.doSysIndex(ctx, onlyNames, f)
	}

	t, ok := ctx.db.root.tables[string(r)]
	var x *indexedCol
	if !ok && isTesting {
		if _, x = ctx.db.root.findIndexByName(string(r)); x != nil {
			return r.doIndex(x, ctx, onlyNames, f)
		}
	}

	if !ok {
		return fmt.Errorf("table %s does not exist", r)
	}

	m, err := f(nil, []interface{}{t.flds()})
	if onlyNames {
		return err
	}

	if !m || err != nil {
		return
	}

	for h := t.head; h > 0 && err == nil; h, err = r.doOne(t, h, f) {
	}
	return
}

type crossJoinRset struct {
	sources []interface{}
}

func (r *crossJoinRset) tables() []struct {
	i            int
	name, rename string
} {
	var ret []struct {
		i            int
		name, rename string
	}
	//dbg("---- %p", r)
	for i, pair0 := range r.sources {
		//dbg("%d/%d, %#v", i, len(r.sources), pair0)
		pair := pair0.([]interface{})
		altName := pair[1].(string)
		switch x := pair[0].(type) {
		case string: // table name
			if altName == "" {
				altName = x
			}
			ret = append(ret, struct {
				i            int
				name, rename string
			}{i, x, altName})
		}
	}
	return ret
}

func (r *crossJoinRset) String() string {
	a := make([]string, len(r.sources))
	for i, pair0 := range r.sources {
		pair := pair0.([]interface{})
		altName := pair[1].(string)
		switch x := pair[0].(type) {
		case string: // table name
			switch {
			case altName == "":
				a[i] = x
			default:
				a[i] = fmt.Sprintf("%s AS %s", x, altName)
			}
		case *selectStmt:
			switch {
			case altName == "":
				a[i] = fmt.Sprintf("(%s)", x)
			default:
				a[i] = fmt.Sprintf("(%s) AS %s", x, altName)
			}
		default:
			log.Panic("internal error 054")
		}
	}
	return strings.Join(a, ", ")
}

func (r *crossJoinRset) isSingleTable() (string, bool) {
	sources := r.sources
	if len(sources) != 1 {
		return "", false
	}

	pair := sources[0].([]interface{})
	s, ok := pair[0].(string)
	return s, ok
}

func (r *crossJoinRset) do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	rsets := make([]rset, len(r.sources))
	altNames := make([]string, len(r.sources))
	//dbg(".... %p", r)
	for i, pair0 := range r.sources {
		pair := pair0.([]interface{})
		//dbg("%d: %#v", len(pair), pair)
		altName := pair[1].(string)
		switch x := pair[0].(type) {
		case string: // table name
			rsets[i] = tableRset(x)
			if altName == "" {
				altName = x
			}
		case *selectStmt:
			rsets[i] = x
		default:
			log.Panic("internal error 055")
		}
		altNames[i] = altName
	}

	if len(rsets) == 1 {
		return rsets[0].do(ctx, onlyNames, f)
	}

	var flds []*fld
	fldsSent := false
	iq := 0
	stop := false
	ids := map[string]interface{}{}
	var g func([]interface{}, []rset, int) error
	g = func(prefix []interface{}, rsets []rset, x int) (err error) {
		rset := rsets[0]
		rsets = rsets[1:]
		ok := false
		return rset.do(ctx, onlyNames, func(id interface{}, in []interface{}) (more bool, err error) {
			if onlyNames && fldsSent {
				stop = true
				return false, nil
			}

			if ok {
				ids[altNames[x]] = id
				if len(rsets) != 0 {
					return true, g(append(prefix, in...), rsets, x+1)
				}

				m, err := f(ids, append(prefix, in...))
				if !m {
					stop = true
				}
				return m && !stop, err
			}

			ok = true
			if !fldsSent {
				f0 := append([]*fld(nil), in[0].([]*fld)...)
				q := altNames[iq]
				for i, elem := range f0 {
					nf := &fld{}
					*nf = *elem
					switch {
					case q == "":
						nf.name = ""
					case nf.name != "":
						nf.name = fmt.Sprintf("%s.%s", altNames[iq], nf.name)
					}
					f0[i] = nf
				}
				iq++
				flds = append(flds, f0...)
			}
			if len(rsets) == 0 && !fldsSent {
				fldsSent = true
				more, err = f(nil, []interface{}{flds})
				if !more {
					stop = true
				}
				return more && !stop, err
			}

			return !stop, nil
		})
	}
	return g(nil, rsets, 0)
}

type fld struct {
	expr expression
	name string
}

func findFldIndex(fields []*fld, name string) int {
	for i, f := range fields {
		if f.name == name {
			return i
		}
	}

	return -1
}

func findFld(fields []*fld, name string) (f *fld) {
	for _, f = range fields {
		if f.name == name {
			return
		}
	}

	return nil
}

type col struct {
	index int
	name  string
	typ   int
}

func findCol(cols []*col, name string) (c *col) {
	for _, c = range cols {
		if c.name == name {
			return
		}
	}

	return nil
}

func (f *col) typeCheck(x interface{}) (ok bool) { //NTYPE
	switch x.(type) {
	case nil:
		return true
	case bool:
		return f.typ == qBool
	case complex64:
		return f.typ == qComplex64
	case complex128:
		return f.typ == qComplex128
	case float32:
		return f.typ == qFloat32
	case float64:
		return f.typ == qFloat64
	case int8:
		return f.typ == qInt8
	case int16:
		return f.typ == qInt16
	case int32:
		return f.typ == qInt32
	case int64:
		return f.typ == qInt64
	case string:
		return f.typ == qString
	case uint8:
		return f.typ == qUint8
	case uint16:
		return f.typ == qUint16
	case uint32:
		return f.typ == qUint32
	case uint64:
		return f.typ == qUint64
	case []byte:
		return f.typ == qBlob
	case *big.Int:
		return f.typ == qBigInt
	case *big.Rat:
		return f.typ == qBigRat
	case time.Time:
		return f.typ == qTime
	case time.Duration:
		return f.typ == qDuration
	case chunk:
		return true // was checked earlier
	}
	return
}

func cols2meta(f []*col) (s string) {
	a := []string{}
	for _, f := range f {
		a = append(a, string(f.typ)+f.name)
	}
	return strings.Join(a, "|")
}

// DB represent the database capable of executing QL statements.
type DB struct {
	cc    *TCtx // Current transaction context
	isMem bool
	mu    sync.Mutex
	root  *root
	rw    bool // DB FSM
	rwmu  sync.RWMutex
	store storage
	tnl   int // Transaction nesting level
}

func newDB(store storage) (db *DB, err error) {
	db0 := &DB{
		store: store,
	}
	if db0.root, err = newRoot(store); err != nil {
		return
	}

	return db0, nil
}

// Name returns the name of the DB.
func (db *DB) Name() string { return db.store.Name() }

// Run compiles and executes a statement list.  It returns, if applicable, a
// RecordSet slice and/or an index and error.
//
// For more details please see DB.Execute
//
// Run is safe for concurrent use by multiple goroutines.
func (db *DB) Run(ctx *TCtx, ql string, arg ...interface{}) (rs []Recordset, index int, err error) {
	l, err := Compile(ql)
	if err != nil {
		return nil, -1, err
	}

	return db.Execute(ctx, l, arg...)
}

// Compile parses the ql statements from src and returns a compiled list for
// DB.Execute or an error if any.
//
// Compile is safe for concurrent use by multiple goroutines.
func Compile(src string) (List, error) {
	l := newLexer(src)
	if yyParse(l) != 0 {
		return List{}, l.errs[0]
	}

	return List{l.list, l.params}, nil
}

// MustCompile is like Compile but panics if the ql statements in src cannot be
// compiled. It simplifies safe initialization of global variables holding
// compiled statement lists for DB.Execute.
//
// MustCompile is safe for concurrent use by multiple goroutines.
func MustCompile(src string) List {
	list, err := Compile(src)
	if err != nil {
		panic("ql: Compile(" + strconv.Quote(src) + "): " + err.Error()) // panic ok here
	}

	return list
}

// Execute executes statements in a list while substituting QL paramaters from
// arg.
//
// The resulting []Recordset corresponds to the SELECT FROM statements in the
// list.
//
// If err != nil then index is the zero based index of the failed QL statement.
// Empty statements do not count.
//
// The FSM STT describing the relations between DB states, statements and the
// ctx parameter.
//
//  +-----------+---------------------+------------------+------------------+------------------+
//  |\  Event   |                     |                  |                  |                  |
//  | \-------\ |     BEGIN           |                  |                  |    Other         |
//  |   State  \|     TRANSACTION     |      COMMIT      |     ROLLBACK     |    statement     |
//  +-----------+---------------------+------------------+------------------+------------------+
//  | RD        | if PC == nil        | return error     | return error     | DB.RLock         |
//  |           |     return error    |                  |                  | Execute(1)       |
//  | CC == nil |                     |                  |                  | DB.RUnlock       |
//  | TNL == 0  | DB.Lock             |                  |                  |                  |
//  |           | CC = PC             |                  |                  |                  |
//  |           | TNL++               |                  |                  |                  |
//  |           | DB.BeginTransaction |                  |                  |                  |
//  |           | State = WR          |                  |                  |                  |
//  +-----------+---------------------+------------------+------------------+------------------+
//  | WR        | if PC == nil        | if PC != CC      | if PC != CC      | if PC == nil     |
//  |           |     return error    |     return error |     return error |     DB.Rlock     |
//  | CC != nil |                     |                  |                  |     Execute(1)   |
//  | TNL != 0  | if PC != CC         | DB.Commit        | DB.Rollback      |     RUnlock      |
//  |           |     DB.Lock         | TNL--            | TNL--            | else if PC != CC |
//  |           |     CC = PC         | if TNL == 0      | if TNL == 0      |     return error |
//  |           |                     |     CC = nil     |     CC = nil     | else             |
//  |           | TNL++               |     State = RD   |     State = RD   |     Execute(2)   |
//  |           | DB.BeginTransaction |     DB.Unlock    |     DB.Unlock    |                  |
//  +-----------+---------------------+------------------+------------------+------------------+
//  CC: Curent transaction context
//  PC: Passed transaction context
//  TNL: Transaction nesting level
//
// Lock, Unlock, RLock, RUnlock semantics above are the same as in
// sync.RWMutex.
//
// (1): Statement list is executed outside of a transaction. Attempts to update
// the DB will fail, the execution context is read-only. Other statements with
// read only context will execute concurrently. If any statement fails, the
// execution of the statement list is aborted.
//
// Note that the RLock/RUnlock surrounds every single "other" statement when it
// is executed outside of a transaction. If read consistency is required by a
// list of more than one statement then an explicit BEGIN TRANSACTION / COMMIT
// or ROLLBACK wrapper must be provided. Otherwise the state of the DB may
// change in between executing any two out-of-transaction statements.
//
// (2): Statement list is executed inside an isolated transaction. Execution of
// statements can update the DB, the execution context is read-write. If any
// statement fails, the execution of the statement list is aborted and the DB
// is automatically rolled back to the TNL which was active before the start of
// execution of the statement list.
//
// Execute is safe for concurrent use by multiple goroutines, but one must
// consider the blocking issues as discussed above.
//
// ACID
//
// Atomicity: Transactions are atomic. Transactions can be nested. Commit or
// rollbacks work on the current transaction level. Transactions are made
// persistent only on the top level commit. Reads made from within an open
// transaction are dirty reads.
//
// Consistency: Transactions bring the DB from one structurally consistent
// state to other structurally consistent state.
//
// Isolation: Transactions are isolated. Isolation is implemented by
// serialization.
//
// Durability: Transactions are durable. A two phase commit protocol and a
// write ahead log is used. Database is recovered after a crash from the write
// ahead log automatically on open.
func (db *DB) Execute(ctx *TCtx, l List, arg ...interface{}) (rs []Recordset, index int, err error) {
	// Sanitize args
	for i, v := range arg {
		switch x := v.(type) {
		case nil, bool, complex64, complex128, float32, float64, string,
			int8, int16, int32, int64, int,
			uint8, uint16, uint32, uint64, uint,
			*big.Int, *big.Rat, []byte, time.Duration, time.Time:
		case big.Int:
			arg[i] = &x
		case big.Rat:
			arg[i] = &x
		default:
			return nil, 0, fmt.Errorf("cannot use arg[%d] (type %T):unsupported type", i, v)
		}
	}

	tnl0 := -1
	if ctx != nil {
		ctx.LastInsertID, ctx.RowsAffected = 0, 0
	}

	var s stmt
	for index, s = range l.l {
		r, err := db.run1(ctx, &tnl0, s, arg...)
		if err != nil {
			for tnl0 >= 0 && db.tnl > tnl0 {
				if _, e2 := db.run1(ctx, &tnl0, rollbackStmt{}); e2 != nil {
					err = e2
				}
			}
			return rs, index, err
		}

		if r != nil {
			rs = append(rs, r)
		}
	}
	return
}

func (db *DB) run1(pc *TCtx, tnl0 *int, s stmt, arg ...interface{}) (rs Recordset, err error) {
	//dbg("%v", s)
	db.mu.Lock()
	switch db.rw {
	case false:
		switch s.(type) {
		case beginTransactionStmt:
			defer db.mu.Unlock()
			if pc == nil {
				return nil, errors.New("BEGIN TRANSACTION: cannot start a transaction in nil TransactionCtx")
			}

			if err = db.store.BeginTransaction(); err != nil {
				return
			}

			db.beginTransaction()
			db.rwmu.Lock()
			db.cc = pc
			*tnl0 = db.tnl // 0
			db.tnl++
			db.rw = true
			return
		case commitStmt:
			defer db.mu.Unlock()
			return nil, errCommitNotInTransaction
		case rollbackStmt:
			defer db.mu.Unlock()
			return nil, errRollbackNotInTransaction
		default:
			if s.isUpdating() {
				db.mu.Unlock()
				return nil, fmt.Errorf("attempt to update the DB outside of a transaction")
			}

			db.rwmu.RLock() // can safely grab before Unlock
			db.mu.Unlock()
			defer db.rwmu.RUnlock()
			return s.exec(&execCtx{db, arg}) // R/O tctx
		}
	default: // case true:
		switch s.(type) {
		case beginTransactionStmt:
			defer db.mu.Unlock()

			if pc == nil {
				return nil, errBeginTransNoCtx
			}

			if pc != db.cc {
				for db.rw == true {
					db.mu.Unlock() // Transaction isolation
					db.mu.Lock()
				}

				db.rw = true
				db.rwmu.Lock()
				*tnl0 = db.tnl // 0
			}

			if err = db.store.BeginTransaction(); err != nil {
				return
			}

			db.beginTransaction()
			db.cc = pc
			db.tnl++
			return
		case commitStmt:
			defer db.mu.Unlock()
			if pc != db.cc {
				return nil, fmt.Errorf("invalid passed transaction context")
			}

			db.commit()
			err = db.store.Commit()
			db.tnl--
			if db.tnl != 0 {
				return
			}

			db.cc = nil
			db.rw = false
			db.rwmu.Unlock()
			return
		case rollbackStmt:
			defer db.mu.Unlock()
			defer func() { pc.LastInsertID = db.root.lastInsertID }()
			if pc != db.cc {
				return nil, fmt.Errorf("invalid passed transaction context")
			}

			db.rollback()
			err = db.store.Rollback()
			db.tnl--
			if db.tnl != 0 {
				return
			}

			db.cc = nil
			db.rw = false
			db.rwmu.Unlock()
			return
		default:
			if pc == nil {
				if s.isUpdating() {
					db.mu.Unlock()
					return nil, fmt.Errorf("attempt to update the DB outside of a transaction")
				}

				db.mu.Unlock() // must Unlock before RLock
				db.rwmu.RLock()
				defer db.rwmu.RUnlock()
				return s.exec(&execCtx{db, arg})
			}

			defer db.mu.Unlock()
			defer func() { pc.LastInsertID = db.root.lastInsertID }()
			if pc != db.cc {
				return nil, fmt.Errorf("invalid passed transaction context")
			}

			if !s.isUpdating() {
				return s.exec(&execCtx{db, arg})
			}

			if rs, err = s.exec(&execCtx{db, arg}); err != nil {
				return
			}

			return rs, nil
		}
	}
}

// Flush ends the transaction collecting window, if applicable. IOW, if the DB
// is dirty, it schedules a 2PC (WAL + DB file) commit on the next outer most
// DB.Commit or performs it synchronously if there's currently no open
// transaction.
//
// The collecting window is an implementation detail and future versions of
// Flush may become a no operation while keeping the operation semantics.
func (db *DB) Flush() (err error) {
	return nil
}

// Close will close the DB. Successful Close is idempotent.
func (db *DB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if db.store == nil {
		return nil
	}

	if db.tnl != 0 {
		return fmt.Errorf("cannot close DB while open transaction exist")
	}

	err := db.store.Close()
	db.root, db.store = nil, nil
	return err
}

func (db *DB) do(r recordset, names int, f func(data []interface{}) (more bool, err error)) (err error) {
	db.mu.Lock()
	switch db.rw {
	case false:
		db.rwmu.RLock() // can safely grab before Unlock
		db.mu.Unlock()
		defer db.rwmu.RUnlock()
	default: // case true:
		if r.tx == nil {
			db.mu.Unlock() // must Unlock before RLock
			db.rwmu.RLock()
			defer db.rwmu.RUnlock()
			break
		}

		defer db.mu.Unlock()
		if r.tx != db.cc {
			return fmt.Errorf("invalid passed transaction context")
		}
	}

	ok := false
	return r.do(r.ctx, names == onlyNames, func(id interface{}, data []interface{}) (more bool, err error) {
		if ok {
			if err = expand(data); err != nil {
				return
			}

			return f(data)
		}

		ok = true
		done := false
		switch names {
		case noNames:
			return true, nil
		case onlyNames:
			done = true
			fallthrough
		default: // returnNames
			flds := data[0].([]*fld)
			a := make([]interface{}, len(flds))
			for i, v := range flds {
				a[i] = v.name
			}
			more, err := f(a)
			return more && !done, err

		}
	})
}

func (db *DB) beginTransaction() { //TODO Rewrite, must use much smaller undo info!
	oldRoot := db.root
	newRoot := &root{}
	*newRoot = *oldRoot
	newRoot.parent = oldRoot
	a := make([]*table, 0, len(oldRoot.tables))
	newRoot.tables = make(map[string]*table, len(oldRoot.tables))
	for k, v := range oldRoot.tables {
		c := v.clone()
		a = append(a, c)
		newRoot.tables[k] = c
	}
	for i := 0; i < len(a)-1; i++ {
		l, p := a[i], a[i+1]
		l.tnext = p
		p.tprev = l
	}
	if len(a) != 0 {
		newRoot.thead = a[0]
	}
	db.root = newRoot
}

func (db *DB) rollback() {
	db.root = db.root.parent
}

func (db *DB) commit() {
	db.root.parent = db.root.parent.parent
}

// Type represents a QL type (bigint, int, string, ...)
type Type int

// Values of ColumnInfo.Type.
const (
	BigInt     Type = qBigInt
	BigRat          = qBigRat
	Blob            = qBlob
	Bool            = qBool
	Complex128      = qComplex128
	Complex64       = qComplex64
	Duration        = qDuration
	Float32         = qFloat32
	Float64         = qFloat64
	Int16           = qInt16
	Int32           = qInt32
	Int64           = qInt64
	Int8            = qInt8
	String          = qString
	Time            = qTime
	Uint16          = qUint16
	Uint32          = qUint32
	Uint64          = qUint64
	Uint8           = qUint8
)

// String implements fmt.Stringer.
func (t Type) String() string {
	return typeStr(int(t))
}

// ColumnInfo provides meta data describing a table column.
type ColumnInfo struct {
	Name string // Column name.
	Type Type   // Column type (BigInt, BigRat, ...).
}

// TableInfo provides meta data describing a DB table.
type TableInfo struct {
	// Table name.
	Name string

	// Table schema. Columns are listed in the order in which they appear
	// in the schema.
	Columns []ColumnInfo
}

// IndexInfo provides meta data describing a DB index.  It corresponds to the
// statement
//
//	CREATE INDEX Name ON Table (Column);
type IndexInfo struct {
	Name   string // Index name
	Table  string // Table name.
	Column string // Column name.
	Unique bool   // Wheter the index is unique.
}

// DbInfo provides meta data describing a DB.
type DbInfo struct {
	Name    string      // DB name.
	Tables  []TableInfo // Tables in the DB.
	Indices []IndexInfo // Indices in the DB.
}

func (db *DB) info() (r *DbInfo, err error) {
	r = &DbInfo{Name: db.Name()}
	for nm, t := range db.root.tables {
		ti := TableInfo{Name: nm}
		for _, c := range t.cols {
			ti.Columns = append(ti.Columns, ColumnInfo{Name: c.name, Type: Type(c.typ)})
		}
		r.Tables = append(r.Tables, ti)
		for i, x := range t.indices {
			if x == nil {
				continue
			}

			var cn string
			switch {
			case i == 0:
				cn = "id()"
			default:
				cn = t.cols0[i-1].name
			}
			r.Indices = append(r.Indices, IndexInfo{x.name, nm, cn, x.unique})
		}
	}
	return
}

// Info provides meta data describing a DB or an error if any. It locks the DB
// to obtain the result.
func (db *DB) Info() (r *DbInfo, err error) {
	db.mu.Lock()
	defer db.mu.Unlock()
	return db.info()
}
