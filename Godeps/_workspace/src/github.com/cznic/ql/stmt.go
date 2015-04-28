// Copyright (c) 2014 ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"bytes"
	"fmt"
	"log"
	"strings"
	"sync"
)

// NOTE: all stmt implementations must be safe for concurrent use by multiple
// goroutines.  If the exec method requires any execution domain local data,
// they must be held out of the implementing instance.
var (
	_ stmt = (*alterTableAddStmt)(nil)
	_ stmt = (*alterTableDropColumnStmt)(nil)
	_ stmt = (*createIndexStmt)(nil)
	_ stmt = (*createTableStmt)(nil)
	_ stmt = (*deleteStmt)(nil) //TODO use indices (need dlist)
	_ stmt = (*dropIndexStmt)(nil)
	_ stmt = (*dropTableStmt)(nil)
	_ stmt = (*insertIntoStmt)(nil)
	_ stmt = (*selectStmt)(nil)
	_ stmt = (*truncateTableStmt)(nil)
	_ stmt = (*updateStmt)(nil) //TODO use indices
	_ stmt = beginTransactionStmt{}
	_ stmt = commitStmt{}
	_ stmt = rollbackStmt{}
)

type stmt interface {
	// never invoked for
	// - beginTransactionStmt
	// - commitStmt
	// - rollbackStmt
	exec(ctx *execCtx) (Recordset, error)

	// return value ignored for
	// - beginTransactionStmt
	// - commitStmt
	// - rollbackStmt
	isUpdating() bool
	String() string
}

type execCtx struct { //LATER +shared temp
	db  *DB
	arg []interface{}
}

type updateStmt struct {
	tableName string
	list      []assignment
	where     expression
}

func (s *updateStmt) String() string {
	u := fmt.Sprintf("UPDATE TABLE %s", s.tableName)
	a := make([]string, len(s.list))
	for i, v := range s.list {
		a[i] = v.String()
	}
	w := ""
	if s.where != nil {
		w = fmt.Sprintf(" WHERE %s", s.where)
	}
	return fmt.Sprintf("%s %s%s", u, strings.Join(a, ", "), w)
}

func (s *updateStmt) exec(ctx *execCtx) (_ Recordset, err error) {
	t, ok := ctx.db.root.tables[s.tableName]
	if !ok {
		return nil, fmt.Errorf("UPDATE: table %s does not exist", s.tableName)
	}

	tcols := make([]*col, len(s.list))
	for i, asgn := range s.list {
		col := findCol(t.cols, asgn.colName)
		if col == nil {
			return nil, fmt.Errorf("UPDATE: unknown column %s", asgn.colName)
		}
		tcols[i] = col
	}

	m := map[interface{}]interface{}{}
	var nh int64
	expr := s.where
	blobCols := t.blobCols()
	cc := ctx.db.cc
	var old []interface{}
	var touched []bool
	if t.hasIndices() {
		old = make([]interface{}, len(t.cols0))
		touched = make([]bool, len(t.cols0))
	}
	for h := t.head; h != 0; h = nh {
		// Read can return lazily expanded chunks
		data, err := t.store.Read(nil, h, t.cols...)
		if err != nil {
			return nil, err
		}

		nh = data[0].(int64)
		for _, col := range t.cols {
			m[col.name] = data[2+col.index]
		}
		m["$id"] = data[1]
		if expr != nil {
			val, err := s.where.eval(ctx, m, ctx.arg)
			if err != nil {
				return nil, err
			}

			if val == nil {
				continue
			}

			x, ok := val.(bool)
			if !ok {
				return nil, fmt.Errorf("invalid WHERE expression %s (value of type %T)", val, val)
			}

			if !x {
				continue
			}
		}

		// hit
		for i, asgn := range s.list {
			val, err := asgn.expr.eval(ctx, m, ctx.arg)
			if err != nil {
				return nil, err
			}

			colIndex := tcols[i].index
			if t.hasIndices() {
				old[colIndex] = data[2+colIndex]
				touched[colIndex] = true
			}
			data[2+colIndex] = val
		}
		if err = typeCheck(data[2:], t.cols); err != nil {
			return nil, err
		}

		for i, v := range t.indices {
			if i == 0 { // id() N/A
				continue
			}

			if v == nil || !touched[i-1] {
				continue
			}

			if err = v.x.Delete(old[i-1], h); err != nil {
				return nil, err
			}
		}

		if err = t.store.UpdateRow(h, blobCols, data...); err != nil { //LATER detect which blobs are actually affected
			return nil, err
		}

		for i, v := range t.indices {
			if i == 0 { // id() N/A
				continue
			}

			if v == nil || !touched[i-1] {
				continue
			}

			if err = v.x.Create(data[2+i-1], h); err != nil {
				return nil, err
			}
		}
		cc.RowsAffected++
	}
	return
}

func (s *updateStmt) isUpdating() bool { return true }

type deleteStmt struct {
	tableName string
	where     expression
}

func (s *deleteStmt) String() string {
	switch {
	case s.where == nil:
		return fmt.Sprintf("DELETE FROM %s;", s.tableName)
	default:
		return fmt.Sprintf("DELETE FROM %s WHERE %s;", s.tableName, s.where)
	}
}

func (s *deleteStmt) exec(ctx *execCtx) (_ Recordset, err error) {
	t, ok := ctx.db.root.tables[s.tableName]
	if !ok {
		return nil, fmt.Errorf("DELETE FROM: table %s does not exist", s.tableName)
	}

	m := map[interface{}]interface{}{}
	var ph, h, nh int64
	var data []interface{}
	blobCols := t.blobCols()
	cc := ctx.db.cc
	for h = t.head; h != 0; ph, h = h, nh {
		for i, v := range data {
			c, ok := v.(chunk)
			if !ok {
				continue
			}

			data[i] = c.b
		}
		// Read can return lazily expanded chunks
		data, err = t.store.Read(nil, h, t.cols...)
		if err != nil {
			return nil, err
		}

		nh = data[0].(int64)
		for _, col := range t.cols {
			m[col.name] = data[2+col.index]
		}
		m["$id"] = data[1]
		val, err := s.where.eval(ctx, m, ctx.arg)
		if err != nil {
			return nil, err
		}

		if val == nil {
			continue
		}

		x, ok := val.(bool)
		if !ok {
			return nil, fmt.Errorf("invalid WHERE expression %s (value of type %T)", val, val)
		}

		if !x {
			continue
		}

		// hit
		for i, v := range t.indices {
			if v == nil {
				continue
			}

			// overflow chunks left in place
			if err = v.x.Delete(data[i+1], h); err != nil {
				return nil, err
			}
		}

		// overflow chunks freed here
		if err = t.store.Delete(h, blobCols...); err != nil {
			return nil, err
		}

		cc.RowsAffected++
		switch {
		case ph == 0 && nh == 0: // "only"
			fallthrough
		case ph == 0 && nh != 0: // "first"
			if err = t.store.Update(t.hhead, nh); err != nil {
				return nil, err
			}

			t.head, h = nh, 0
		case ph != 0 && nh == 0: // "last"
			fallthrough
		case ph != 0 && nh != 0: // "inner"
			pdata, err := t.store.Read(nil, ph, t.cols...)
			if err != nil {
				return nil, err
			}

			for i, v := range pdata {
				if x, ok := v.(chunk); ok {
					pdata[i] = x.b
				}
			}
			pdata[0] = nh
			if err = t.store.Update(ph, pdata...); err != nil {
				return nil, err
			}

			h = ph
		}
	}

	return
}

func (s *deleteStmt) isUpdating() bool { return true }

type truncateTableStmt struct {
	tableName string
}

func (s *truncateTableStmt) String() string { return fmt.Sprintf("TRUNCATE TABLE %s;", s.tableName) }

func (s *truncateTableStmt) exec(ctx *execCtx) (Recordset, error) {
	t, ok := ctx.db.root.tables[s.tableName]
	if !ok {
		return nil, fmt.Errorf("TRUNCATE TABLE: table %s does not exist", s.tableName)
	}

	return nil, t.truncate()
}

func (s *truncateTableStmt) isUpdating() bool { return true }

type dropIndexStmt struct {
	ifExists  bool
	indexName string
}

func (s *dropIndexStmt) String() string { return fmt.Sprintf("DROP INDEX %s;", s.indexName) }

func (s *dropIndexStmt) exec(ctx *execCtx) (Recordset, error) {
	t, x := ctx.db.root.findIndexByName(s.indexName)
	if x == nil {
		if s.ifExists {
			return nil, nil
		}

		return nil, fmt.Errorf("DROP INDEX: index %s does not exist", s.indexName)
	}

	for i, v := range t.indices {
		if v == nil {
			continue
		}

		return nil, t.dropIndex(i)
	}

	panic("internal error 058")
}

func (s *dropIndexStmt) isUpdating() bool { return true }

type dropTableStmt struct {
	ifExists  bool
	tableName string
}

func (s *dropTableStmt) String() string { return fmt.Sprintf("DROP TABLE %s;", s.tableName) }

func (s *dropTableStmt) exec(ctx *execCtx) (Recordset, error) {
	t, ok := ctx.db.root.tables[s.tableName]
	if !ok {
		if s.ifExists {
			return nil, nil
		}

		return nil, fmt.Errorf("DROP TABLE: table %s does not exist", s.tableName)
	}

	return nil, ctx.db.root.dropTable(t)
}

func (s *dropTableStmt) isUpdating() bool { return true }

type alterTableDropColumnStmt struct {
	tableName, colName string
}

func (s *alterTableDropColumnStmt) String() string {
	return fmt.Sprintf("ALTER TABLE %s DROP COLUMN %s;", s.tableName, s.colName)
}

func (s *alterTableDropColumnStmt) exec(ctx *execCtx) (Recordset, error) {
	t, ok := ctx.db.root.tables[s.tableName]
	if !ok {
		return nil, fmt.Errorf("ALTER TABLE: table %s does not exist", s.tableName)
	}

	cols := t.cols
	for _, c := range cols {
		if c.name == s.colName {
			if len(cols) == 1 {
				return nil, fmt.Errorf("ALTER TABLE %s DROP COLUMN: cannot drop the only column: %s", s.tableName, s.colName)
			}

			c.name = ""
			t.cols0[c.index].name = ""
			if t.hasIndices() {
				if v := t.indices[c.index+1]; v != nil {
					if err := t.dropIndex(c.index + 1); err != nil {
						return nil, err
					}
				}
			}
			return nil, t.updated()
		}
	}

	return nil, fmt.Errorf("ALTER TABLE %s DROP COLUMN: column %s does not exist", s.tableName, s.colName)
}

func (s *alterTableDropColumnStmt) isUpdating() bool { return true }

type alterTableAddStmt struct {
	tableName string
	c         *col
}

func (s *alterTableAddStmt) String() string {
	return fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s;", s.tableName, s.c.name)
}

func (s *alterTableAddStmt) exec(ctx *execCtx) (Recordset, error) {
	t, ok := ctx.db.root.tables[s.tableName]
	if !ok {
		return nil, fmt.Errorf("ALTER TABLE: table %s does not exist", s.tableName)
	}

	cols := t.cols
	for _, c := range cols {
		nm := c.name
		if nm == s.c.name {
			return nil, fmt.Errorf("ALTER TABLE %s ADD COLUMN %s: column exists", s.tableName, nm)
		}
	}

	if t.hasIndices() {
		t.indices = append(t.indices, nil)
		t.xroots = append(t.xroots, 0)
		if err := t.store.Update(t.hxroots, t.xroots...); err != nil {
			return nil, err
		}
	}

	t.cols0 = append(t.cols0, s.c)
	return nil, t.updated()
}

func (s *alterTableAddStmt) isUpdating() bool { return true }

type selectStmt struct {
	distinct      bool
	flds          []*fld
	from          *crossJoinRset
	group         *groupByRset
	hasAggregates bool
	limit         *limitRset
	mu            sync.Mutex
	offset        *offsetRset
	order         *orderByRset
	where         *whereRset
}

func (s *selectStmt) String() string {
	var b bytes.Buffer
	b.WriteString("SELECT")
	if s.distinct {
		b.WriteString(" DISTINCT")
	}
	switch {
	case len(s.flds) == 0:
		b.WriteString(" *")
	default:
		a := make([]string, len(s.flds))
		for i, v := range s.flds {
			s := v.expr.String()
			if v.name != "" {
				s += " AS " + v.name
			}
			a[i] = s
		}
		b.WriteString(" " + strings.Join(a, ", "))
	}
	b.WriteString(" FROM ")
	b.WriteString(s.from.String())
	if s.where != nil {
		b.WriteString(" WHERE ")
		b.WriteString(s.where.expr.String())
	}
	if s.group != nil {
		b.WriteString(" GROUP BY ")
		b.WriteString(strings.Join(s.group.colNames, ", "))
	}
	if s.order != nil {
		b.WriteString(" ORDER BY ")
		b.WriteString(s.order.String())
	}
	if s.limit != nil {
		b.WriteString(" LIMIT ")
		b.WriteString(s.limit.expr.String())
	}
	if s.offset != nil {
		b.WriteString(" OFFSET ")
		b.WriteString(s.offset.expr.String())
	}
	b.WriteRune(';')
	return b.String()
}

func (s *selectStmt) do(ctx *execCtx, onlyNames bool, f func(id interface{}, data []interface{}) (more bool, err error)) (err error) {
	return s.exec0().do(ctx, onlyNames, f)
}

func (s *selectStmt) exec0() (r rset) { //LATER overlapping goroutines/pipelines
	s.mu.Lock()
	defer s.mu.Unlock()
	r = rset(s.from)
	if w := s.where; w != nil {
		switch ok, list := isPossiblyRewriteableCrossJoinWhereExpression(w.expr); ok && len(s.from.sources) > 1 {
		case true:
			//dbg("====(in, %d)\n%s\n----", len(list), s)
			tables := s.from.tables()
			if len(list) != len(tables) {
				r = &whereRset{expr: w.expr, src: r}
				break
			}

			m := map[string]int{}
			for i, v := range tables {
				m[v.name] = i
			}
			list2 := make([]int, len(list))
			for i, v := range list {
				itab, ok := m[v.table]
				if !ok {
					break
				}

				delete(m, v.table)
				list2[i] = itab
				if i == len(list)-1 { // last cycle
					if len(m) != 0 { // all tabs "consumed" exactly once
						break
					}

					// Can rewrite
					crs := s.from
					for i, v := range list {
						sel := &selectStmt{
							flds:  []*fld{}, // SELECT *
							from:  &crossJoinRset{sources: []interface{}{[]interface{}{v.table, ""}}},
							where: &whereRset{expr: v.expr},
						}
						info := tables[list2[i]]
						crs.sources[info.i] = []interface{}{sel, info.rename}
					}
					r = rset(crs)
					s.where = nil
					//dbg("====(out)\n%s\n----", s)
				}
			}
			if s.where == nil {
				break
			}

			fallthrough
		default:
			r = &whereRset{expr: w.expr, src: r}
		}
	}
	switch {
	case !s.hasAggregates && s.group == nil: // nop
	case !s.hasAggregates && s.group != nil:
		r = &groupByRset{colNames: s.group.colNames, src: r}
	case s.hasAggregates && s.group == nil:
		r = &groupByRset{src: r}
	case s.hasAggregates && s.group != nil:
		r = &groupByRset{colNames: s.group.colNames, src: r}
	}
	r = &selectRset{flds: s.flds, src: r}
	if s.distinct {
		r = &distinctRset{src: r}
	}
	if s := s.order; s != nil {
		r = &orderByRset{asc: s.asc, by: s.by, src: r}
	}
	if s := s.offset; s != nil {
		r = &offsetRset{s.expr, r}
	}
	if s := s.limit; s != nil {
		r = &limitRset{s.expr, r}
	}
	return
}

func (s *selectStmt) exec(ctx *execCtx) (rs Recordset, err error) {
	return recordset{ctx, s.exec0(), nil}, nil
}

func (s *selectStmt) isUpdating() bool { return false }

type insertIntoStmt struct {
	colNames  []string
	lists     [][]expression
	sel       *selectStmt
	tableName string
}

func (s *insertIntoStmt) String() string {
	cn := ""
	if len(s.colNames) != 0 {
		cn = fmt.Sprintf(" (%s)", strings.Join(s.colNames, ", "))
	}
	switch {
	case s.sel != nil:
		return fmt.Sprintf("INSERT INTO %s%s (%s);", s.tableName, cn, s.sel)
	default:
		a := make([]string, len(s.lists))
		for i, v := range s.lists {
			b := make([]string, len(v))
			for i, v := range v {
				b[i] = v.String()
			}
			a[i] = fmt.Sprintf("(%s)", strings.Join(b, ", "))
		}
		return fmt.Sprintf("INSERT INTO %s%s VALUES %s;", s.tableName, cn, strings.Join(a, ", "))
	}
}

func (s *insertIntoStmt) execSelect(t *table, cols []*col, ctx *execCtx) (_ Recordset, err error) {
	r := s.sel.exec0()
	ok := false
	h := t.head
	data0 := make([]interface{}, len(t.cols0)+2)
	cc := ctx.db.cc
	if err = r.do(ctx, false, func(id interface{}, data []interface{}) (more bool, err error) {
		if ok {
			for i, d := range data {
				data0[cols[i].index+2] = d
			}
			if err = typeCheck(data0[2:], cols); err != nil {
				return
			}

			id, err := t.store.ID()
			if err != nil {
				return false, err
			}

			data0[0] = h
			data0[1] = id

			// Any overflow chunks are written here.
			if h, err = t.store.Create(data0...); err != nil {
				return false, err
			}

			for i, v := range t.indices {
				if v == nil {
					continue
				}

				// Any overflow chunks are shared with the BTree key
				if err = v.x.Create(data0[i+1], h); err != nil {
					return false, err
				}
			}

			cc.RowsAffected++
			ctx.db.root.lastInsertID = id
			return true, nil
		}

		ok = true
		flds := data[0].([]*fld)
		if g, e := len(flds), len(cols); g != e {
			return false, fmt.Errorf("INSERT INTO SELECT: mismatched column counts, have %d, need %d", g, e)
		}

		return true, nil
	}); err != nil {
		return
	}

	if err = t.store.Update(t.hhead, h); err != nil {
		return
	}

	t.head = h
	return
}

func (s *insertIntoStmt) exec(ctx *execCtx) (_ Recordset, err error) {
	t, ok := ctx.db.root.tables[s.tableName]
	if !ok {
		return nil, fmt.Errorf("INSERT INTO %s: table does not exist", s.tableName)
	}

	var cols []*col
	switch len(s.colNames) {
	case 0:
		cols = t.cols
	default:
		for _, colName := range s.colNames {
			if col := findCol(t.cols, colName); col != nil {
				cols = append(cols, col)
				continue
			}

			return nil, fmt.Errorf("INSERT INTO %s: unknown column %s", s.tableName, colName)
		}
	}

	if s.sel != nil {
		return s.execSelect(t, cols, ctx)
	}

	for _, list := range s.lists {
		if g, e := len(list), len(cols); g != e {
			return nil, fmt.Errorf("INSERT INTO %s: expected %d value(s), have %d", s.tableName, e, g)
		}
	}

	arg := ctx.arg
	root := ctx.db.root
	cc := ctx.db.cc
	r := make([]interface{}, len(t.cols0))
	for _, list := range s.lists {
		for i, expr := range list {
			val, err := expr.eval(ctx, nil, arg)
			if err != nil {
				return nil, err
			}

			r[cols[i].index] = val
		}
		if err = typeCheck(r, cols); err != nil {
			return
		}

		id, err := t.addRecord(r)
		if err != nil {
			return nil, err
		}

		cc.RowsAffected++
		root.lastInsertID = id
	}
	return
}

func (s *insertIntoStmt) isUpdating() bool { return true }

type beginTransactionStmt struct{}

func (beginTransactionStmt) String() string { return "BEGIN TRANSACTION;" }
func (beginTransactionStmt) exec(*execCtx) (Recordset, error) {
	log.Panic("internal error 059")
	panic("unreachable")
}
func (beginTransactionStmt) isUpdating() bool {
	log.Panic("internal error 060")
	panic("unreachable")
}

type commitStmt struct{}

func (commitStmt) String() string { return "COMMIT;" }
func (commitStmt) exec(*execCtx) (Recordset, error) {
	log.Panic("internal error 061")
	panic("unreachable")
}
func (commitStmt) isUpdating() bool {
	log.Panic("internal error 062")
	panic("unreachable")
}

type rollbackStmt struct{}

func (rollbackStmt) String() string { return "ROLLBACK;" }
func (rollbackStmt) exec(*execCtx) (Recordset, error) {
	log.Panic("internal error 063")
	panic("unreachable")
}
func (rollbackStmt) isUpdating() bool {
	log.Panic("internal error 064")
	panic("unreachable")
}

type createIndexStmt struct {
	colName     string // alt. "id()" for index on id()
	ifNotExists bool
	indexName   string
	tableName   string
	unique      bool
}

func (s *createIndexStmt) String() string {
	u := ""
	if s.unique {
		u = "UNIQUE "
	}
	e := ""
	if s.ifNotExists {
		e = "IF NOT EXISTS "
	}
	return fmt.Sprintf("CREATE %sINDEX %s%s ON %s (%s);", u, e, s.indexName, s.tableName, s.colName)
}

func (s *createIndexStmt) exec(ctx *execCtx) (Recordset, error) {
	root := ctx.db.root
	if t, i := root.findIndexByName(s.indexName); i != nil {
		if s.ifNotExists {
			return nil, nil
		}

		return nil, fmt.Errorf("CREATE INDEX: table %s already has an index named %s", t.name, i.name)
	}

	if root.tables[s.indexName] != nil {
		return nil, fmt.Errorf("CREATE INDEX: index name collision with existing table: %s", s.indexName)
	}

	t, ok := root.tables[s.tableName]
	if !ok {
		return nil, fmt.Errorf("CREATE INDEX: table does not exist %s", s.tableName)
	}

	if findCol(t.cols, s.indexName) != nil {
		return nil, fmt.Errorf("CREATE INDEX: index name collision with existing column: %s", s.indexName)
	}

	if s.colName == "id()" {
		if err := t.addIndex(s.unique, s.indexName, -1); err != nil {
			return nil, fmt.Errorf("CREATE INDEX: %v", err)
		}

		return nil, t.updated()
	}

	c := findCol(t.cols, s.colName)
	if c == nil {
		return nil, fmt.Errorf("CREATE INDEX: column does not exist: %s", s.colName)
	}

	if err := t.addIndex(s.unique, s.indexName, c.index); err != nil {
		return nil, fmt.Errorf("CREATE INDEX: %v", err)
	}

	return nil, t.updated()
}

func (s *createIndexStmt) isUpdating() bool { return true }

type createTableStmt struct {
	ifNotExists bool
	tableName   string
	cols        []*col
}

func (s *createTableStmt) String() string {
	a := make([]string, len(s.cols))
	for i, v := range s.cols {
		a[i] = fmt.Sprintf("%s %s", v.name, typeStr(v.typ))
	}
	e := ""
	if s.ifNotExists {
		e = "IF NOT EXISTS "
	}
	return fmt.Sprintf("CREATE TABLE %s%s (%s);", e, s.tableName, strings.Join(a, ", "))
}

func (s *createTableStmt) exec(ctx *execCtx) (_ Recordset, err error) {
	root := ctx.db.root
	if _, ok := root.tables[s.tableName]; ok {
		if s.ifNotExists {
			return nil, nil
		}

		return nil, fmt.Errorf("CREATE TABLE: table exists %s", s.tableName)
	}

	if t, x := root.findIndexByName(s.tableName); x != nil {
		return nil, fmt.Errorf("CREATE TABLE: table %s has index %s", t.name, s.tableName)
	}

	m := map[string]bool{}
	for i, c := range s.cols {
		nm := c.name
		if m[nm] {
			return nil, fmt.Errorf("CREATE TABLE: duplicate column %s", nm)
		}

		m[nm] = true
		c.index = i
	}
	_, err = root.createTable(s.tableName, s.cols)
	return
}

func (s *createTableStmt) isUpdating() bool { return true }
