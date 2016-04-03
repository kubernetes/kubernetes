package influxql_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/influxdb/influxdb/influxql"
)

// Ensure a value's data type can be retrieved.
func TestInspectDataType(t *testing.T) {
	for i, tt := range []struct {
		v   interface{}
		typ influxql.DataType
	}{
		{float64(100), influxql.Float},
	} {
		if typ := influxql.InspectDataType(tt.v); tt.typ != typ {
			t.Errorf("%d. %v (%s): unexpected type: %s", i, tt.v, tt.typ, typ)
			continue
		}
	}
}

// Ensure the SELECT statement can extract substatements.
func TestSelectStatement_Substatement(t *testing.T) {
	var tests = []struct {
		stmt string
		expr *influxql.VarRef
		sub  string
		err  string
	}{
		// 0. Single series
		{
			stmt: `SELECT value FROM myseries WHERE value > 1`,
			expr: &influxql.VarRef{Val: "value"},
			sub:  `SELECT value FROM myseries WHERE value > 1.000`,
		},

		// 1. Simple join
		{
			stmt: `SELECT sum(aa.value) + sum(bb.value) FROM aa, bb`,
			expr: &influxql.VarRef{Val: "aa.value"},
			sub:  `SELECT aa.value FROM aa`,
		},

		// 2. Simple merge
		{
			stmt: `SELECT sum(aa.value) + sum(bb.value) FROM aa, bb`,
			expr: &influxql.VarRef{Val: "bb.value"},
			sub:  `SELECT bb.value FROM bb`,
		},

		// 3. Join with condition
		{
			stmt: `SELECT sum(aa.value) + sum(bb.value) FROM aa, bb WHERE aa.host = 'servera' AND bb.host = 'serverb'`,
			expr: &influxql.VarRef{Val: "bb.value"},
			sub:  `SELECT bb.value FROM bb WHERE bb.host = 'serverb'`,
		},

		// 4. Join with complex condition
		{
			stmt: `SELECT sum(aa.value) + sum(bb.value) FROM aa, bb WHERE aa.host = 'servera' AND (bb.host = 'serverb' OR bb.host = 'serverc') AND 1 = 2`,
			expr: &influxql.VarRef{Val: "bb.value"},
			sub:  `SELECT bb.value FROM bb WHERE (bb.host = 'serverb' OR bb.host = 'serverc') AND 1.000 = 2.000`,
		},

		// 5. 4 with different condition order
		{
			stmt: `SELECT sum(aa.value) + sum(bb.value) FROM aa, bb WHERE ((bb.host = 'serverb' OR bb.host = 'serverc') AND aa.host = 'servera') AND 1 = 2`,
			expr: &influxql.VarRef{Val: "bb.value"},
			sub:  `SELECT bb.value FROM bb WHERE ((bb.host = 'serverb' OR bb.host = 'serverc')) AND 1.000 = 2.000`,
		},
	}

	for i, tt := range tests {
		// Parse statement.
		stmt, err := influxql.NewParser(strings.NewReader(tt.stmt)).ParseStatement()
		if err != nil {
			t.Fatalf("invalid statement: %q: %s", tt.stmt, err)
		}

		// Extract substatement.
		sub, err := stmt.(*influxql.SelectStatement).Substatement(tt.expr)
		if err != nil {
			t.Errorf("%d. %q: unexpected error: %s", i, tt.stmt, err)
			continue
		}
		if substr := sub.String(); tt.sub != substr {
			t.Errorf("%d. %q: unexpected substatement:\n\nexp=%s\n\ngot=%s\n\n", i, tt.stmt, tt.sub, substr)
			continue
		}
	}
}

// Ensure the SELECT statement can extract GROUP BY interval.
func TestSelectStatement_GroupByInterval(t *testing.T) {
	q := "SELECT sum(value) from foo  where time < now() GROUP BY time(10m)"
	stmt, err := influxql.NewParser(strings.NewReader(q)).ParseStatement()
	if err != nil {
		t.Fatalf("invalid statement: %q: %s", stmt, err)
	}

	s := stmt.(*influxql.SelectStatement)
	d, err := s.GroupByInterval()
	if d != 10*time.Minute {
		t.Fatalf("group by interval not equal:\nexp=%s\ngot=%s", 10*time.Minute, d)
	}
	if err != nil {
		t.Fatalf("error parsing group by interval: %s", err.Error())
	}
}

// Ensure the SELECT statement can have its start and end time set
func TestSelectStatement_SetTimeRange(t *testing.T) {
	q := "SELECT sum(value) from foo where time < now() GROUP BY time(10m)"
	stmt, err := influxql.NewParser(strings.NewReader(q)).ParseStatement()
	if err != nil {
		t.Fatalf("invalid statement: %q: %s", stmt, err)
	}

	s := stmt.(*influxql.SelectStatement)
	min, max := influxql.TimeRange(s.Condition)
	start := time.Now().Add(-20 * time.Hour).Round(time.Second).UTC()
	end := time.Now().Add(10 * time.Hour).Round(time.Second).UTC()
	s.SetTimeRange(start, end)
	min, max = influxql.TimeRange(s.Condition)

	if min != start {
		t.Fatalf("start time wasn't set properly.\n  exp: %s\n  got: %s", start, min)
	}
	// the end range is actually one nanosecond before the given one since end is exclusive
	end = end.Add(-time.Nanosecond)
	if max != end {
		t.Fatalf("end time wasn't set properly.\n  exp: %s\n  got: %s", end, max)
	}

	// ensure we can set a time on a select that already has one set
	start = time.Now().Add(-20 * time.Hour).Round(time.Second).UTC()
	end = time.Now().Add(10 * time.Hour).Round(time.Second).UTC()
	q = fmt.Sprintf("SELECT sum(value) from foo WHERE time >= %ds and time <= %ds GROUP BY time(10m)", start.Unix(), end.Unix())
	stmt, err = influxql.NewParser(strings.NewReader(q)).ParseStatement()
	if err != nil {
		t.Fatalf("invalid statement: %q: %s", stmt, err)
	}

	s = stmt.(*influxql.SelectStatement)
	min, max = influxql.TimeRange(s.Condition)
	if start != min || end != max {
		t.Fatalf("start and end times weren't equal:\n  exp: %s\n  got: %s\n  exp: %s\n  got:%s\n", start, min, end, max)
	}

	// update and ensure it saves it
	start = time.Now().Add(-40 * time.Hour).Round(time.Second).UTC()
	end = time.Now().Add(20 * time.Hour).Round(time.Second).UTC()
	s.SetTimeRange(start, end)
	min, max = influxql.TimeRange(s.Condition)

	// TODO: right now the SetTimeRange can't override the start time if it's more recent than what they're trying to set it to.
	//       shouldn't matter for our purposes with continuous queries, but fix this later

	if min != start {
		t.Fatalf("start time wasn't set properly.\n  exp: %s\n  got: %s", start, min)
	}
	// the end range is actually one nanosecond before the given one since end is exclusive
	end = end.Add(-time.Nanosecond)
	if max != end {
		t.Fatalf("end time wasn't set properly.\n  exp: %s\n  got: %s", end, max)
	}

	// ensure that when we set a time range other where clause conditions are still there
	q = "SELECT sum(value) from foo WHERE foo = 'bar' and time < now() GROUP BY time(10m)"
	stmt, err = influxql.NewParser(strings.NewReader(q)).ParseStatement()
	if err != nil {
		t.Fatalf("invalid statement: %q: %s", stmt, err)
	}

	s = stmt.(*influxql.SelectStatement)

	// update and ensure it saves it
	start = time.Now().Add(-40 * time.Hour).Round(time.Second).UTC()
	end = time.Now().Add(20 * time.Hour).Round(time.Second).UTC()
	s.SetTimeRange(start, end)
	min, max = influxql.TimeRange(s.Condition)

	if min != start {
		t.Fatalf("start time wasn't set properly.\n  exp: %s\n  got: %s", start, min)
	}
	// the end range is actually one nanosecond before the given one since end is exclusive
	end = end.Add(-time.Nanosecond)
	if max != end {
		t.Fatalf("end time wasn't set properly.\n  exp: %s\n  got: %s", end, max)
	}

	// ensure the where clause is there
	hasWhere := false
	influxql.WalkFunc(s.Condition, func(n influxql.Node) {
		if ex, ok := n.(*influxql.BinaryExpr); ok {
			if lhs, ok := ex.LHS.(*influxql.VarRef); ok {
				if lhs.Val == "foo" {
					if rhs, ok := ex.RHS.(*influxql.StringLiteral); ok {
						if rhs.Val == "bar" {
							hasWhere = true
						}
					}
				}
			}
		}
	})
	if !hasWhere {
		t.Fatal("set time range cleared out the where clause")
	}
}

// Ensure the idents from the select clause can come out
func TestSelect_NamesInSelect(t *testing.T) {
	s := MustParseSelectStatement("select count(asdf), bar from cpu")
	a := s.NamesInSelect()
	if !reflect.DeepEqual(a, []string{"asdf", "bar"}) {
		t.Fatal("expected names asdf and bar")
	}
}

// Ensure the idents from the where clause can come out
func TestSelect_NamesInWhere(t *testing.T) {
	s := MustParseSelectStatement("select * from cpu where time > 23s AND (asdf = 'jkl' OR (foo = 'bar' AND baz = 'bar'))")
	a := s.NamesInWhere()
	if !reflect.DeepEqual(a, []string{"time", "asdf", "foo", "baz"}) {
		t.Fatalf("exp: time,asdf,foo,baz\ngot: %s\n", strings.Join(a, ","))
	}
}

func TestSelectStatement_HasWildcard(t *testing.T) {
	var tests = []struct {
		stmt     string
		wildcard bool
	}{
		// No wildcards
		{
			stmt:     `SELECT value FROM cpu`,
			wildcard: false,
		},

		// Query wildcard
		{
			stmt:     `SELECT * FROM cpu`,
			wildcard: true,
		},

		// No GROUP BY wildcards
		{
			stmt:     `SELECT value FROM cpu GROUP BY host`,
			wildcard: false,
		},

		// No GROUP BY wildcards, time only
		{
			stmt:     `SELECT mean(value) FROM cpu where time < now() GROUP BY time(5ms)`,
			wildcard: false,
		},

		// GROUP BY wildcard
		{
			stmt:     `SELECT value FROM cpu GROUP BY *`,
			wildcard: true,
		},

		// GROUP BY wildcard with time
		{
			stmt:     `SELECT mean(value) FROM cpu where time < now() GROUP BY *,time(1m)`,
			wildcard: true,
		},

		// GROUP BY wildcard with explicit
		{
			stmt:     `SELECT value FROM cpu GROUP BY *,host`,
			wildcard: true,
		},

		// GROUP BY multiple wildcards
		{
			stmt:     `SELECT value FROM cpu GROUP BY *,*`,
			wildcard: true,
		},

		// Combo
		{
			stmt:     `SELECT * FROM cpu GROUP BY *`,
			wildcard: true,
		},
	}

	for i, tt := range tests {
		// Parse statement.
		t.Logf("index: %d, statement: %s", i, tt.stmt)
		stmt, err := influxql.NewParser(strings.NewReader(tt.stmt)).ParseStatement()
		if err != nil {
			t.Fatalf("invalid statement: %q: %s", tt.stmt, err)
		}

		// Test wildcard detection.
		if w := stmt.(*influxql.SelectStatement).HasWildcard(); tt.wildcard != w {
			t.Errorf("%d. %q: unexpected wildcard detection:\n\nexp=%v\n\ngot=%v\n\n", i, tt.stmt, tt.wildcard, w)
			continue
		}
	}
}

// Test SELECT statement wildcard rewrite.
func TestSelectStatement_RewriteWildcards(t *testing.T) {
	var fields = influxql.Fields{
		&influxql.Field{Expr: &influxql.VarRef{Val: "value1"}},
		&influxql.Field{Expr: &influxql.VarRef{Val: "value2"}},
	}
	var dimensions = influxql.Dimensions{
		&influxql.Dimension{Expr: &influxql.VarRef{Val: "host"}},
		&influxql.Dimension{Expr: &influxql.VarRef{Val: "region"}},
	}

	var tests = []struct {
		stmt    string
		rewrite string
	}{
		// No wildcards
		{
			stmt:    `SELECT value FROM cpu`,
			rewrite: `SELECT value FROM cpu`,
		},

		// Query wildcard
		{
			stmt:    `SELECT * FROM cpu`,
			rewrite: `SELECT value1, value2 FROM cpu GROUP BY host, region`,
		},

		// Parser fundamentally prohibits multiple query sources

		// Query wildcard with explicit
		// {
		//	stmt:    `SELECT *,value1 FROM cpu`,
		//		rewrite: `SELECT value1, value2, value1 FROM cpu`,
		//	},

		// Query multiple wildcards
		//	{
		//			stmt:    `SELECT *,* FROM cpu`,
		//			rewrite: `SELECT value1,value2,value1,value2 FROM cpu`,
		//  },

		// No GROUP BY wildcards
		{
			stmt:    `SELECT value FROM cpu GROUP BY host`,
			rewrite: `SELECT value FROM cpu GROUP BY host`,
		},

		// No GROUP BY wildcards, time only
		{
			stmt:    `SELECT mean(value) FROM cpu where time < now() GROUP BY time(5ms)`,
			rewrite: `SELECT mean(value) FROM cpu WHERE time < now() GROUP BY time(5ms)`,
		},

		// GROUP BY wildcard
		{
			stmt:    `SELECT value FROM cpu GROUP BY *`,
			rewrite: `SELECT value FROM cpu GROUP BY host, region`,
		},

		// GROUP BY wildcard with time
		{
			stmt:    `SELECT mean(value) FROM cpu where time < now() GROUP BY *,time(1m)`,
			rewrite: `SELECT mean(value) FROM cpu WHERE time < now() GROUP BY host, region, time(1m)`,
		},

		// GROUP BY wildarde with fill
		{
			stmt:    `SELECT mean(value) FROM cpu where time < now() GROUP BY *,time(1m) fill(0)`,
			rewrite: `SELECT mean(value) FROM cpu WHERE time < now() GROUP BY host, region, time(1m) fill(0)`,
		},

		// GROUP BY wildcard with explicit
		{
			stmt:    `SELECT value FROM cpu GROUP BY *,host`,
			rewrite: `SELECT value FROM cpu GROUP BY host, region, host`,
		},

		// GROUP BY multiple wildcards
		{
			stmt:    `SELECT value FROM cpu GROUP BY *,*`,
			rewrite: `SELECT value FROM cpu GROUP BY host, region, host, region`,
		},

		// Combo
		{
			stmt:    `SELECT * FROM cpu GROUP BY *`,
			rewrite: `SELECT value1, value2 FROM cpu GROUP BY host, region`,
		},
	}

	for i, tt := range tests {
		t.Logf("index: %d, statement: %s", i, tt.stmt)
		// Parse statement.
		stmt, err := influxql.NewParser(strings.NewReader(tt.stmt)).ParseStatement()
		if err != nil {
			t.Fatalf("invalid statement: %q: %s", tt.stmt, err)
		}

		// Rewrite statement.
		rw := stmt.(*influxql.SelectStatement).RewriteWildcards(fields, dimensions)
		if rw == nil {
			t.Errorf("%d. %q: unexpected nil statement", i, tt.stmt)
			continue
		}
		if rw := rw.String(); tt.rewrite != rw {
			t.Errorf("%d. %q: unexpected rewrite:\n\nexp=%s\n\ngot=%s\n\n", i, tt.stmt, tt.rewrite, rw)
			continue
		}
	}
}

// Ensure that the IsRawQuery flag gets set properly
func TestSelectStatement_IsRawQuerySet(t *testing.T) {
	var tests = []struct {
		stmt  string
		isRaw bool
	}{
		{
			stmt:  "select * from foo",
			isRaw: true,
		},
		{
			stmt:  "select value1,value2 from foo",
			isRaw: true,
		},
		{
			stmt:  "select value1,value2 from foo, time(10m)",
			isRaw: true,
		},
		{
			stmt:  "select mean(value) from foo where time < now() group by time(5m)",
			isRaw: false,
		},
		{
			stmt:  "select mean(value) from foo group by bar",
			isRaw: false,
		},
		{
			stmt:  "select mean(value) from foo group by *",
			isRaw: false,
		},
		{
			stmt:  "select mean(*) from foo group by *",
			isRaw: false,
		},
	}

	for i, tt := range tests {
		t.Logf("index: %d, statement: %s", i, tt.stmt)
		s := MustParseSelectStatement(tt.stmt)
		if s.IsRawQuery != tt.isRaw {
			t.Errorf("'%s', IsRawQuery should be %v", tt.stmt, tt.isRaw)
		}
	}
}

// Ensure the time range of an expression can be extracted.
func TestTimeRange(t *testing.T) {
	for i, tt := range []struct {
		expr     string
		min, max string
	}{
		// LHS VarRef
		{expr: `time > '2000-01-01 00:00:00'`, min: `2000-01-01T00:00:00.000000001Z`, max: `0001-01-01T00:00:00Z`},
		{expr: `time >= '2000-01-01 00:00:00'`, min: `2000-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`},
		{expr: `time < '2000-01-01 00:00:00'`, min: `0001-01-01T00:00:00Z`, max: `1999-12-31T23:59:59.999999999Z`},
		{expr: `time <= '2000-01-01 00:00:00'`, min: `0001-01-01T00:00:00Z`, max: `2000-01-01T00:00:00Z`},

		// RHS VarRef
		{expr: `'2000-01-01 00:00:00' > time`, min: `0001-01-01T00:00:00Z`, max: `1999-12-31T23:59:59.999999999Z`},
		{expr: `'2000-01-01 00:00:00' >= time`, min: `0001-01-01T00:00:00Z`, max: `2000-01-01T00:00:00Z`},
		{expr: `'2000-01-01 00:00:00' < time`, min: `2000-01-01T00:00:00.000000001Z`, max: `0001-01-01T00:00:00Z`},
		{expr: `'2000-01-01 00:00:00' <= time`, min: `2000-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`},

		// Equality
		{expr: `time = '2000-01-01 00:00:00'`, min: `2000-01-01T00:00:00Z`, max: `2000-01-01T00:00:00Z`},

		// Multiple time expressions.
		{expr: `time >= '2000-01-01 00:00:00' AND time < '2000-01-02 00:00:00'`, min: `2000-01-01T00:00:00Z`, max: `2000-01-01T23:59:59.999999999Z`},

		// Min/max crossover
		{expr: `time >= '2000-01-01 00:00:00' AND time <= '1999-01-01 00:00:00'`, min: `2000-01-01T00:00:00Z`, max: `1999-01-01T00:00:00Z`},

		// Absolute time
		{expr: `time = 1388534400s`, min: `2014-01-01T00:00:00Z`, max: `2014-01-01T00:00:00Z`},

		// Non-comparative expressions.
		{expr: `time`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`},
		{expr: `time + 2`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`},
		{expr: `time - '2000-01-01 00:00:00'`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`},
		{expr: `time AND '2000-01-01 00:00:00'`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`},
	} {
		// Extract time range.
		expr := MustParseExpr(tt.expr)
		min, max := influxql.TimeRange(expr)

		// Compare with expected min/max.
		if min := min.Format(time.RFC3339Nano); tt.min != min {
			t.Errorf("%d. %s: unexpected min:\n\nexp=%s\n\ngot=%s\n\n", i, tt.expr, tt.min, min)
			continue
		}
		if max := max.Format(time.RFC3339Nano); tt.max != max {
			t.Errorf("%d. %s: unexpected max:\n\nexp=%s\n\ngot=%s\n\n", i, tt.expr, tt.max, max)
			continue
		}
	}
}

// Ensure that we see if a where clause has only time limitations
func TestSelectStatement_OnlyTimeDimensions(t *testing.T) {
	var tests = []struct {
		stmt string
		exp  bool
	}{
		{
			stmt: `SELECT value FROM myseries WHERE value > 1`,
			exp:  false,
		},
		{
			stmt: `SELECT value FROM foo WHERE time >= '2000-01-01T00:00:05Z'`,
			exp:  true,
		},
		{
			stmt: `SELECT value FROM foo WHERE time >= '2000-01-01T00:00:05Z' AND time < '2000-01-01T00:00:05Z'`,
			exp:  true,
		},
		{
			stmt: `SELECT value FROM foo WHERE time >= '2000-01-01T00:00:05Z' AND asdf = 'bar'`,
			exp:  false,
		},
		{
			stmt: `SELECT value FROM foo WHERE asdf = 'jkl' AND (time >= '2000-01-01T00:00:05Z' AND time < '2000-01-01T00:00:05Z')`,
			exp:  false,
		},
	}

	for i, tt := range tests {
		// Parse statement.
		stmt, err := influxql.NewParser(strings.NewReader(tt.stmt)).ParseStatement()
		if err != nil {
			t.Fatalf("invalid statement: %q: %s", tt.stmt, err)
		}
		if stmt.(*influxql.SelectStatement).OnlyTimeDimensions() != tt.exp {
			t.Fatalf("%d. expected statement to return only time dimension to be %t: %s", i, tt.exp, tt.stmt)
		}
	}
}

// Ensure an AST node can be rewritten.
func TestRewrite(t *testing.T) {
	expr := MustParseExpr(`time > 1 OR foo = 2`)

	// Flip LHS & RHS in all binary expressions.
	act := influxql.RewriteFunc(expr, func(n influxql.Node) influxql.Node {
		switch n := n.(type) {
		case *influxql.BinaryExpr:
			return &influxql.BinaryExpr{Op: n.Op, LHS: n.RHS, RHS: n.LHS}
		default:
			return n
		}
	})

	// Verify that everything is flipped.
	if act := act.String(); act != `2.000 = foo OR 1.000 > time` {
		t.Fatalf("unexpected result: %s", act)
	}
}

// Ensure that the String() value of a statement is parseable
func TestParseString(t *testing.T) {
	var tests = []struct {
		stmt string
	}{
		{
			stmt: `SELECT "cpu load" FROM myseries`,
		},
		{
			stmt: `SELECT "cpu load" FROM "my series"`,
		},
		{
			stmt: `SELECT "cpu\"load" FROM myseries`,
		},
		{
			stmt: `SELECT "cpu'load" FROM myseries`,
		},
		{
			stmt: `SELECT "cpu load" FROM "my\"series"`,
		},
		{
			stmt: `SELECT * FROM myseries`,
		},
	}

	for _, tt := range tests {
		// Parse statement.
		stmt, err := influxql.NewParser(strings.NewReader(tt.stmt)).ParseStatement()
		if err != nil {
			t.Fatalf("invalid statement: %q: %s", tt.stmt, err)
		}

		_, err = influxql.NewParser(strings.NewReader(stmt.String())).ParseStatement()
		if err != nil {
			t.Fatalf("failed to parse string: %v\norig: %v\ngot: %v", err, tt.stmt, stmt.String())
		}
	}
}

// Ensure an expression can be reduced.
func TestEval(t *testing.T) {
	for i, tt := range []struct {
		in   string
		out  interface{}
		data map[string]interface{}
	}{
		// Number literals.
		{in: `1 + 2`, out: float64(3)},
		{in: `(foo*2) + ( (4/2) + (3 * 5) - 0.5 )`, out: float64(26.5), data: map[string]interface{}{"foo": float64(5)}},
		{in: `foo / 2`, out: float64(2), data: map[string]interface{}{"foo": float64(4)}},
		{in: `4 = 4`, out: true},
		{in: `4 <> 4`, out: false},
		{in: `6 > 4`, out: true},
		{in: `4 >= 4`, out: true},
		{in: `4 < 6`, out: true},
		{in: `4 <= 4`, out: true},
		{in: `4 AND 5`, out: nil},

		// Boolean literals.
		{in: `true AND false`, out: false},
		{in: `true OR false`, out: true},

		// String literals.
		{in: `'foo' = 'bar'`, out: false},
		{in: `'foo' = 'foo'`, out: true},

		// Variable references.
		{in: `foo`, out: "bar", data: map[string]interface{}{"foo": "bar"}},
		{in: `foo = 'bar'`, out: true, data: map[string]interface{}{"foo": "bar"}},
		{in: `foo = 'bar'`, out: nil, data: map[string]interface{}{"foo": nil}},
		{in: `foo <> 'bar'`, out: true, data: map[string]interface{}{"foo": "xxx"}},
	} {
		// Evaluate expression.
		out := influxql.Eval(MustParseExpr(tt.in), tt.data)

		// Compare with expected output.
		if !reflect.DeepEqual(tt.out, out) {
			t.Errorf("%d. %s: unexpected output:\n\nexp=%#v\n\ngot=%#v\n\n", i, tt.in, tt.out, out)
			continue
		}
	}
}

// Ensure an expression can be reduced.
func TestReduce(t *testing.T) {
	now := mustParseTime("2000-01-01T00:00:00Z")

	for i, tt := range []struct {
		in   string
		out  string
		data Valuer
	}{
		// Number literals.
		{in: `1 + 2`, out: `3.000`},
		{in: `(foo*2) + ( (4/2) + (3 * 5) - 0.5 )`, out: `(foo * 2.000) + 16.500`},
		{in: `foo(bar(2 + 3), 4)`, out: `foo(bar(5.000), 4.000)`},
		{in: `4 / 0`, out: `0.000`},
		{in: `4 = 4`, out: `true`},
		{in: `4 <> 4`, out: `false`},
		{in: `6 > 4`, out: `true`},
		{in: `4 >= 4`, out: `true`},
		{in: `4 < 6`, out: `true`},
		{in: `4 <= 4`, out: `true`},
		{in: `4 AND 5`, out: `4.000 AND 5.000`},

		// Boolean literals.
		{in: `true AND false`, out: `false`},
		{in: `true OR false`, out: `true`},
		{in: `true OR (foo = bar AND 1 > 2)`, out: `true`},
		{in: `(foo = bar AND 1 > 2) OR true`, out: `true`},
		{in: `false OR (foo = bar AND 1 > 2)`, out: `false`},
		{in: `(foo = bar AND 1 > 2) OR false`, out: `false`},
		{in: `true = false`, out: `false`},
		{in: `true <> false`, out: `true`},
		{in: `true + false`, out: `true + false`},

		// Time literals.
		{in: `now() + 2h`, out: `'2000-01-01T02:00:00Z'`, data: map[string]interface{}{"now()": now}},
		{in: `now() / 2h`, out: `'2000-01-01T00:00:00Z' / 2h`, data: map[string]interface{}{"now()": now}},
		{in: `4µ + now()`, out: `'2000-01-01T00:00:00.000004Z'`, data: map[string]interface{}{"now()": now}},
		{in: `now() = now()`, out: `true`, data: map[string]interface{}{"now()": now}},
		{in: `now() <> now()`, out: `false`, data: map[string]interface{}{"now()": now}},
		{in: `now() < now() + 1h`, out: `true`, data: map[string]interface{}{"now()": now}},
		{in: `now() <= now() + 1h`, out: `true`, data: map[string]interface{}{"now()": now}},
		{in: `now() >= now() - 1h`, out: `true`, data: map[string]interface{}{"now()": now}},
		{in: `now() > now() - 1h`, out: `true`, data: map[string]interface{}{"now()": now}},
		{in: `now() - (now() - 60s)`, out: `1m`, data: map[string]interface{}{"now()": now}},
		{in: `now() AND now()`, out: `'2000-01-01T00:00:00Z' AND '2000-01-01T00:00:00Z'`, data: map[string]interface{}{"now()": now}},
		{in: `now()`, out: `now()`},

		// Duration literals.
		{in: `10m + 1h - 60s`, out: `69m`},
		{in: `(10m / 2) * 5`, out: `25m`},
		{in: `60s = 1m`, out: `true`},
		{in: `60s <> 1m`, out: `false`},
		{in: `60s < 1h`, out: `true`},
		{in: `60s <= 1h`, out: `true`},
		{in: `60s > 12s`, out: `true`},
		{in: `60s >= 1m`, out: `true`},
		{in: `60s AND 1m`, out: `1m AND 1m`},
		{in: `60m / 0`, out: `0s`},
		{in: `60m + 50`, out: `1h + 50.000`},

		// String literals.
		{in: `'foo' + 'bar'`, out: `'foobar'`},

		// Variable references.
		{in: `foo`, out: `'bar'`, data: map[string]interface{}{"foo": "bar"}},
		{in: `foo = 'bar'`, out: `true`, data: map[string]interface{}{"foo": "bar"}},
		{in: `foo = 'bar'`, out: `false`, data: map[string]interface{}{"foo": nil}},
		{in: `foo <> 'bar'`, out: `false`, data: map[string]interface{}{"foo": nil}},
	} {
		// Fold expression.
		expr := influxql.Reduce(MustParseExpr(tt.in), tt.data)

		// Compare with expected output.
		if out := expr.String(); tt.out != out {
			t.Errorf("%d. %s: unexpected expr:\n\nexp=%s\n\ngot=%s\n\n", i, tt.in, tt.out, out)
			continue
		}
	}
}

// Valuer represents a simple wrapper around a map to implement the influxql.Valuer interface.
type Valuer map[string]interface{}

// Value returns the value and existence of a key.
func (o Valuer) Value(key string) (v interface{}, ok bool) {
	v, ok = o[key]
	return
}

// mustParseTime parses an IS0-8601 string. Panic on error.
func mustParseTime(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err.Error())
	}
	return t
}
