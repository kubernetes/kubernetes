package influxql_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/influxdata/influxdb/influxql"
)

func BenchmarkQuery_String(b *testing.B) {
	p := influxql.NewParser(strings.NewReader(`SELECT foo AS zoo, a AS b FROM bar WHERE value > 10 AND q = 'hello'`))
	q, _ := p.ParseStatement()
	for i := 0; i < b.N; i++ {
		_ = q.String()
	}
}

// Ensure a value's data type can be retrieved.
func TestInspectDataType(t *testing.T) {
	for i, tt := range []struct {
		v   interface{}
		typ influxql.DataType
	}{
		{float64(100), influxql.Float},
		{int64(100), influxql.Integer},
		{int32(100), influxql.Integer},
		{100, influxql.Integer},
		{true, influxql.Boolean},
		{"string", influxql.String},
		{time.Now(), influxql.Time},
		{time.Second, influxql.Duration},
		{nil, influxql.Unknown},
	} {
		if typ := influxql.InspectDataType(tt.v); tt.typ != typ {
			t.Errorf("%d. %v (%s): unexpected type: %s", i, tt.v, tt.typ, typ)
			continue
		}
	}
}

func TestDataType_String(t *testing.T) {
	for i, tt := range []struct {
		typ influxql.DataType
		v   string
	}{
		{influxql.Float, "float"},
		{influxql.Integer, "integer"},
		{influxql.Boolean, "boolean"},
		{influxql.String, "string"},
		{influxql.Time, "time"},
		{influxql.Duration, "duration"},
		{influxql.Tag, "tag"},
		{influxql.Unknown, "unknown"},
	} {
		if v := tt.typ.String(); tt.v != v {
			t.Errorf("%d. %v (%s): unexpected string: %s", i, tt.typ, tt.v, v)
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
	start := time.Now().Add(-20 * time.Hour).Round(time.Second).UTC()
	end := time.Now().Add(10 * time.Hour).Round(time.Second).UTC()
	s.SetTimeRange(start, end)
	min, max := MustTimeRange(s.Condition)

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
	min, max = MustTimeRange(s.Condition)
	if start != min || end != max {
		t.Fatalf("start and end times weren't equal:\n  exp: %s\n  got: %s\n  exp: %s\n  got:%s\n", start, min, end, max)
	}

	// update and ensure it saves it
	start = time.Now().Add(-40 * time.Hour).Round(time.Second).UTC()
	end = time.Now().Add(20 * time.Hour).Round(time.Second).UTC()
	s.SetTimeRange(start, end)
	min, max = MustTimeRange(s.Condition)

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
	min, max = MustTimeRange(s.Condition)

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
	s := MustParseSelectStatement("select count(asdf), count(bar) from cpu")
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

// Test SELECT statement field rewrite.
func TestSelectStatement_RewriteFields(t *testing.T) {
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
			rewrite: `SELECT host::tag, region::tag, value1::float, value2::integer FROM cpu`,
		},

		// Parser fundamentally prohibits multiple query sources

		// Query wildcard with explicit
		{
			stmt:    `SELECT *,value1 FROM cpu`,
			rewrite: `SELECT host::tag, region::tag, value1::float, value2::integer, value1::float FROM cpu`,
		},

		// Query multiple wildcards
		{
			stmt:    `SELECT *,* FROM cpu`,
			rewrite: `SELECT host::tag, region::tag, value1::float, value2::integer, host::tag, region::tag, value1::float, value2::integer FROM cpu`,
		},

		// Query wildcards with group by
		{
			stmt:    `SELECT * FROM cpu GROUP BY host`,
			rewrite: `SELECT region::tag, value1::float, value2::integer FROM cpu GROUP BY host`,
		},

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

		// GROUP BY wildcard with fill
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
			rewrite: `SELECT value1::float, value2::integer FROM cpu GROUP BY host, region`,
		},

		// Wildcard function with all fields.
		{
			stmt:    `SELECT mean(*) FROM cpu`,
			rewrite: `SELECT mean(value1::float) AS mean_value1, mean(value2::integer) AS mean_value2 FROM cpu`,
		},

		{
			stmt:    `SELECT distinct(*) FROM strings`,
			rewrite: `SELECT distinct(string::string) AS distinct_string, distinct(value::float) AS distinct_value FROM strings`,
		},

		{
			stmt:    `SELECT distinct(*) FROM bools`,
			rewrite: `SELECT distinct(bool::boolean) AS distinct_bool, distinct(value::float) AS distinct_value FROM bools`,
		},

		// Wildcard function with some fields excluded.
		{
			stmt:    `SELECT mean(*) FROM strings`,
			rewrite: `SELECT mean(value::float) AS mean_value FROM strings`,
		},

		{
			stmt:    `SELECT mean(*) FROM bools`,
			rewrite: `SELECT mean(value::float) AS mean_value FROM bools`,
		},

		// Wildcard function with an alias.
		{
			stmt:    `SELECT mean(*) AS alias FROM cpu`,
			rewrite: `SELECT mean(value1::float) AS alias_value1, mean(value2::integer) AS alias_value2 FROM cpu`,
		},

		// Query regex
		{
			stmt:    `SELECT /1/ FROM cpu`,
			rewrite: `SELECT value1::float FROM cpu`,
		},

		{
			stmt:    `SELECT value1 FROM cpu GROUP BY /h/`,
			rewrite: `SELECT value1::float FROM cpu GROUP BY host`,
		},

		// Query regex
		{
			stmt:    `SELECT mean(/1/) FROM cpu`,
			rewrite: `SELECT mean(value1::float) AS mean_value1 FROM cpu`,
		},
	}

	for i, tt := range tests {
		// Parse statement.
		stmt, err := influxql.NewParser(strings.NewReader(tt.stmt)).ParseStatement()
		if err != nil {
			t.Fatalf("invalid statement: %q: %s", tt.stmt, err)
		}

		var ic IteratorCreator
		ic.FieldDimensionsFn = func(sources influxql.Sources) (fields map[string]influxql.DataType, dimensions map[string]struct{}, err error) {
			source := sources[0].(*influxql.Measurement)
			switch source.Name {
			case "cpu":
				fields = map[string]influxql.DataType{
					"value1": influxql.Float,
					"value2": influxql.Integer,
				}
			case "strings":
				fields = map[string]influxql.DataType{
					"value":  influxql.Float,
					"string": influxql.String,
				}
			case "bools":
				fields = map[string]influxql.DataType{
					"value": influxql.Float,
					"bool":  influxql.Boolean,
				}
			}
			dimensions = map[string]struct{}{"host": struct{}{}, "region": struct{}{}}
			return
		}

		// Rewrite statement.
		rw, err := stmt.(*influxql.SelectStatement).RewriteFields(&ic)
		if err != nil {
			t.Errorf("%d. %q: error: %s", i, tt.stmt, err)
		} else if rw == nil {
			t.Errorf("%d. %q: unexpected nil statement", i, tt.stmt)
		} else if rw := rw.String(); tt.rewrite != rw {
			t.Errorf("%d. %q: unexpected rewrite:\n\nexp=%s\n\ngot=%s\n\n", i, tt.stmt, tt.rewrite, rw)
		}
	}
}

// Test SELECT statement regex conditions rewrite.
func TestSelectStatement_RewriteRegexConditions(t *testing.T) {
	var tests = []struct {
		in  string
		out string
	}{
		{in: `SELECT value FROM cpu`, out: `SELECT value FROM cpu`},
		{in: `SELECT value FROM cpu WHERE host='server-1'`, out: `SELECT value FROM cpu WHERE host='server-1'`},
		{in: `SELECT value FROM cpu WHERE host = 'server-1'`, out: `SELECT value FROM cpu WHERE host = 'server-1'`},
		{in: `SELECT value FROM cpu WHERE host != 'server-1'`, out: `SELECT value FROM cpu WHERE host != 'server-1'`},

		// Non matching regex
		{in: `SELECT value FROM cpu WHERE host =~ /server-1|server-2|server-3/`, out: `SELECT value FROM cpu WHERE host =~ /server-1|server-2|server-3/`},
		{in: `SELECT value FROM cpu WHERE host =~ /server-1/`, out: `SELECT value FROM cpu WHERE host =~ /server-1/`},
		{in: `SELECT value FROM cpu WHERE host !~ /server-1/`, out: `SELECT value FROM cpu WHERE host !~ /server-1/`},
		{in: `SELECT value FROM cpu WHERE host =~ /^server-1/`, out: `SELECT value FROM cpu WHERE host =~ /^server-1/`},
		{in: `SELECT value FROM cpu WHERE host =~ /server-1$/`, out: `SELECT value FROM cpu WHERE host =~ /server-1$/`},
		{in: `SELECT value FROM cpu WHERE host !~ /\^server-1$/`, out: `SELECT value FROM cpu WHERE host !~ /\^server-1$/`},
		{in: `SELECT value FROM cpu WHERE host !~ /\^$/`, out: `SELECT value FROM cpu WHERE host !~ /\^$/`},
		{in: `SELECT value FROM cpu WHERE host !~ /^server-1\$/`, out: `SELECT value FROM cpu WHERE host !~ /^server-1\$/`},
		{in: `SELECT value FROM cpu WHERE host =~ /^\$/`, out: `SELECT value FROM cpu WHERE host =~ /^\$/`},
		{in: `SELECT value FROM cpu WHERE host !~ /^a/`, out: `SELECT value FROM cpu WHERE host !~ /^a/`},

		// These regexes are not supported due to the presence of escaped or meta characters.
		{in: `SELECT value FROM cpu WHERE host !~ /^(foo|bar)$/`, out: `SELECT value FROM cpu WHERE host !~ /^(foo|bar)$/`},
		{in: `SELECT value FROM cpu WHERE host !~ /^?a$/`, out: `SELECT value FROM cpu WHERE host !~ /^?a$/`},
		{in: `SELECT value FROM cpu WHERE host !~ /^[a-z]$/`, out: `SELECT value FROM cpu WHERE host !~ /^[a-z]$/`},
		{in: `SELECT value FROM cpu WHERE host !~ /^\d$/`, out: `SELECT value FROM cpu WHERE host !~ /^\d$/`},
		{in: `SELECT value FROM cpu WHERE host !~ /^a*$/`, out: `SELECT value FROM cpu WHERE host !~ /^a*$/`},
		{in: `SELECT value FROM cpu WHERE host !~ /^a.b$/`, out: `SELECT value FROM cpu WHERE host !~ /^a.b$/`},
		{in: `SELECT value FROM cpu WHERE host !~ /^ab+$/`, out: `SELECT value FROM cpu WHERE host !~ /^ab+$/`},
		{in: `SELECT value FROM cpu WHERE host =~ /^hello\world$/`, out: `SELECT value FROM cpu WHERE host =~ /^hello\world$/`},

		// These regexes all match and will be rewritten.
		{in: `SELECT value FROM cpu WHERE host !~ /^a[2]$/`, out: `SELECT value FROM cpu WHERE host != 'a2'`},
		{in: `SELECT value FROM cpu WHERE host =~ /^server-1$/`, out: `SELECT value FROM cpu WHERE host = 'server-1'`},
		{in: `SELECT value FROM cpu WHERE host !~ /^server-1$/`, out: `SELECT value FROM cpu WHERE host != 'server-1'`},
		{in: `SELECT value FROM cpu WHERE host =~ /^server 1$/`, out: `SELECT value FROM cpu WHERE host = 'server 1'`},
		{in: `SELECT value FROM cpu WHERE host =~ /^$/`, out: `SELECT value FROM cpu WHERE host = ''`},
		{in: `SELECT value FROM cpu WHERE host !~ /^$/`, out: `SELECT value FROM cpu WHERE host != ''`},
		{in: `SELECT value FROM cpu WHERE host =~ /^server-1$/ OR host =~ /^server-2$/`, out: `SELECT value FROM cpu WHERE host = 'server-1' OR host = 'server-2'`},
		{in: `SELECT value FROM cpu WHERE host =~ /^server-1$/ OR host =~ /^server]a$/`, out: `SELECT value FROM cpu WHERE host = 'server-1' OR host = 'server]a'`},
		{in: `SELECT value FROM cpu WHERE host =~ /^hello\?$/`, out: `SELECT value FROM cpu WHERE host = 'hello?'`},
		{in: `SELECT value FROM cpu WHERE host !~ /^\\$/`, out: `SELECT value FROM cpu WHERE host != '\\'`},
		{in: `SELECT value FROM cpu WHERE host !~ /^\\\$$/`, out: `SELECT value FROM cpu WHERE host != '\\$'`},
	}

	for i, test := range tests {
		stmt, err := influxql.NewParser(strings.NewReader(test.in)).ParseStatement()
		if err != nil {
			t.Fatalf("[Example %d], %v", i, err)
		}

		// Rewrite any supported regex conditions.
		stmt.(*influxql.SelectStatement).RewriteRegexConditions()

		// Get the expected rewritten statement.
		expStmt, err := influxql.NewParser(strings.NewReader(test.out)).ParseStatement()
		if err != nil {
			t.Fatalf("[Example %d], %v", i, err)
		}

		// Compare the (potentially) rewritten AST to the expected AST.
		if got, exp := stmt, expStmt; !reflect.DeepEqual(got, exp) {
			t.Errorf("[Example %d]\nattempting %v\ngot %v\n%s\n\nexpected %v\n%s\n", i+1, test.in, got, mustMarshalJSON(got), exp, mustMarshalJSON(exp))
		}
	}
}

// Test SELECT statement time field rewrite.
func TestSelectStatement_RewriteTimeFields(t *testing.T) {
	var tests = []struct {
		s    string
		stmt influxql.Statement
	}{
		{
			s: `SELECT time, field1 FROM cpu`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields: []*influxql.Field{
					{Expr: &influxql.VarRef{Val: "field1"}},
				},
				Sources: []influxql.Source{
					&influxql.Measurement{Name: "cpu"},
				},
			},
		},
		{
			s: `SELECT time AS timestamp, field1 FROM cpu`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields: []*influxql.Field{
					{Expr: &influxql.VarRef{Val: "field1"}},
				},
				Sources: []influxql.Source{
					&influxql.Measurement{Name: "cpu"},
				},
				TimeAlias: "timestamp",
			},
		},
	}

	for i, tt := range tests {
		// Parse statement.
		stmt, err := influxql.NewParser(strings.NewReader(tt.s)).ParseStatement()
		if err != nil {
			t.Fatalf("invalid statement: %q: %s", tt.s, err)
		}

		// Rewrite statement.
		stmt.(*influxql.SelectStatement).RewriteTimeFields()
		if !reflect.DeepEqual(tt.stmt, stmt) {
			t.Logf("\n# %s\nexp=%s\ngot=%s\n", tt.s, mustMarshalJSON(tt.stmt), mustMarshalJSON(stmt))
			t.Logf("\nSQL exp=%s\nSQL got=%s\n", tt.stmt.String(), stmt.String())
			t.Errorf("%d. %q\n\nstmt mismatch:\n\nexp=%#v\n\ngot=%#v\n\n", i, tt.s, tt.stmt, stmt)
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
			stmt:  "select mean(value) from foo group by *",
			isRaw: false,
		},
	}

	for _, tt := range tests {
		s := MustParseSelectStatement(tt.stmt)
		if s.IsRawQuery != tt.isRaw {
			t.Errorf("'%s', IsRawQuery should be %v", tt.stmt, tt.isRaw)
		}
	}
}

func TestSelectStatement_HasDerivative(t *testing.T) {
	var tests = []struct {
		stmt       string
		derivative bool
	}{
		// No derivatives
		{
			stmt:       `SELECT value FROM cpu`,
			derivative: false,
		},

		// Query derivative
		{
			stmt:       `SELECT derivative(value) FROM cpu`,
			derivative: true,
		},

		// No GROUP BY time only
		{
			stmt:       `SELECT mean(value) FROM cpu where time < now() GROUP BY time(5ms)`,
			derivative: false,
		},

		// No GROUP BY derivatives, time only
		{
			stmt:       `SELECT derivative(mean(value)) FROM cpu where time < now() GROUP BY time(5ms)`,
			derivative: true,
		},

		{
			stmt:       `SELECT value FROM cpu`,
			derivative: false,
		},

		// Query derivative
		{
			stmt:       `SELECT non_negative_derivative(value) FROM cpu`,
			derivative: true,
		},

		// No GROUP BY derivatives, time only
		{
			stmt:       `SELECT non_negative_derivative(mean(value)) FROM cpu where time < now() GROUP BY time(5ms)`,
			derivative: true,
		},

		// Invalid derivative function name
		{
			stmt:       `SELECT typoDerivative(value) FROM cpu where time < now()`,
			derivative: false,
		},
	}

	for i, tt := range tests {
		// Parse statement.
		t.Logf("index: %d, statement: %s", i, tt.stmt)
		stmt, err := influxql.NewParser(strings.NewReader(tt.stmt)).ParseStatement()
		if err != nil {
			t.Fatalf("invalid statement: %q: %s", tt.stmt, err)
		}

		// Test derivative detection.
		if d := stmt.(*influxql.SelectStatement).HasDerivative(); tt.derivative != d {
			t.Errorf("%d. %q: unexpected derivative detection:\n\nexp=%v\n\ngot=%v\n\n", i, tt.stmt, tt.derivative, d)
			continue
		}
	}
}

func TestSelectStatement_IsSimpleDerivative(t *testing.T) {
	var tests = []struct {
		stmt       string
		derivative bool
	}{
		// No derivatives
		{
			stmt:       `SELECT value FROM cpu`,
			derivative: false,
		},

		// Query derivative
		{
			stmt:       `SELECT derivative(value) FROM cpu`,
			derivative: true,
		},

		// Query derivative
		{
			stmt:       `SELECT non_negative_derivative(value) FROM cpu`,
			derivative: true,
		},

		// No GROUP BY time only
		{
			stmt:       `SELECT mean(value) FROM cpu where time < now() GROUP BY time(5ms)`,
			derivative: false,
		},

		// No GROUP BY derivatives, time only
		{
			stmt:       `SELECT non_negative_derivative(mean(value)) FROM cpu where time < now() GROUP BY time(5ms)`,
			derivative: false,
		},

		// Invalid derivative function name
		{
			stmt:       `SELECT typoDerivative(value) FROM cpu where time < now()`,
			derivative: false,
		},
	}

	for i, tt := range tests {
		// Parse statement.
		t.Logf("index: %d, statement: %s", i, tt.stmt)
		stmt, err := influxql.NewParser(strings.NewReader(tt.stmt)).ParseStatement()
		if err != nil {
			t.Fatalf("invalid statement: %q: %s", tt.stmt, err)
		}

		// Test derivative detection.
		if d := stmt.(*influxql.SelectStatement).IsSimpleDerivative(); tt.derivative != d {
			t.Errorf("%d. %q: unexpected derivative detection:\n\nexp=%v\n\ngot=%v\n\n", i, tt.stmt, tt.derivative, d)
			continue
		}
	}
}

// Ensure binary expression names can be evaluated.
func TestBinaryExprName(t *testing.T) {
	for i, tt := range []struct {
		expr string
		name string
	}{
		{expr: `value + 1`, name: `value`},
		{expr: `"user" / total`, name: `user_total`},
		{expr: `("user" + total) / total`, name: `user_total_total`},
	} {
		expr := influxql.MustParseExpr(tt.expr)
		switch expr := expr.(type) {
		case *influxql.BinaryExpr:
			name := influxql.BinaryExprName(expr)
			if name != tt.name {
				t.Errorf("%d. unexpected name %s, got %s", i, name, tt.name)
			}
		default:
			t.Errorf("%d. unexpected expr type: %T", i, expr)
		}
	}
}

// Ensure the time range of an expression can be extracted.
func TestTimeRange(t *testing.T) {
	for i, tt := range []struct {
		expr          string
		min, max, err string
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

		// number literal
		{expr: `time < 10`, min: `0001-01-01T00:00:00Z`, max: `1970-01-01T00:00:00.000000009Z`},

		// Equality
		{expr: `time = '2000-01-01 00:00:00'`, min: `2000-01-01T00:00:00Z`, max: `2000-01-01T00:00:00.000000001Z`},

		// Multiple time expressions.
		{expr: `time >= '2000-01-01 00:00:00' AND time < '2000-01-02 00:00:00'`, min: `2000-01-01T00:00:00Z`, max: `2000-01-01T23:59:59.999999999Z`},

		// Min/max crossover
		{expr: `time >= '2000-01-01 00:00:00' AND time <= '1999-01-01 00:00:00'`, min: `2000-01-01T00:00:00Z`, max: `1999-01-01T00:00:00Z`},

		// Absolute time
		{expr: `time = 1388534400s`, min: `2014-01-01T00:00:00Z`, max: `2014-01-01T00:00:00.000000001Z`},

		// Non-comparative expressions.
		{expr: `time`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`},
		{expr: `time + 2`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`},
		{expr: `time - '2000-01-01 00:00:00'`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`},
		{expr: `time AND '2000-01-01 00:00:00'`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`},

		// Invalid time expressions.
		{expr: `time > "2000-01-01 00:00:00"`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`, err: `invalid operation: time and *influxql.VarRef are not compatible`},
		{expr: `time > '2262-04-11 23:47:17'`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`, err: `time 2262-04-11T23:47:17Z overflows time literal`},
		{expr: `time > '1677-09-20 19:12:43'`, min: `0001-01-01T00:00:00Z`, max: `0001-01-01T00:00:00Z`, err: `time 1677-09-20T19:12:43Z underflows time literal`},
	} {
		// Extract time range.
		expr := MustParseExpr(tt.expr)
		min, max, err := influxql.TimeRange(expr)

		// Compare with expected min/max.
		if min := min.Format(time.RFC3339Nano); tt.min != min {
			t.Errorf("%d. %s: unexpected min:\n\nexp=%s\n\ngot=%s\n\n", i, tt.expr, tt.min, min)
			continue
		}
		if max := max.Format(time.RFC3339Nano); tt.max != max {
			t.Errorf("%d. %s: unexpected max:\n\nexp=%s\n\ngot=%s\n\n", i, tt.expr, tt.max, max)
			continue
		}
		if (err != nil && err.Error() != tt.err) || (err == nil && tt.err != "") {
			t.Errorf("%d. %s: unexpected error:\n\nexp=%s\n\ngot=%s\n\n", i, tt.expr, tt.err, err)
		}
	}
}

// Ensure that we see if a where clause has only time limitations
func TestOnlyTimeExpr(t *testing.T) {
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
		if influxql.OnlyTimeExpr(stmt.(*influxql.SelectStatement).Condition) != tt.exp {
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
	if act := act.String(); act != `2 = foo OR 1 > time` {
		t.Fatalf("unexpected result: %s", act)
	}
}

// Ensure an Expr can be rewritten handling nils.
func TestRewriteExpr(t *testing.T) {
	expr := MustParseExpr(`(time > 1 AND time < 10) OR foo = 2`)

	// Remove all time expressions.
	act := influxql.RewriteExpr(expr, func(e influxql.Expr) influxql.Expr {
		switch e := e.(type) {
		case *influxql.BinaryExpr:
			if lhs, ok := e.LHS.(*influxql.VarRef); ok && lhs.Val == "time" {
				return nil
			}
		}
		return e
	})

	// Verify that everything is flipped.
	if act := act.String(); act != `foo = 2` {
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
			stmt: `SELECT "field with spaces" FROM "\"ugly\" db"."\"ugly\" rp"."\"ugly\" measurement"`,
		},
		{
			stmt: `SELECT * FROM myseries`,
		},
		{
			stmt: `DROP DATABASE "!"`,
		},
		{
			stmt: `DROP RETENTION POLICY "my rp" ON "a database"`,
		},
		{
			stmt: `CREATE RETENTION POLICY "my rp" ON "a database" DURATION 1d REPLICATION 1`,
		},
		{
			stmt: `ALTER RETENTION POLICY "my rp" ON "a database" DEFAULT`,
		},
		{
			stmt: `SHOW RETENTION POLICIES ON "a database"`,
		},
		{
			stmt: `SHOW TAG VALUES WITH KEY IN ("a long name", short)`,
		},
		{
			stmt: `DROP CONTINUOUS QUERY "my query" ON "my database"`,
		},
		// See issues https://github.com/influxdata/influxdb/issues/1647
		// and https://github.com/influxdata/influxdb/issues/4404
		//{
		//	stmt: `DELETE FROM "my db"."my rp"."my measurement"`,
		//},
		{
			stmt: `DROP SUBSCRIPTION "ugly \"subscription\" name" ON "\"my\" db"."\"my\" rp"`,
		},
		{
			stmt: `CREATE SUBSCRIPTION "ugly \"subscription\" name" ON "\"my\" db"."\"my\" rp" DESTINATIONS ALL 'my host', 'my other host'`,
		},
		{
			stmt: `SHOW MEASUREMENTS WITH MEASUREMENT =~ /foo/`,
		},
		{
			stmt: `SHOW MEASUREMENTS WITH MEASUREMENT = "and/or"`,
		},
		{
			stmt: `DROP USER "user with spaces"`,
		},
		{
			stmt: `GRANT ALL PRIVILEGES ON "db with spaces" TO "user with spaces"`,
		},
		{
			stmt: `GRANT ALL PRIVILEGES TO "user with spaces"`,
		},
		{
			stmt: `SHOW GRANTS FOR "user with spaces"`,
		},
		{
			stmt: `REVOKE ALL PRIVILEGES ON "db with spaces" FROM "user with spaces"`,
		},
		{
			stmt: `REVOKE ALL PRIVILEGES FROM "user with spaces"`,
		},
		{
			stmt: `CREATE DATABASE "db with spaces"`,
		},
	}

	for _, tt := range tests {
		// Parse statement.
		stmt, err := influxql.NewParser(strings.NewReader(tt.stmt)).ParseStatement()
		if err != nil {
			t.Fatalf("invalid statement: %q: %s", tt.stmt, err)
		}

		stmtCopy, err := influxql.NewParser(strings.NewReader(stmt.String())).ParseStatement()
		if err != nil {
			t.Fatalf("failed to parse string: %v\norig: %v\ngot: %v", err, tt.stmt, stmt.String())
		}

		if !reflect.DeepEqual(stmt, stmtCopy) {
			t.Fatalf("statement changed after stringifying and re-parsing:\noriginal : %v\nre-parsed: %v\n", tt.stmt, stmtCopy.String())
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
		{in: `1 + 2`, out: int64(3)},
		{in: `(foo*2) + ( (4/2) + (3 * 5) - 0.5 )`, out: float64(26.5), data: map[string]interface{}{"foo": float64(5)}},
		{in: `foo / 2`, out: float64(2), data: map[string]interface{}{"foo": float64(4)}},
		{in: `4 = 4`, out: true},
		{in: `4 <> 4`, out: false},
		{in: `6 > 4`, out: true},
		{in: `4 >= 4`, out: true},
		{in: `4 < 6`, out: true},
		{in: `4 <= 4`, out: true},
		{in: `4 AND 5`, out: nil},
		{in: `0 = 'test'`, out: false},
		{in: `1.0 = 1`, out: true},
		{in: `1.2 = 1`, out: false},

		// Boolean literals.
		{in: `true AND false`, out: false},
		{in: `true OR false`, out: true},
		{in: `false = 4`, out: false},

		// String literals.
		{in: `'foo' = 'bar'`, out: false},
		{in: `'foo' = 'foo'`, out: true},
		{in: `'' = 4`, out: false},

		// Regex literals.
		{in: `'foo' =~ /f.*/`, out: true},
		{in: `'foo' =~ /b.*/`, out: false},
		{in: `'foo' !~ /f.*/`, out: false},
		{in: `'foo' !~ /b.*/`, out: true},

		// Variable references.
		{in: `foo`, out: "bar", data: map[string]interface{}{"foo": "bar"}},
		{in: `foo = 'bar'`, out: true, data: map[string]interface{}{"foo": "bar"}},
		{in: `foo = 'bar'`, out: nil, data: map[string]interface{}{"foo": nil}},
		{in: `foo <> 'bar'`, out: true, data: map[string]interface{}{"foo": "xxx"}},
		{in: `foo =~ /b.*/`, out: true, data: map[string]interface{}{"foo": "bar"}},
		{in: `foo !~ /b.*/`, out: false, data: map[string]interface{}{"foo": "bar"}},
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
		{in: `1 + 2`, out: `3`},
		{in: `(foo*2) + ( (4/2) + (3 * 5) - 0.5 )`, out: `(foo * 2) + 16.500`},
		{in: `foo(bar(2 + 3), 4)`, out: `foo(bar(5), 4)`},
		{in: `4 / 0`, out: `0.000`},
		{in: `1 / 2`, out: `0.500`},
		{in: `4 = 4`, out: `true`},
		{in: `4 <> 4`, out: `false`},
		{in: `6 > 4`, out: `true`},
		{in: `4 >= 4`, out: `true`},
		{in: `4 < 6`, out: `true`},
		{in: `4 <= 4`, out: `true`},
		{in: `4 AND 5`, out: `4 AND 5`},

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

		// Time literals with now().
		{in: `now() + 2h`, out: `'2000-01-01T02:00:00Z'`, data: map[string]interface{}{"now()": now}},
		{in: `now() / 2h`, out: `'2000-01-01T00:00:00Z' / 2h`, data: map[string]interface{}{"now()": now}},
		{in: `4µ + now()`, out: `'2000-01-01T00:00:00.000004Z'`, data: map[string]interface{}{"now()": now}},
		{in: `now() + 2000000000`, out: `'2000-01-01T00:00:02Z'`, data: map[string]interface{}{"now()": now}},
		{in: `2000000000 + now()`, out: `'2000-01-01T00:00:02Z'`, data: map[string]interface{}{"now()": now}},
		{in: `now() - 2000000000`, out: `'1999-12-31T23:59:58Z'`, data: map[string]interface{}{"now()": now}},
		{in: `now() = now()`, out: `true`, data: map[string]interface{}{"now()": now}},
		{in: `now() <> now()`, out: `false`, data: map[string]interface{}{"now()": now}},
		{in: `now() < now() + 1h`, out: `true`, data: map[string]interface{}{"now()": now}},
		{in: `now() <= now() + 1h`, out: `true`, data: map[string]interface{}{"now()": now}},
		{in: `now() >= now() - 1h`, out: `true`, data: map[string]interface{}{"now()": now}},
		{in: `now() > now() - 1h`, out: `true`, data: map[string]interface{}{"now()": now}},
		{in: `now() - (now() - 60s)`, out: `1m`, data: map[string]interface{}{"now()": now}},
		{in: `now() AND now()`, out: `'2000-01-01T00:00:00Z' AND '2000-01-01T00:00:00Z'`, data: map[string]interface{}{"now()": now}},
		{in: `now()`, out: `now()`},
		{in: `946684800000000000 + 2h`, out: `'2000-01-01T02:00:00Z'`},

		// Time literals.
		{in: `'2000-01-01T00:00:00Z' + 2h`, out: `'2000-01-01T02:00:00Z'`},
		{in: `'2000-01-01T00:00:00Z' / 2h`, out: `'2000-01-01T00:00:00Z' / 2h`},
		{in: `4µ + '2000-01-01T00:00:00Z'`, out: `'2000-01-01T00:00:00.000004Z'`},
		{in: `'2000-01-01T00:00:00Z' + 2000000000`, out: `'2000-01-01T00:00:02Z'`},
		{in: `2000000000 + '2000-01-01T00:00:00Z'`, out: `'2000-01-01T00:00:02Z'`},
		{in: `'2000-01-01T00:00:00Z' - 2000000000`, out: `'1999-12-31T23:59:58Z'`},
		{in: `'2000-01-01T00:00:00Z' = '2000-01-01T00:00:00Z'`, out: `true`},
		{in: `'2000-01-01T00:00:00.000000000Z' = '2000-01-01T00:00:00Z'`, out: `true`},
		{in: `'2000-01-01T00:00:00Z' <> '2000-01-01T00:00:00Z'`, out: `false`},
		{in: `'2000-01-01T00:00:00.000000000Z' <> '2000-01-01T00:00:00Z'`, out: `false`},
		{in: `'2000-01-01T00:00:00Z' < '2000-01-01T00:00:00Z' + 1h`, out: `true`},
		{in: `'2000-01-01T00:00:00.000000000Z' < '2000-01-01T00:00:00Z' + 1h`, out: `true`},
		{in: `'2000-01-01T00:00:00Z' <= '2000-01-01T00:00:00Z' + 1h`, out: `true`},
		{in: `'2000-01-01T00:00:00.000000000Z' <= '2000-01-01T00:00:00Z' + 1h`, out: `true`},
		{in: `'2000-01-01T00:00:00Z' > '2000-01-01T00:00:00Z' - 1h`, out: `true`},
		{in: `'2000-01-01T00:00:00.000000000Z' > '2000-01-01T00:00:00Z' - 1h`, out: `true`},
		{in: `'2000-01-01T00:00:00Z' >= '2000-01-01T00:00:00Z' - 1h`, out: `true`},
		{in: `'2000-01-01T00:00:00.000000000Z' >= '2000-01-01T00:00:00Z' - 1h`, out: `true`},
		{in: `'2000-01-01T00:00:00Z' - ('2000-01-01T00:00:00Z' - 60s)`, out: `1m`},
		{in: `'2000-01-01T00:00:00Z' AND '2000-01-01T00:00:00Z'`, out: `'2000-01-01T00:00:00Z' AND '2000-01-01T00:00:00Z'`},

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
		{in: `60m + 50`, out: `1h + 50`},

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

func Test_fieldsNames(t *testing.T) {
	for _, test := range []struct {
		in    []string
		out   []string
		alias []string
	}{
		{ //case: binary expr(valRef)
			in:    []string{"value+value"},
			out:   []string{"value", "value"},
			alias: []string{"value_value"},
		},
		{ //case: binary expr + valRef
			in:    []string{"value+value", "temperature"},
			out:   []string{"value", "value", "temperature"},
			alias: []string{"value_value", "temperature"},
		},
		{ //case: aggregate expr
			in:    []string{"mean(value)"},
			out:   []string{"mean"},
			alias: []string{"mean"},
		},
		{ //case: binary expr(aggregate expr)
			in:    []string{"mean(value) + max(value)"},
			out:   []string{"value", "value"},
			alias: []string{"mean_max"},
		},
		{ //case: binary expr(aggregate expr) + valRef
			in:    []string{"mean(value) + max(value)", "temperature"},
			out:   []string{"value", "value", "temperature"},
			alias: []string{"mean_max", "temperature"},
		},
		{ //case: mixed aggregate and varRef
			in:    []string{"mean(value) + temperature"},
			out:   []string{"value", "temperature"},
			alias: []string{"mean_temperature"},
		},
		{ //case: ParenExpr(varRef)
			in:    []string{"(value)"},
			out:   []string{"value"},
			alias: []string{"value"},
		},
		{ //case: ParenExpr(varRef + varRef)
			in:    []string{"(value + value)"},
			out:   []string{"value", "value"},
			alias: []string{"value_value"},
		},
		{ //case: ParenExpr(aggregate)
			in:    []string{"(mean(value))"},
			out:   []string{"value"},
			alias: []string{"mean"},
		},
		{ //case: ParenExpr(aggregate + aggregate)
			in:    []string{"(mean(value) + max(value))"},
			out:   []string{"value", "value"},
			alias: []string{"mean_max"},
		},
	} {
		fields := influxql.Fields{}
		for _, s := range test.in {
			expr := MustParseExpr(s)
			fields = append(fields, &influxql.Field{Expr: expr})
		}
		got := fields.Names()
		if !reflect.DeepEqual(got, test.out) {
			t.Errorf("get fields name:\nexp=%v\ngot=%v\n", test.out, got)
		}
		alias := fields.AliasNames()
		if !reflect.DeepEqual(alias, test.alias) {
			t.Errorf("get fields alias name:\nexp=%v\ngot=%v\n", test.alias, alias)
		}
	}

}

func TestSelect_ColumnNames(t *testing.T) {
	for i, tt := range []struct {
		stmt    *influxql.SelectStatement
		columns []string
	}{
		{
			stmt: &influxql.SelectStatement{
				Fields: influxql.Fields([]*influxql.Field{
					{Expr: &influxql.VarRef{Val: "value"}},
				}),
			},
			columns: []string{"time", "value"},
		},
		{
			stmt: &influxql.SelectStatement{
				Fields: influxql.Fields([]*influxql.Field{
					{Expr: &influxql.VarRef{Val: "value"}},
					{Expr: &influxql.VarRef{Val: "value"}},
					{Expr: &influxql.VarRef{Val: "value_1"}},
				}),
			},
			columns: []string{"time", "value", "value_1", "value_1_1"},
		},
		{
			stmt: &influxql.SelectStatement{
				Fields: influxql.Fields([]*influxql.Field{
					{Expr: &influxql.VarRef{Val: "value"}},
					{Expr: &influxql.VarRef{Val: "value_1"}},
					{Expr: &influxql.VarRef{Val: "value"}},
				}),
			},
			columns: []string{"time", "value", "value_1", "value_2"},
		},
		{
			stmt: &influxql.SelectStatement{
				Fields: influxql.Fields([]*influxql.Field{
					{Expr: &influxql.VarRef{Val: "value"}},
					{Expr: &influxql.VarRef{Val: "total"}, Alias: "value"},
					{Expr: &influxql.VarRef{Val: "value"}},
				}),
			},
			columns: []string{"time", "value_1", "value", "value_2"},
		},
		{
			stmt: &influxql.SelectStatement{
				Fields: influxql.Fields([]*influxql.Field{
					{Expr: &influxql.VarRef{Val: "value"}},
				}),
				TimeAlias: "timestamp",
			},
			columns: []string{"timestamp", "value"},
		},
	} {
		columns := tt.stmt.ColumnNames()
		if !reflect.DeepEqual(columns, tt.columns) {
			t.Errorf("%d. expected %s, got %s", i, tt.columns, columns)
		}
	}
}

func TestSelect_Privileges(t *testing.T) {
	stmt := &influxql.SelectStatement{
		Target: &influxql.Target{
			Measurement: &influxql.Measurement{Database: "db2"},
		},
		Sources: []influxql.Source{
			&influxql.Measurement{Database: "db0"},
			&influxql.Measurement{Database: "db1"},
		},
	}

	exp := influxql.ExecutionPrivileges{
		influxql.ExecutionPrivilege{Name: "db0", Privilege: influxql.ReadPrivilege},
		influxql.ExecutionPrivilege{Name: "db1", Privilege: influxql.ReadPrivilege},
		influxql.ExecutionPrivilege{Name: "db2", Privilege: influxql.WritePrivilege},
	}

	got, err := stmt.RequiredPrivileges()
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(exp, got) {
		t.Errorf("exp: %v, got: %v", exp, got)
	}
}

func TestSources_Names(t *testing.T) {
	sources := influxql.Sources([]influxql.Source{
		&influxql.Measurement{
			Name: "cpu",
		},
		&influxql.Measurement{
			Name: "mem",
		},
	})

	names := sources.Names()
	if names[0] != "cpu" {
		t.Errorf("expected cpu, got %s", names[0])
	}
	if names[1] != "mem" {
		t.Errorf("expected mem, got %s", names[1])
	}
}

func TestSources_HasSystemSource(t *testing.T) {
	sources := influxql.Sources([]influxql.Source{
		&influxql.Measurement{
			Name: "_measurements",
		},
	})

	ok := sources.HasSystemSource()
	if !ok {
		t.Errorf("expected to find a system source, found none")
	}

	sources = influxql.Sources([]influxql.Source{
		&influxql.Measurement{
			Name: "cpu",
		},
	})

	ok = sources.HasSystemSource()
	if ok {
		t.Errorf("expected to find no system source, found one")
	}
}

// Parse statements that might appear valid but should return an error.
// If allowed to execute, at least some of these statements would result in a panic.
func TestParse_Errors(t *testing.T) {
	for _, tt := range []struct {
		tmpl string
		good string
		bad  string
	}{
		// Second argument to derivative must be duration
		{tmpl: `SELECT derivative(f, %s) FROM m`, good: "1h", bad: "true"},
	} {
		good := fmt.Sprintf(tt.tmpl, tt.good)
		if _, err := influxql.ParseStatement(good); err != nil {
			t.Fatalf("statement %q should have parsed correctly but returned error: %s", good, err)
		}

		bad := fmt.Sprintf(tt.tmpl, tt.bad)
		if _, err := influxql.ParseStatement(bad); err == nil {
			t.Fatalf("statement %q should have resulted in a parse error but did not", bad)
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

// MustTimeRange will parse a time range. Panic on error.
func MustTimeRange(expr influxql.Expr) (min, max time.Time) {
	min, max, err := influxql.TimeRange(expr)
	if err != nil {
		panic(err)
	}
	return min, max
}

// mustParseTime parses an IS0-8601 string. Panic on error.
func mustParseTime(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err.Error())
	}
	return t
}
