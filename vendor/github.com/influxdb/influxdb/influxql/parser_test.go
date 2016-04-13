package influxql_test

import (
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/influxdb/influxdb/influxql"
)

// Ensure the parser can parse a multi-statement query.
func TestParser_ParseQuery(t *testing.T) {
	s := `SELECT a FROM b; SELECT c FROM d`
	q, err := influxql.NewParser(strings.NewReader(s)).ParseQuery()
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if len(q.Statements) != 2 {
		t.Fatalf("unexpected statement count: %d", len(q.Statements))
	}
}

func TestParser_ParseQuery_TrailingSemicolon(t *testing.T) {
	s := `SELECT value FROM cpu;`
	q, err := influxql.NewParser(strings.NewReader(s)).ParseQuery()
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if len(q.Statements) != 1 {
		t.Fatalf("unexpected statement count: %d", len(q.Statements))
	}
}

// Ensure the parser can parse an empty query.
func TestParser_ParseQuery_Empty(t *testing.T) {
	q, err := influxql.NewParser(strings.NewReader(``)).ParseQuery()
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	} else if len(q.Statements) != 0 {
		t.Fatalf("unexpected statement count: %d", len(q.Statements))
	}
}

// Ensure the parser can return an error from an malformed statement.
func TestParser_ParseQuery_ParseError(t *testing.T) {
	_, err := influxql.NewParser(strings.NewReader(`SELECT`)).ParseQuery()
	if err == nil || err.Error() != `found EOF, expected identifier, string, number, bool at line 1, char 8` {
		t.Fatalf("unexpected error: %s", err)
	}
}

// Ensure the parser can parse strings into Statement ASTs.
func TestParser_ParseStatement(t *testing.T) {
	// For use in various tests.
	now := time.Now()

	var tests = []struct {
		skip bool
		s    string
		stmt influxql.Statement
		err  string
	}{
		// SELECT * statement
		{
			s: `SELECT * FROM myseries`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields: []*influxql.Field{
					{Expr: &influxql.Wildcard{}},
				},
				Sources: []influxql.Source{&influxql.Measurement{Name: "myseries"}},
			},
		},

		// SELECT statement
		{
			skip: true,
			s:    fmt.Sprintf(`SELECT mean(field1), sum(field2) ,count(field3) AS field_x FROM myseries WHERE host = 'hosta.influxdb.org' and time > '%s' GROUP BY time(10h) ORDER BY ASC LIMIT 20 OFFSET 10;`, now.UTC().Format(time.RFC3339Nano)),
			stmt: &influxql.SelectStatement{
				IsRawQuery: false,
				Fields: []*influxql.Field{
					{Expr: &influxql.Call{Name: "mean", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}}}},
					{Expr: &influxql.Call{Name: "sum", Args: []influxql.Expr{&influxql.VarRef{Val: "field2"}}}},
					{Expr: &influxql.Call{Name: "count", Args: []influxql.Expr{&influxql.VarRef{Val: "field3"}}}, Alias: "field_x"},
				},
				Sources: []influxql.Source{&influxql.Measurement{Name: "myseries"}},
				Condition: &influxql.BinaryExpr{
					Op: influxql.AND,
					LHS: &influxql.BinaryExpr{
						Op:  influxql.EQ,
						LHS: &influxql.VarRef{Val: "host"},
						RHS: &influxql.StringLiteral{Val: "hosta.influxdb.org"},
					},
					RHS: &influxql.BinaryExpr{
						Op:  influxql.GT,
						LHS: &influxql.VarRef{Val: "time"},
						RHS: &influxql.TimeLiteral{Val: now.UTC()},
					},
				},
				Dimensions: []*influxql.Dimension{{Expr: &influxql.Call{Name: "time", Args: []influxql.Expr{&influxql.DurationLiteral{Val: 10 * time.Hour}}}}},
				SortFields: []*influxql.SortField{
					{Ascending: true},
				},
				Limit:  20,
				Offset: 10,
			},
		},

		// derivative
		{
			s: `SELECT derivative(field1, 1h) FROM myseries;`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: false,
				Fields: []*influxql.Field{
					{Expr: &influxql.Call{Name: "derivative", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}, &influxql.DurationLiteral{Val: time.Hour}}}},
				},
				Sources: []influxql.Source{&influxql.Measurement{Name: "myseries"}},
			},
		},

		{
			s: `SELECT derivative(mean(field1), 1h) FROM myseries;`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: false,
				Fields: []*influxql.Field{
					{Expr: &influxql.Call{Name: "derivative", Args: []influxql.Expr{&influxql.Call{Name: "mean", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}}}, &influxql.DurationLiteral{Val: time.Hour}}}},
				},
				Sources: []influxql.Source{&influxql.Measurement{Name: "myseries"}},
			},
		},

		{
			s: `SELECT derivative(mean(field1)) FROM myseries;`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: false,
				Fields: []*influxql.Field{
					{Expr: &influxql.Call{Name: "derivative", Args: []influxql.Expr{&influxql.Call{Name: "mean", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}}}}}},
				},
				Sources: []influxql.Source{&influxql.Measurement{Name: "myseries"}},
			},
		},

		// SELECT statement (lowercase)
		{
			s: `select my_field from myseries`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.VarRef{Val: "my_field"}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "myseries"}},
			},
		},

		// SELECT statement (lowercase) with quoted field
		{
			s: `select 'my_field' from myseries`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.StringLiteral{Val: "my_field"}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "myseries"}},
			},
		},

		// SELECT statement with multiple ORDER BY fields
		{
			skip: true,
			s:    `SELECT field1 FROM myseries ORDER BY ASC, field1, field2 DESC LIMIT 10`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.VarRef{Val: "field1"}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "myseries"}},
				SortFields: []*influxql.SortField{
					{Ascending: true},
					{Name: "field1"},
					{Name: "field2"},
				},
				Limit: 10,
			},
		},

		// SELECT statement with SLIMIT and SOFFSET
		{
			s: `SELECT field1 FROM myseries SLIMIT 10 SOFFSET 5`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.VarRef{Val: "field1"}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "myseries"}},
				SLimit:     10,
				SOffset:    5,
			},
		},

		// SELECT * FROM cpu WHERE host = 'serverC' AND region =~ /.*west.*/
		{
			s: `SELECT * FROM cpu WHERE host = 'serverC' AND region =~ /.*west.*/`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op: influxql.AND,
					LHS: &influxql.BinaryExpr{
						Op:  influxql.EQ,
						LHS: &influxql.VarRef{Val: "host"},
						RHS: &influxql.StringLiteral{Val: "serverC"},
					},
					RHS: &influxql.BinaryExpr{
						Op:  influxql.EQREGEX,
						LHS: &influxql.VarRef{Val: "region"},
						RHS: &influxql.RegexLiteral{Val: regexp.MustCompile(".*west.*")},
					},
				},
			},
		},

		// select distinct statements
		{
			s: `select distinct(field1) from cpu`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: false,
				Fields: []*influxql.Field{
					{Expr: &influxql.Call{Name: "distinct", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}}}},
				},
				Sources: []influxql.Source{&influxql.Measurement{Name: "cpu"}},
			},
		},

		{
			s: `select distinct field2 from network`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields: []*influxql.Field{
					{Expr: &influxql.Distinct{Val: "field2"}},
				},
				Sources: []influxql.Source{&influxql.Measurement{Name: "network"}},
			},
		},

		{
			s: `select count(distinct field3) from metrics`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: false,
				Fields: []*influxql.Field{
					{Expr: &influxql.Call{Name: "count", Args: []influxql.Expr{&influxql.Distinct{Val: "field3"}}}},
				},
				Sources: []influxql.Source{&influxql.Measurement{Name: "metrics"}},
			},
		},

		{
			s: `select count(distinct field3), sum(field4) from metrics`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: false,
				Fields: []*influxql.Field{
					{Expr: &influxql.Call{Name: "count", Args: []influxql.Expr{&influxql.Distinct{Val: "field3"}}}},
					{Expr: &influxql.Call{Name: "sum", Args: []influxql.Expr{&influxql.VarRef{Val: "field4"}}}},
				},
				Sources: []influxql.Source{&influxql.Measurement{Name: "metrics"}},
			},
		},

		{
			s: `select count(distinct(field3)), sum(field4) from metrics`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: false,
				Fields: []*influxql.Field{
					{Expr: &influxql.Call{Name: "count", Args: []influxql.Expr{&influxql.Call{Name: "distinct", Args: []influxql.Expr{&influxql.VarRef{Val: "field3"}}}}}},
					{Expr: &influxql.Call{Name: "sum", Args: []influxql.Expr{&influxql.VarRef{Val: "field4"}}}},
				},
				Sources: []influxql.Source{&influxql.Measurement{Name: "metrics"}},
			},
		},

		// SELECT * FROM WHERE time
		{
			s: fmt.Sprintf(`SELECT * FROM cpu WHERE time > '%s'`, now.UTC().Format(time.RFC3339Nano)),
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.GT,
					LHS: &influxql.VarRef{Val: "time"},
					RHS: &influxql.TimeLiteral{Val: now.UTC()},
				},
			},
		},

		// SELECT * FROM WHERE field comparisons
		{
			s: `SELECT * FROM cpu WHERE load > 100`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.GT,
					LHS: &influxql.VarRef{Val: "load"},
					RHS: &influxql.NumberLiteral{Val: 100},
				},
			},
		},
		{
			s: `SELECT * FROM cpu WHERE load >= 100`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.GTE,
					LHS: &influxql.VarRef{Val: "load"},
					RHS: &influxql.NumberLiteral{Val: 100},
				},
			},
		},
		{
			s: `SELECT * FROM cpu WHERE load = 100`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "load"},
					RHS: &influxql.NumberLiteral{Val: 100},
				},
			},
		},
		{
			s: `SELECT * FROM cpu WHERE load <= 100`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.LTE,
					LHS: &influxql.VarRef{Val: "load"},
					RHS: &influxql.NumberLiteral{Val: 100},
				},
			},
		},
		{
			s: `SELECT * FROM cpu WHERE load < 100`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.LT,
					LHS: &influxql.VarRef{Val: "load"},
					RHS: &influxql.NumberLiteral{Val: 100},
				},
			},
		},
		{
			s: `SELECT * FROM cpu WHERE load != 100`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.NEQ,
					LHS: &influxql.VarRef{Val: "load"},
					RHS: &influxql.NumberLiteral{Val: 100},
				},
			},
		},

		// SELECT * FROM /<regex>/
		{
			s: `SELECT * FROM /cpu.*/`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources: []influxql.Source{&influxql.Measurement{
					Regex: &influxql.RegexLiteral{Val: regexp.MustCompile("cpu.*")}},
				},
			},
		},

		// SELECT * FROM "db"."rp"./<regex>/
		{
			s: `SELECT * FROM "db"."rp"./cpu.*/`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources: []influxql.Source{&influxql.Measurement{
					Database:        `db`,
					RetentionPolicy: `rp`,
					Regex:           &influxql.RegexLiteral{Val: regexp.MustCompile("cpu.*")}},
				},
			},
		},

		// SELECT * FROM "db"../<regex>/
		{
			s: `SELECT * FROM "db"../cpu.*/`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources: []influxql.Source{&influxql.Measurement{
					Database: `db`,
					Regex:    &influxql.RegexLiteral{Val: regexp.MustCompile("cpu.*")}},
				},
			},
		},

		// SELECT * FROM "rp"./<regex>/
		{
			s: `SELECT * FROM "rp"./cpu.*/`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: true,
				Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
				Sources: []influxql.Source{&influxql.Measurement{
					RetentionPolicy: `rp`,
					Regex:           &influxql.RegexLiteral{Val: regexp.MustCompile("cpu.*")}},
				},
			},
		},

		// SELECT statement with group by
		{
			s: `SELECT sum(value) FROM "kbps" WHERE time > now() - 120s AND deliveryservice='steam-dns' and cachegroup = 'total' GROUP BY time(60s)`,
			stmt: &influxql.SelectStatement{
				IsRawQuery: false,
				Fields: []*influxql.Field{
					{Expr: &influxql.Call{Name: "sum", Args: []influxql.Expr{&influxql.VarRef{Val: "value"}}}},
				},
				Sources:    []influxql.Source{&influxql.Measurement{Name: "kbps"}},
				Dimensions: []*influxql.Dimension{{Expr: &influxql.Call{Name: "time", Args: []influxql.Expr{&influxql.DurationLiteral{Val: 60 * time.Second}}}}},
				Condition: &influxql.BinaryExpr{ // 1
					Op: influxql.AND,
					LHS: &influxql.BinaryExpr{ // 2
						Op: influxql.AND,
						LHS: &influxql.BinaryExpr{ //3
							Op:  influxql.GT,
							LHS: &influxql.VarRef{Val: "time"},
							RHS: &influxql.BinaryExpr{
								Op:  influxql.SUB,
								LHS: &influxql.Call{Name: "now"},
								RHS: &influxql.DurationLiteral{Val: mustParseDuration("120s")},
							},
						},
						RHS: &influxql.BinaryExpr{
							Op:  influxql.EQ,
							LHS: &influxql.VarRef{Val: "deliveryservice"},
							RHS: &influxql.StringLiteral{Val: "steam-dns"},
						},
					},
					RHS: &influxql.BinaryExpr{
						Op:  influxql.EQ,
						LHS: &influxql.VarRef{Val: "cachegroup"},
						RHS: &influxql.StringLiteral{Val: "total"},
					},
				},
			},
		},

		// SELECT statement with fill
		{
			s: fmt.Sprintf(`SELECT mean(value) FROM cpu where time < '%s' GROUP BY time(5m) fill(1)`, now.UTC().Format(time.RFC3339Nano)),
			stmt: &influxql.SelectStatement{
				Fields: []*influxql.Field{{
					Expr: &influxql.Call{
						Name: "mean",
						Args: []influxql.Expr{&influxql.VarRef{Val: "value"}}}}},
				Sources: []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.LT,
					LHS: &influxql.VarRef{Val: "time"},
					RHS: &influxql.TimeLiteral{Val: now.UTC()},
				},
				Dimensions: []*influxql.Dimension{{Expr: &influxql.Call{Name: "time", Args: []influxql.Expr{&influxql.DurationLiteral{Val: 5 * time.Minute}}}}},
				Fill:       influxql.NumberFill,
				FillValue:  float64(1),
			},
		},

		// SELECT statement with FILL(none) -- check case insensitivity
		{
			s: fmt.Sprintf(`SELECT mean(value) FROM cpu where time < '%s' GROUP BY time(5m) FILL(none)`, now.UTC().Format(time.RFC3339Nano)),
			stmt: &influxql.SelectStatement{
				Fields: []*influxql.Field{{
					Expr: &influxql.Call{
						Name: "mean",
						Args: []influxql.Expr{&influxql.VarRef{Val: "value"}}}}},
				Sources: []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.LT,
					LHS: &influxql.VarRef{Val: "time"},
					RHS: &influxql.TimeLiteral{Val: now.UTC()},
				},
				Dimensions: []*influxql.Dimension{{Expr: &influxql.Call{Name: "time", Args: []influxql.Expr{&influxql.DurationLiteral{Val: 5 * time.Minute}}}}},
				Fill:       influxql.NoFill,
			},
		},

		// SELECT statement with previous fill
		{
			s: fmt.Sprintf(`SELECT mean(value) FROM cpu where time < '%s' GROUP BY time(5m) FILL(previous)`, now.UTC().Format(time.RFC3339Nano)),
			stmt: &influxql.SelectStatement{
				Fields: []*influxql.Field{{
					Expr: &influxql.Call{
						Name: "mean",
						Args: []influxql.Expr{&influxql.VarRef{Val: "value"}}}}},
				Sources: []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.LT,
					LHS: &influxql.VarRef{Val: "time"},
					RHS: &influxql.TimeLiteral{Val: now.UTC()},
				},
				Dimensions: []*influxql.Dimension{{Expr: &influxql.Call{Name: "time", Args: []influxql.Expr{&influxql.DurationLiteral{Val: 5 * time.Minute}}}}},
				Fill:       influxql.PreviousFill,
			},
		},

		// DELETE statement
		{
			s: `DELETE FROM myseries WHERE host = 'hosta.influxdb.org'`,
			stmt: &influxql.DeleteStatement{
				Source: &influxql.Measurement{Name: "myseries"},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "host"},
					RHS: &influxql.StringLiteral{Val: "hosta.influxdb.org"},
				},
			},
		},

		// SHOW SERVERS
		{
			s:    `SHOW SERVERS`,
			stmt: &influxql.ShowServersStatement{},
		},

		// SHOW GRANTS
		{
			s:    `SHOW GRANTS FOR jdoe`,
			stmt: &influxql.ShowGrantsForUserStatement{Name: "jdoe"},
		},

		// SHOW DATABASES
		{
			s:    `SHOW DATABASES`,
			stmt: &influxql.ShowDatabasesStatement{},
		},

		// SHOW SERIES statement
		{
			s:    `SHOW SERIES`,
			stmt: &influxql.ShowSeriesStatement{},
		},

		// SHOW SERIES FROM
		{
			s: `SHOW SERIES FROM cpu`,
			stmt: &influxql.ShowSeriesStatement{
				Sources: []influxql.Source{&influxql.Measurement{Name: "cpu"}},
			},
		},

		// SHOW SERIES FROM /<regex>/
		{
			s: `SHOW SERIES FROM /[cg]pu/`,
			stmt: &influxql.ShowSeriesStatement{
				Sources: []influxql.Source{
					&influxql.Measurement{
						Regex: &influxql.RegexLiteral{Val: regexp.MustCompile(`[cg]pu`)},
					},
				},
			},
		},

		// SHOW SERIES with OFFSET 0
		{
			s:    `SHOW SERIES OFFSET 0`,
			stmt: &influxql.ShowSeriesStatement{Offset: 0},
		},

		// SHOW SERIES with LIMIT 2 OFFSET 0
		{
			s:    `SHOW SERIES LIMIT 2 OFFSET 0`,
			stmt: &influxql.ShowSeriesStatement{Offset: 0, Limit: 2},
		},

		// SHOW SERIES WHERE with ORDER BY and LIMIT
		{
			skip: true,
			s:    `SHOW SERIES WHERE region = 'uswest' ORDER BY ASC, field1, field2 DESC LIMIT 10`,
			stmt: &influxql.ShowSeriesStatement{
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "region"},
					RHS: &influxql.StringLiteral{Val: "uswest"},
				},
				SortFields: []*influxql.SortField{
					{Ascending: true},
					{Name: "field1"},
					{Name: "field2"},
				},
				Limit: 10,
			},
		},

		// SHOW MEASUREMENTS WHERE with ORDER BY and LIMIT
		{
			skip: true,
			s:    `SHOW MEASUREMENTS WHERE region = 'uswest' ORDER BY ASC, field1, field2 DESC LIMIT 10`,
			stmt: &influxql.ShowMeasurementsStatement{
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "region"},
					RHS: &influxql.StringLiteral{Val: "uswest"},
				},
				SortFields: []*influxql.SortField{
					{Ascending: true},
					{Name: "field1"},
					{Name: "field2"},
				},
				Limit: 10,
			},
		},

		// SHOW RETENTION POLICIES
		{
			s: `SHOW RETENTION POLICIES ON mydb`,
			stmt: &influxql.ShowRetentionPoliciesStatement{
				Database: "mydb",
			},
		},

		// SHOW TAG KEYS
		{
			s: `SHOW TAG KEYS FROM src`,
			stmt: &influxql.ShowTagKeysStatement{
				Sources: []influxql.Source{&influxql.Measurement{Name: "src"}},
			},
		},

		// SHOW TAG KEYS FROM /<regex>/
		{
			s: `SHOW TAG KEYS FROM /[cg]pu/`,
			stmt: &influxql.ShowTagKeysStatement{
				Sources: []influxql.Source{
					&influxql.Measurement{
						Regex: &influxql.RegexLiteral{Val: regexp.MustCompile(`[cg]pu`)},
					},
				},
			},
		},

		// SHOW TAG KEYS
		{
			skip: true,
			s:    `SHOW TAG KEYS FROM src WHERE region = 'uswest' ORDER BY ASC, field1, field2 DESC LIMIT 10`,
			stmt: &influxql.ShowTagKeysStatement{
				Sources: []influxql.Source{&influxql.Measurement{Name: "src"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "region"},
					RHS: &influxql.StringLiteral{Val: "uswest"},
				},
				SortFields: []*influxql.SortField{
					{Ascending: true},
					{Name: "field1"},
					{Name: "field2"},
				},
				Limit: 10,
			},
		},

		// SHOW TAG VALUES FROM ... WITH KEY = ...
		{
			skip: true,
			s:    `SHOW TAG VALUES FROM src WITH KEY = region WHERE region = 'uswest' ORDER BY ASC, field1, field2 DESC LIMIT 10`,
			stmt: &influxql.ShowTagValuesStatement{
				Sources: []influxql.Source{&influxql.Measurement{Name: "src"}},
				TagKeys: []string{"region"},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "region"},
					RHS: &influxql.StringLiteral{Val: "uswest"},
				},
				SortFields: []*influxql.SortField{
					{Ascending: true},
					{Name: "field1"},
					{Name: "field2"},
				},
				Limit: 10,
			},
		},

		// SHOW TAG VALUES FROM ... WITH KEY IN...
		{
			s: `SHOW TAG VALUES FROM cpu WITH KEY IN (region, host) WHERE region = 'uswest'`,
			stmt: &influxql.ShowTagValuesStatement{
				Sources: []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				TagKeys: []string{"region", "host"},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "region"},
					RHS: &influxql.StringLiteral{Val: "uswest"},
				},
			},
		},

		// SHOW TAG VALUES ... AND TAG KEY =
		{
			s: `SHOW TAG VALUES FROM cpu WITH KEY IN (region,service,host)WHERE region = 'uswest'`,
			stmt: &influxql.ShowTagValuesStatement{
				Sources: []influxql.Source{&influxql.Measurement{Name: "cpu"}},
				TagKeys: []string{"region", "service", "host"},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "region"},
					RHS: &influxql.StringLiteral{Val: "uswest"},
				},
			},
		},

		// SHOW TAG VALUES WITH KEY = ...
		{
			s: `SHOW TAG VALUES WITH KEY = host WHERE region = 'uswest'`,
			stmt: &influxql.ShowTagValuesStatement{
				TagKeys: []string{"host"},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "region"},
					RHS: &influxql.StringLiteral{Val: "uswest"},
				},
			},
		},

		// SHOW TAG VALUES FROM /<regex>/ WITH KEY = ...
		{
			s: `SHOW TAG VALUES FROM /[cg]pu/ WITH KEY = host`,
			stmt: &influxql.ShowTagValuesStatement{
				Sources: []influxql.Source{
					&influxql.Measurement{
						Regex: &influxql.RegexLiteral{Val: regexp.MustCompile(`[cg]pu`)},
					},
				},
				TagKeys: []string{"host"},
			},
		},

		// SHOW TAG VALUES WITH KEY = "..."
		{
			s: `SHOW TAG VALUES WITH KEY = "host" WHERE region = 'uswest'`,
			stmt: &influxql.ShowTagValuesStatement{
				TagKeys: []string{`host`},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "region"},
					RHS: &influxql.StringLiteral{Val: "uswest"},
				},
			},
		},

		// SHOW USERS
		{
			s:    `SHOW USERS`,
			stmt: &influxql.ShowUsersStatement{},
		},

		// SHOW FIELD KEYS
		{
			skip: true,
			s:    `SHOW FIELD KEYS FROM src ORDER BY ASC, field1, field2 DESC LIMIT 10`,
			stmt: &influxql.ShowFieldKeysStatement{
				Sources: []influxql.Source{&influxql.Measurement{Name: "src"}},
				SortFields: []*influxql.SortField{
					{Ascending: true},
					{Name: "field1"},
					{Name: "field2"},
				},
				Limit: 10,
			},
		},
		{
			s: `SHOW FIELD KEYS FROM /[cg]pu/`,
			stmt: &influxql.ShowFieldKeysStatement{
				Sources: []influxql.Source{
					&influxql.Measurement{
						Regex: &influxql.RegexLiteral{Val: regexp.MustCompile(`[cg]pu`)},
					},
				},
			},
		},

		// DROP SERIES statement
		{
			s:    `DROP SERIES FROM src`,
			stmt: &influxql.DropSeriesStatement{Sources: []influxql.Source{&influxql.Measurement{Name: "src"}}},
		},
		{
			s: `DROP SERIES WHERE host = 'hosta.influxdb.org'`,
			stmt: &influxql.DropSeriesStatement{
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "host"},
					RHS: &influxql.StringLiteral{Val: "hosta.influxdb.org"},
				},
			},
		},
		{
			s: `DROP SERIES FROM src WHERE host = 'hosta.influxdb.org'`,
			stmt: &influxql.DropSeriesStatement{
				Sources: []influxql.Source{&influxql.Measurement{Name: "src"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "host"},
					RHS: &influxql.StringLiteral{Val: "hosta.influxdb.org"},
				},
			},
		},

		// SHOW CONTINUOUS QUERIES statement
		{
			s:    `SHOW CONTINUOUS QUERIES`,
			stmt: &influxql.ShowContinuousQueriesStatement{},
		},

		// CREATE CONTINUOUS QUERY ... INTO <measurement>
		{
			s: `CREATE CONTINUOUS QUERY myquery ON testdb BEGIN SELECT count(field1) INTO measure1 FROM myseries GROUP BY time(5m) END`,
			stmt: &influxql.CreateContinuousQueryStatement{
				Name:     "myquery",
				Database: "testdb",
				Source: &influxql.SelectStatement{
					Fields:  []*influxql.Field{{Expr: &influxql.Call{Name: "count", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}}}}},
					Target:  &influxql.Target{Measurement: &influxql.Measurement{Name: "measure1"}},
					Sources: []influxql.Source{&influxql.Measurement{Name: "myseries"}},
					Dimensions: []*influxql.Dimension{
						{
							Expr: &influxql.Call{
								Name: "time",
								Args: []influxql.Expr{
									&influxql.DurationLiteral{Val: 5 * time.Minute},
								},
							},
						},
					},
				},
			},
		},

		{
			s: `create continuous query "this.is-a.test" on segments begin select * into measure1 from cpu_load_short end`,
			stmt: &influxql.CreateContinuousQueryStatement{
				Name:     "this.is-a.test",
				Database: "segments",
				Source: &influxql.SelectStatement{
					IsRawQuery: true,
					Fields:     []*influxql.Field{{Expr: &influxql.Wildcard{}}},
					Target:     &influxql.Target{Measurement: &influxql.Measurement{Name: "measure1"}},
					Sources:    []influxql.Source{&influxql.Measurement{Name: "cpu_load_short"}},
				},
			},
		},

		// CREATE CONTINUOUS QUERY ... INTO <retention-policy>.<measurement>
		{
			s: `CREATE CONTINUOUS QUERY myquery ON testdb BEGIN SELECT count(field1) INTO "1h.policy1"."cpu.load" FROM myseries GROUP BY time(5m) END`,
			stmt: &influxql.CreateContinuousQueryStatement{
				Name:     "myquery",
				Database: "testdb",
				Source: &influxql.SelectStatement{
					Fields: []*influxql.Field{{Expr: &influxql.Call{Name: "count", Args: []influxql.Expr{&influxql.VarRef{Val: "field1"}}}}},
					Target: &influxql.Target{
						Measurement: &influxql.Measurement{RetentionPolicy: "1h.policy1", Name: "cpu.load"},
					},
					Sources: []influxql.Source{&influxql.Measurement{Name: "myseries"}},
					Dimensions: []*influxql.Dimension{
						{
							Expr: &influxql.Call{
								Name: "time",
								Args: []influxql.Expr{
									&influxql.DurationLiteral{Val: 5 * time.Minute},
								},
							},
						},
					},
				},
			},
		},

		// CREATE CONTINUOUS QUERY for non-aggregate SELECT stmts
		{
			s: `CREATE CONTINUOUS QUERY myquery ON testdb BEGIN SELECT value INTO "policy1"."value" FROM myseries END`,
			stmt: &influxql.CreateContinuousQueryStatement{
				Name:     "myquery",
				Database: "testdb",
				Source: &influxql.SelectStatement{
					IsRawQuery: true,
					Fields:     []*influxql.Field{{Expr: &influxql.VarRef{Val: "value"}}},
					Target: &influxql.Target{
						Measurement: &influxql.Measurement{RetentionPolicy: "policy1", Name: "value"},
					},
					Sources: []influxql.Source{&influxql.Measurement{Name: "myseries"}},
				},
			},
		},

		// CREATE CONTINUOUS QUERY for non-aggregate SELECT stmts with multiple values
		{
			s: `CREATE CONTINUOUS QUERY myquery ON testdb BEGIN SELECT transmit_rx, transmit_tx INTO "policy1"."network" FROM myseries END`,
			stmt: &influxql.CreateContinuousQueryStatement{
				Name:     "myquery",
				Database: "testdb",
				Source: &influxql.SelectStatement{
					IsRawQuery: true,
					Fields: []*influxql.Field{{Expr: &influxql.VarRef{Val: "transmit_rx"}},
						{Expr: &influxql.VarRef{Val: "transmit_tx"}}},
					Target: &influxql.Target{
						Measurement: &influxql.Measurement{RetentionPolicy: "policy1", Name: "network"},
					},
					Sources: []influxql.Source{&influxql.Measurement{Name: "myseries"}},
				},
			},
		},

		// CREATE DATABASE statement
		{
			s: `CREATE DATABASE testdb`,
			stmt: &influxql.CreateDatabaseStatement{
				Name: "testdb",
			},
		},

		// CREATE USER statement
		{
			s: `CREATE USER testuser WITH PASSWORD 'pwd1337'`,
			stmt: &influxql.CreateUserStatement{
				Name:     "testuser",
				Password: "pwd1337",
			},
		},

		// CREATE USER ... WITH ALL PRIVILEGES
		{
			s: `CREATE USER testuser WITH PASSWORD 'pwd1337' WITH ALL PRIVILEGES`,
			stmt: &influxql.CreateUserStatement{
				Name:     "testuser",
				Password: "pwd1337",
				Admin:    true,
			},
		},

		// SET PASSWORD FOR USER
		{
			s: `SET PASSWORD FOR testuser = 'pwd1337'`,
			stmt: &influxql.SetPasswordUserStatement{
				Name:     "testuser",
				Password: "pwd1337",
			},
		},

		// DROP CONTINUOUS QUERY statement
		{
			s:    `DROP CONTINUOUS QUERY myquery ON foo`,
			stmt: &influxql.DropContinuousQueryStatement{Name: "myquery", Database: "foo"},
		},

		// DROP DATABASE statement
		{
			s:    `DROP DATABASE testdb`,
			stmt: &influxql.DropDatabaseStatement{Name: "testdb"},
		},

		// DROP MEASUREMENT statement
		{
			s:    `DROP MEASUREMENT cpu`,
			stmt: &influxql.DropMeasurementStatement{Name: "cpu"},
		},

		// DROP RETENTION POLICY
		{
			s: `DROP RETENTION POLICY "1h.cpu" ON mydb`,
			stmt: &influxql.DropRetentionPolicyStatement{
				Name:     `1h.cpu`,
				Database: `mydb`,
			},
		},

		// DROP USER statement
		{
			s:    `DROP USER jdoe`,
			stmt: &influxql.DropUserStatement{Name: "jdoe"},
		},

		// GRANT READ
		{
			s: `GRANT READ ON testdb TO jdoe`,
			stmt: &influxql.GrantStatement{
				Privilege: influxql.ReadPrivilege,
				On:        "testdb",
				User:      "jdoe",
			},
		},

		// GRANT WRITE
		{
			s: `GRANT WRITE ON testdb TO jdoe`,
			stmt: &influxql.GrantStatement{
				Privilege: influxql.WritePrivilege,
				On:        "testdb",
				User:      "jdoe",
			},
		},

		// GRANT ALL
		{
			s: `GRANT ALL ON testdb TO jdoe`,
			stmt: &influxql.GrantStatement{
				Privilege: influxql.AllPrivileges,
				On:        "testdb",
				User:      "jdoe",
			},
		},

		// GRANT ALL PRIVILEGES
		{
			s: `GRANT ALL PRIVILEGES ON testdb TO jdoe`,
			stmt: &influxql.GrantStatement{
				Privilege: influxql.AllPrivileges,
				On:        "testdb",
				User:      "jdoe",
			},
		},

		// GRANT ALL admin privilege
		{
			s: `GRANT ALL TO jdoe`,
			stmt: &influxql.GrantAdminStatement{
				User: "jdoe",
			},
		},

		// GRANT ALL PRVILEGES admin privilege
		{
			s: `GRANT ALL PRIVILEGES TO jdoe`,
			stmt: &influxql.GrantAdminStatement{
				User: "jdoe",
			},
		},

		// REVOKE READ
		{
			s: `REVOKE READ on testdb FROM jdoe`,
			stmt: &influxql.RevokeStatement{
				Privilege: influxql.ReadPrivilege,
				On:        "testdb",
				User:      "jdoe",
			},
		},

		// REVOKE WRITE
		{
			s: `REVOKE WRITE ON testdb FROM jdoe`,
			stmt: &influxql.RevokeStatement{
				Privilege: influxql.WritePrivilege,
				On:        "testdb",
				User:      "jdoe",
			},
		},

		// REVOKE ALL
		{
			s: `REVOKE ALL ON testdb FROM jdoe`,
			stmt: &influxql.RevokeStatement{
				Privilege: influxql.AllPrivileges,
				On:        "testdb",
				User:      "jdoe",
			},
		},

		// REVOKE ALL PRIVILEGES
		{
			s: `REVOKE ALL PRIVILEGES ON testdb FROM jdoe`,
			stmt: &influxql.RevokeStatement{
				Privilege: influxql.AllPrivileges,
				On:        "testdb",
				User:      "jdoe",
			},
		},

		// REVOKE ALL admin privilege
		{
			s: `REVOKE ALL FROM jdoe`,
			stmt: &influxql.RevokeAdminStatement{
				User: "jdoe",
			},
		},

		// REVOKE ALL PRIVILEGES admin privilege
		{
			s: `REVOKE ALL PRIVILEGES FROM jdoe`,
			stmt: &influxql.RevokeAdminStatement{
				User: "jdoe",
			},
		},

		// CREATE RETENTION POLICY
		{
			s: `CREATE RETENTION POLICY policy1 ON testdb DURATION 1h REPLICATION 2`,
			stmt: &influxql.CreateRetentionPolicyStatement{
				Name:        "policy1",
				Database:    "testdb",
				Duration:    time.Hour,
				Replication: 2,
			},
		},

		// CREATE RETENTION POLICY with infinite retention
		{
			s: `CREATE RETENTION POLICY policy1 ON testdb DURATION INF REPLICATION 2`,
			stmt: &influxql.CreateRetentionPolicyStatement{
				Name:        "policy1",
				Database:    "testdb",
				Duration:    0,
				Replication: 2,
			},
		},

		// CREATE RETENTION POLICY ... DEFAULT
		{
			s: `CREATE RETENTION POLICY policy1 ON testdb DURATION 2m REPLICATION 4 DEFAULT`,
			stmt: &influxql.CreateRetentionPolicyStatement{
				Name:        "policy1",
				Database:    "testdb",
				Duration:    2 * time.Minute,
				Replication: 4,
				Default:     true,
			},
		},

		// ALTER RETENTION POLICY
		{
			s:    `ALTER RETENTION POLICY policy1 ON testdb DURATION 1m REPLICATION 4 DEFAULT`,
			stmt: newAlterRetentionPolicyStatement("policy1", "testdb", time.Minute, 4, true),
		},

		// ALTER RETENTION POLICY with options in reverse order
		{
			s:    `ALTER RETENTION POLICY policy1 ON testdb DEFAULT REPLICATION 4 DURATION 1m`,
			stmt: newAlterRetentionPolicyStatement("policy1", "testdb", time.Minute, 4, true),
		},

		// ALTER RETENTION POLICY with infinite retention
		{
			s:    `ALTER RETENTION POLICY policy1 ON testdb DEFAULT REPLICATION 4 DURATION INF`,
			stmt: newAlterRetentionPolicyStatement("policy1", "testdb", 0, 4, true),
		},

		// ALTER RETENTION POLICY without optional DURATION
		{
			s:    `ALTER RETENTION POLICY policy1 ON testdb DEFAULT REPLICATION 4`,
			stmt: newAlterRetentionPolicyStatement("policy1", "testdb", -1, 4, true),
		},

		// ALTER RETENTION POLICY without optional REPLICATION
		{
			s:    `ALTER RETENTION POLICY policy1 ON testdb DEFAULT`,
			stmt: newAlterRetentionPolicyStatement("policy1", "testdb", -1, -1, true),
		},

		// ALTER RETENTION POLICY without optional DEFAULT
		{
			s:    `ALTER RETENTION POLICY policy1 ON testdb REPLICATION 4`,
			stmt: newAlterRetentionPolicyStatement("policy1", "testdb", -1, 4, false),
		},
		// ALTER default retention policy unquoted
		{
			s:    `ALTER RETENTION POLICY default ON testdb REPLICATION 4`,
			stmt: newAlterRetentionPolicyStatement("default", "testdb", -1, 4, false),
		},

		// SHOW STATS
		{
			s: `SHOW STATS`,
			stmt: &influxql.ShowStatsStatement{
				Host: "",
			},
		},
		{
			s: `SHOW STATS ON 'servera'`,
			stmt: &influxql.ShowStatsStatement{
				Host: "servera",
			},
		},
		{
			s: `SHOW STATS ON '192.167.1.44'`,
			stmt: &influxql.ShowStatsStatement{
				Host: "192.167.1.44",
			},
		},

		// SHOW DIAGNOSTICS
		{
			s:    `SHOW DIAGNOSTICS`,
			stmt: &influxql.ShowDiagnosticsStatement{},
		},

		// Errors
		{s: ``, err: `found EOF, expected SELECT, DELETE, SHOW, CREATE, DROP, GRANT, REVOKE, ALTER, SET at line 1, char 1`},
		{s: `SELECT`, err: `found EOF, expected identifier, string, number, bool at line 1, char 8`},
		{s: `blah blah`, err: `found blah, expected SELECT, DELETE, SHOW, CREATE, DROP, GRANT, REVOKE, ALTER, SET at line 1, char 1`},
		{s: `SELECT field1 X`, err: `found X, expected FROM at line 1, char 15`},
		{s: `SELECT field1 FROM "series" WHERE X +;`, err: `found ;, expected identifier, string, number, bool at line 1, char 38`},
		{s: `SELECT field1 FROM myseries GROUP`, err: `found EOF, expected BY at line 1, char 35`},
		{s: `SELECT field1 FROM myseries LIMIT`, err: `found EOF, expected number at line 1, char 35`},
		{s: `SELECT field1 FROM myseries LIMIT 10.5`, err: `fractional parts not allowed in LIMIT at line 1, char 35`},
		{s: `SELECT field1 FROM myseries OFFSET`, err: `found EOF, expected number at line 1, char 36`},
		{s: `SELECT field1 FROM myseries OFFSET 10.5`, err: `fractional parts not allowed in OFFSET at line 1, char 36`},
		{s: `SELECT field1 FROM myseries ORDER`, err: `found EOF, expected BY at line 1, char 35`},
		{s: `SELECT field1 FROM myseries ORDER BY`, err: `found EOF, expected identifier, ASC, DESC at line 1, char 38`},
		{s: `SELECT field1 FROM myseries ORDER BY /`, err: `found /, expected identifier, ASC, DESC at line 1, char 38`},
		{s: `SELECT field1 FROM myseries ORDER BY 1`, err: `found 1, expected identifier, ASC, DESC at line 1, char 38`},
		{s: `SELECT field1 FROM myseries ORDER BY time ASC,`, err: `found EOF, expected identifier at line 1, char 47`},
		{s: `SELECT field1 FROM myseries ORDER BY DESC`, err: `only ORDER BY time ASC supported at this time`},
		{s: `SELECT field1 FROM myseries ORDER BY field1`, err: `only ORDER BY time ASC supported at this time`},
		{s: `SELECT field1 FROM myseries ORDER BY time DESC`, err: `only ORDER BY time ASC supported at this time`},
		{s: `SELECT field1 FROM myseries ORDER BY time, field1`, err: `only ORDER BY time ASC supported at this time`},
		{s: `SELECT field1 AS`, err: `found EOF, expected identifier at line 1, char 18`},
		{s: `SELECT field1 FROM foo group by time(1s)`, err: `GROUP BY requires at least one aggregate function`},
		{s: `SELECT count(value) FROM foo group by time(1s)`, err: `aggregate functions with GROUP BY time require a WHERE time clause`},
		{s: `SELECT count(value) FROM foo group by time(1s) where host = 'hosta.influxdb.org'`, err: `aggregate functions with GROUP BY time require a WHERE time clause`},
		{s: `SELECT field1 FROM 12`, err: `found 12, expected identifier at line 1, char 20`},
		{s: `SELECT 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 FROM myseries`, err: `unable to parse number at line 1, char 8`},
		{s: `SELECT 10.5h FROM myseries`, err: `found h, expected FROM at line 1, char 12`},
		{s: `SELECT derivative(field1), field1 FROM myseries`, err: `derivative cannot be used with other fields`},
		{s: `SELECT distinct(field1), sum(field1) FROM myseries`, err: `aggregate function distinct() can not be combined with other functions or fields`},
		{s: `SELECT distinct(field1), field2 FROM myseries`, err: `aggregate function distinct() can not be combined with other functions or fields`},
		{s: `SELECT distinct(field1, field2) FROM myseries`, err: `distinct function can only have one argument`},
		{s: `SELECT distinct() FROM myseries`, err: `distinct function requires at least one argument`},
		{s: `SELECT distinct FROM myseries`, err: `found FROM, expected identifier at line 1, char 17`},
		{s: `SELECT distinct field1, field2 FROM myseries`, err: `aggregate function distinct() can not be combined with other functions or fields`},
		{s: `SELECT count(distinct) FROM myseries`, err: `found ), expected (, identifier at line 1, char 22`},
		{s: `SELECT count(distinct field1, field2) FROM myseries`, err: `count(distinct <field>) can only have one argument`},
		{s: `select count(distinct(too, many, arguments)) from myseries`, err: `count(distinct <field>) can only have one argument`},
		{s: `select count() from myseries`, err: `invalid number of arguments for count, expected 1, got 0`},
		{s: `select derivative() from myseries`, err: `invalid number of arguments for derivative, expected at least 1 but no more than 2, got 0`},
		{s: `select derivative(mean(value), 1h, 3) from myseries`, err: `invalid number of arguments for derivative, expected at least 1 but no more than 2, got 3`},
		{s: `SELECT field1 from myseries WHERE host =~ 'asd' LIMIT 1`, err: `found asd, expected regex at line 1, char 42`},
		{s: `DELETE`, err: `found EOF, expected FROM at line 1, char 8`},
		{s: `DELETE FROM`, err: `found EOF, expected identifier at line 1, char 13`},
		{s: `DELETE FROM myseries WHERE`, err: `found EOF, expected identifier, string, number, bool at line 1, char 28`},
		{s: `DROP MEASUREMENT`, err: `found EOF, expected identifier at line 1, char 18`},
		{s: `DROP SERIES`, err: `found EOF, expected FROM, WHERE at line 1, char 13`},
		{s: `DROP SERIES FROM`, err: `found EOF, expected identifier at line 1, char 18`},
		{s: `DROP SERIES FROM src WHERE`, err: `found EOF, expected identifier, string, number, bool at line 1, char 28`},
		{s: `SHOW CONTINUOUS`, err: `found EOF, expected QUERIES at line 1, char 17`},
		{s: `SHOW RETENTION`, err: `found EOF, expected POLICIES at line 1, char 16`},
		{s: `SHOW RETENTION ON`, err: `found ON, expected POLICIES at line 1, char 16`},
		{s: `SHOW RETENTION POLICIES`, err: `found EOF, expected ON at line 1, char 25`},
		{s: `SHOW RETENTION POLICIES mydb`, err: `found mydb, expected ON at line 1, char 25`},
		{s: `SHOW RETENTION POLICIES ON`, err: `found EOF, expected identifier at line 1, char 28`},
		{s: `SHOW FOO`, err: `found FOO, expected CONTINUOUS, DATABASES, FIELD, GRANTS, MEASUREMENTS, RETENTION, SERIES, SERVERS, TAG, USERS at line 1, char 6`},
		{s: `SHOW STATS ON`, err: `found EOF, expected string at line 1, char 15`},
		{s: `SHOW GRANTS`, err: `found EOF, expected FOR at line 1, char 13`},
		{s: `SHOW GRANTS FOR`, err: `found EOF, expected identifier at line 1, char 17`},
		{s: `DROP CONTINUOUS`, err: `found EOF, expected QUERY at line 1, char 17`},
		{s: `DROP CONTINUOUS QUERY`, err: `found EOF, expected identifier at line 1, char 23`},
		{s: `DROP CONTINUOUS QUERY myquery`, err: `found EOF, expected ON at line 1, char 31`},
		{s: `DROP CONTINUOUS QUERY myquery ON`, err: `found EOF, expected identifier at line 1, char 34`},
		{s: `CREATE CONTINUOUS`, err: `found EOF, expected QUERY at line 1, char 19`},
		{s: `CREATE CONTINUOUS QUERY`, err: `found EOF, expected identifier at line 1, char 25`},
		{s: `DROP FOO`, err: `found FOO, expected SERIES, CONTINUOUS, MEASUREMENT at line 1, char 6`},
		{s: `DROP DATABASE`, err: `found EOF, expected identifier at line 1, char 15`},
		{s: `DROP RETENTION`, err: `found EOF, expected POLICY at line 1, char 16`},
		{s: `DROP RETENTION POLICY`, err: `found EOF, expected identifier at line 1, char 23`},
		{s: `DROP RETENTION POLICY "1h.cpu"`, err: `found EOF, expected ON at line 1, char 31`},
		{s: `DROP RETENTION POLICY "1h.cpu" ON`, err: `found EOF, expected identifier at line 1, char 35`},
		{s: `DROP USER`, err: `found EOF, expected identifier at line 1, char 11`},
		{s: `CREATE USER testuser`, err: `found EOF, expected WITH at line 1, char 22`},
		{s: `CREATE USER testuser WITH`, err: `found EOF, expected PASSWORD at line 1, char 27`},
		{s: `CREATE USER testuser WITH PASSWORD`, err: `found EOF, expected string at line 1, char 36`},
		{s: `CREATE USER testuser WITH PASSWORD 'pwd' WITH`, err: `found EOF, expected ALL at line 1, char 47`},
		{s: `CREATE USER testuser WITH PASSWORD 'pwd' WITH ALL`, err: `found EOF, expected PRIVILEGES at line 1, char 51`},
		{s: `GRANT`, err: `found EOF, expected READ, WRITE, ALL [PRIVILEGES] at line 1, char 7`},
		{s: `GRANT BOGUS`, err: `found BOGUS, expected READ, WRITE, ALL [PRIVILEGES] at line 1, char 7`},
		{s: `GRANT READ`, err: `found EOF, expected ON at line 1, char 12`},
		{s: `GRANT READ FROM`, err: `found FROM, expected ON at line 1, char 12`},
		{s: `GRANT READ ON`, err: `found EOF, expected identifier at line 1, char 15`},
		{s: `GRANT READ ON TO`, err: `found TO, expected identifier at line 1, char 15`},
		{s: `GRANT READ ON testdb`, err: `found EOF, expected TO at line 1, char 22`},
		{s: `GRANT READ ON testdb TO`, err: `found EOF, expected identifier at line 1, char 25`},
		{s: `GRANT READ TO`, err: `found TO, expected ON at line 1, char 12`},
		{s: `GRANT WRITE`, err: `found EOF, expected ON at line 1, char 13`},
		{s: `GRANT WRITE FROM`, err: `found FROM, expected ON at line 1, char 13`},
		{s: `GRANT WRITE ON`, err: `found EOF, expected identifier at line 1, char 16`},
		{s: `GRANT WRITE ON TO`, err: `found TO, expected identifier at line 1, char 16`},
		{s: `GRANT WRITE ON testdb`, err: `found EOF, expected TO at line 1, char 23`},
		{s: `GRANT WRITE ON testdb TO`, err: `found EOF, expected identifier at line 1, char 26`},
		{s: `GRANT WRITE TO`, err: `found TO, expected ON at line 1, char 13`},
		{s: `GRANT ALL`, err: `found EOF, expected ON, TO at line 1, char 11`},
		{s: `GRANT ALL PRIVILEGES`, err: `found EOF, expected ON, TO at line 1, char 22`},
		{s: `GRANT ALL FROM`, err: `found FROM, expected ON, TO at line 1, char 11`},
		{s: `GRANT ALL PRIVILEGES FROM`, err: `found FROM, expected ON, TO at line 1, char 22`},
		{s: `GRANT ALL ON`, err: `found EOF, expected identifier at line 1, char 14`},
		{s: `GRANT ALL PRIVILEGES ON`, err: `found EOF, expected identifier at line 1, char 25`},
		{s: `GRANT ALL ON TO`, err: `found TO, expected identifier at line 1, char 14`},
		{s: `GRANT ALL PRIVILEGES ON TO`, err: `found TO, expected identifier at line 1, char 25`},
		{s: `GRANT ALL ON testdb`, err: `found EOF, expected TO at line 1, char 21`},
		{s: `GRANT ALL PRIVILEGES ON testdb`, err: `found EOF, expected TO at line 1, char 32`},
		{s: `GRANT ALL ON testdb FROM`, err: `found FROM, expected TO at line 1, char 21`},
		{s: `GRANT ALL PRIVILEGES ON testdb FROM`, err: `found FROM, expected TO at line 1, char 32`},
		{s: `GRANT ALL ON testdb TO`, err: `found EOF, expected identifier at line 1, char 24`},
		{s: `GRANT ALL PRIVILEGES ON testdb TO`, err: `found EOF, expected identifier at line 1, char 35`},
		{s: `GRANT ALL TO`, err: `found EOF, expected identifier at line 1, char 14`},
		{s: `GRANT ALL PRIVILEGES TO`, err: `found EOF, expected identifier at line 1, char 25`},
		{s: `REVOKE`, err: `found EOF, expected READ, WRITE, ALL [PRIVILEGES] at line 1, char 8`},
		{s: `REVOKE BOGUS`, err: `found BOGUS, expected READ, WRITE, ALL [PRIVILEGES] at line 1, char 8`},
		{s: `REVOKE READ`, err: `found EOF, expected ON at line 1, char 13`},
		{s: `REVOKE READ TO`, err: `found TO, expected ON at line 1, char 13`},
		{s: `REVOKE READ ON`, err: `found EOF, expected identifier at line 1, char 16`},
		{s: `REVOKE READ ON FROM`, err: `found FROM, expected identifier at line 1, char 16`},
		{s: `REVOKE READ ON testdb`, err: `found EOF, expected FROM at line 1, char 23`},
		{s: `REVOKE READ ON testdb FROM`, err: `found EOF, expected identifier at line 1, char 28`},
		{s: `REVOKE READ FROM`, err: `found FROM, expected ON at line 1, char 13`},
		{s: `REVOKE WRITE`, err: `found EOF, expected ON at line 1, char 14`},
		{s: `REVOKE WRITE TO`, err: `found TO, expected ON at line 1, char 14`},
		{s: `REVOKE WRITE ON`, err: `found EOF, expected identifier at line 1, char 17`},
		{s: `REVOKE WRITE ON FROM`, err: `found FROM, expected identifier at line 1, char 17`},
		{s: `REVOKE WRITE ON testdb`, err: `found EOF, expected FROM at line 1, char 24`},
		{s: `REVOKE WRITE ON testdb FROM`, err: `found EOF, expected identifier at line 1, char 29`},
		{s: `REVOKE WRITE FROM`, err: `found FROM, expected ON at line 1, char 14`},
		{s: `REVOKE ALL`, err: `found EOF, expected ON, FROM at line 1, char 12`},
		{s: `REVOKE ALL PRIVILEGES`, err: `found EOF, expected ON, FROM at line 1, char 23`},
		{s: `REVOKE ALL TO`, err: `found TO, expected ON, FROM at line 1, char 12`},
		{s: `REVOKE ALL PRIVILEGES TO`, err: `found TO, expected ON, FROM at line 1, char 23`},
		{s: `REVOKE ALL ON`, err: `found EOF, expected identifier at line 1, char 15`},
		{s: `REVOKE ALL PRIVILEGES ON`, err: `found EOF, expected identifier at line 1, char 26`},
		{s: `REVOKE ALL ON FROM`, err: `found FROM, expected identifier at line 1, char 15`},
		{s: `REVOKE ALL PRIVILEGES ON FROM`, err: `found FROM, expected identifier at line 1, char 26`},
		{s: `REVOKE ALL ON testdb`, err: `found EOF, expected FROM at line 1, char 22`},
		{s: `REVOKE ALL PRIVILEGES ON testdb`, err: `found EOF, expected FROM at line 1, char 33`},
		{s: `REVOKE ALL ON testdb TO`, err: `found TO, expected FROM at line 1, char 22`},
		{s: `REVOKE ALL PRIVILEGES ON testdb TO`, err: `found TO, expected FROM at line 1, char 33`},
		{s: `REVOKE ALL ON testdb FROM`, err: `found EOF, expected identifier at line 1, char 27`},
		{s: `REVOKE ALL PRIVILEGES ON testdb FROM`, err: `found EOF, expected identifier at line 1, char 38`},
		{s: `REVOKE ALL FROM`, err: `found EOF, expected identifier at line 1, char 17`},
		{s: `REVOKE ALL PRIVILEGES FROM`, err: `found EOF, expected identifier at line 1, char 28`},
		{s: `CREATE RETENTION`, err: `found EOF, expected POLICY at line 1, char 18`},
		{s: `CREATE RETENTION POLICY`, err: `found EOF, expected identifier at line 1, char 25`},
		{s: `CREATE RETENTION POLICY policy1`, err: `found EOF, expected ON at line 1, char 33`},
		{s: `CREATE RETENTION POLICY policy1 ON`, err: `found EOF, expected identifier at line 1, char 36`},
		{s: `CREATE RETENTION POLICY policy1 ON testdb`, err: `found EOF, expected DURATION at line 1, char 43`},
		{s: `CREATE RETENTION POLICY policy1 ON testdb DURATION`, err: `found EOF, expected duration at line 1, char 52`},
		{s: `CREATE RETENTION POLICY policy1 ON testdb DURATION bad`, err: `found bad, expected duration at line 1, char 52`},
		{s: `CREATE RETENTION POLICY policy1 ON testdb DURATION 1h`, err: `found EOF, expected REPLICATION at line 1, char 54`},
		{s: `CREATE RETENTION POLICY policy1 ON testdb DURATION 1h REPLICATION`, err: `found EOF, expected number at line 1, char 67`},
		{s: `CREATE RETENTION POLICY policy1 ON testdb DURATION 1h REPLICATION 3.14`, err: `number must be an integer at line 1, char 67`},
		{s: `CREATE RETENTION POLICY policy1 ON testdb DURATION 1h REPLICATION 0`, err: `invalid value 0: must be 1 <= n <= 2147483647 at line 1, char 67`},
		{s: `CREATE RETENTION POLICY policy1 ON testdb DURATION 1h REPLICATION bad`, err: `found bad, expected number at line 1, char 67`},
		{s: `ALTER`, err: `found EOF, expected RETENTION at line 1, char 7`},
		{s: `ALTER RETENTION`, err: `found EOF, expected POLICY at line 1, char 17`},
		{s: `ALTER RETENTION POLICY`, err: `found EOF, expected identifier at line 1, char 24`},
		{s: `ALTER RETENTION POLICY policy1`, err: `found EOF, expected ON at line 1, char 32`}, {s: `ALTER RETENTION POLICY policy1 ON`, err: `found EOF, expected identifier at line 1, char 35`},
		{s: `ALTER RETENTION POLICY policy1 ON testdb`, err: `found EOF, expected DURATION, RETENTION, DEFAULT at line 1, char 42`},
		{s: `SET`, err: `found EOF, expected PASSWORD at line 1, char 5`},
		{s: `SET PASSWORD`, err: `found EOF, expected FOR at line 1, char 14`},
		{s: `SET PASSWORD something`, err: `found something, expected FOR at line 1, char 14`},
		{s: `SET PASSWORD FOR`, err: `found EOF, expected identifier at line 1, char 18`},
		{s: `SET PASSWORD FOR dejan`, err: `found EOF, expected = at line 1, char 24`},
		{s: `SET PASSWORD FOR dejan =`, err: `found EOF, expected string at line 1, char 25`},
		{s: `SET PASSWORD FOR dejan = bla`, err: `found bla, expected string at line 1, char 26`},
	}

	for i, tt := range tests {
		if tt.skip {
			t.Logf("skipping test of '%s'", tt.s)
			continue
		}
		stmt, err := influxql.NewParser(strings.NewReader(tt.s)).ParseStatement()

		// We are memoizing a field so for testing we need to...
		if s, ok := tt.stmt.(*influxql.SelectStatement); ok {
			s.GroupByInterval()
		} else if st, ok := stmt.(*influxql.CreateContinuousQueryStatement); ok { // if it's a CQ, there is a non-exported field that gets memoized during parsing that needs to be set
			if st != nil && st.Source != nil {
				tt.stmt.(*influxql.CreateContinuousQueryStatement).Source.GroupByInterval()
			}
		}

		if !reflect.DeepEqual(tt.err, errstring(err)) {
			t.Errorf("%d. %q: error mismatch:\n  exp=%s\n  got=%s\n\n", i, tt.s, tt.err, err)
		} else if tt.err == "" && !reflect.DeepEqual(tt.stmt, stmt) {
			t.Logf("\nexp=%s\ngot=%s\n", mustMarshalJSON(tt.stmt), mustMarshalJSON(stmt))
			t.Errorf("%d. %q\n\nstmt mismatch:\n\nexp=%#v\n\ngot=%#v\n\n", i, tt.s, tt.stmt, stmt)
		}
	}
}

// Ensure the parser can parse expressions into an AST.
func TestParser_ParseExpr(t *testing.T) {
	var tests = []struct {
		s    string
		expr influxql.Expr
		err  string
	}{
		// Primitives
		{s: `100`, expr: &influxql.NumberLiteral{Val: 100}},
		{s: `'foo bar'`, expr: &influxql.StringLiteral{Val: "foo bar"}},
		{s: `true`, expr: &influxql.BooleanLiteral{Val: true}},
		{s: `false`, expr: &influxql.BooleanLiteral{Val: false}},
		{s: `my_ident`, expr: &influxql.VarRef{Val: "my_ident"}},
		{s: `'2000-01-01 00:00:00'`, expr: &influxql.TimeLiteral{Val: mustParseTime("2000-01-01T00:00:00Z")}},
		{s: `'2000-01-01 00:00:00.232'`, expr: &influxql.TimeLiteral{Val: mustParseTime("2000-01-01T00:00:00.232Z")}},
		{s: `'2000-01-32 00:00:00'`, err: `unable to parse datetime at line 1, char 1`},
		{s: `'2000-01-01'`, expr: &influxql.TimeLiteral{Val: mustParseTime("2000-01-01T00:00:00Z")}},
		{s: `'2000-01-99'`, err: `unable to parse date at line 1, char 1`},

		// Simple binary expression
		{
			s: `1 + 2`,
			expr: &influxql.BinaryExpr{
				Op:  influxql.ADD,
				LHS: &influxql.NumberLiteral{Val: 1},
				RHS: &influxql.NumberLiteral{Val: 2},
			},
		},

		// Binary expression with LHS precedence
		{
			s: `1 * 2 + 3`,
			expr: &influxql.BinaryExpr{
				Op: influxql.ADD,
				LHS: &influxql.BinaryExpr{
					Op:  influxql.MUL,
					LHS: &influxql.NumberLiteral{Val: 1},
					RHS: &influxql.NumberLiteral{Val: 2},
				},
				RHS: &influxql.NumberLiteral{Val: 3},
			},
		},

		// Binary expression with RHS precedence
		{
			s: `1 + 2 * 3`,
			expr: &influxql.BinaryExpr{
				Op:  influxql.ADD,
				LHS: &influxql.NumberLiteral{Val: 1},
				RHS: &influxql.BinaryExpr{
					Op:  influxql.MUL,
					LHS: &influxql.NumberLiteral{Val: 2},
					RHS: &influxql.NumberLiteral{Val: 3},
				},
			},
		},

		// Binary expression with LHS paren group.
		{
			s: `(1 + 2) * 3`,
			expr: &influxql.BinaryExpr{
				Op: influxql.MUL,
				LHS: &influxql.ParenExpr{
					Expr: &influxql.BinaryExpr{
						Op:  influxql.ADD,
						LHS: &influxql.NumberLiteral{Val: 1},
						RHS: &influxql.NumberLiteral{Val: 2},
					},
				},
				RHS: &influxql.NumberLiteral{Val: 3},
			},
		},

		// Binary expression with no precedence, tests left associativity.
		{
			s: `1 * 2 * 3`,
			expr: &influxql.BinaryExpr{
				Op: influxql.MUL,
				LHS: &influxql.BinaryExpr{
					Op:  influxql.MUL,
					LHS: &influxql.NumberLiteral{Val: 1},
					RHS: &influxql.NumberLiteral{Val: 2},
				},
				RHS: &influxql.NumberLiteral{Val: 3},
			},
		},

		// Binary expression with regex.
		{
			s: "region =~ /us.*/",
			expr: &influxql.BinaryExpr{
				Op:  influxql.EQREGEX,
				LHS: &influxql.VarRef{Val: "region"},
				RHS: &influxql.RegexLiteral{Val: regexp.MustCompile(`us.*`)},
			},
		},

		// Complex binary expression.
		{
			s: `value + 3 < 30 AND 1 + 2 OR true`,
			expr: &influxql.BinaryExpr{
				Op: influxql.OR,
				LHS: &influxql.BinaryExpr{
					Op: influxql.AND,
					LHS: &influxql.BinaryExpr{
						Op: influxql.LT,
						LHS: &influxql.BinaryExpr{
							Op:  influxql.ADD,
							LHS: &influxql.VarRef{Val: "value"},
							RHS: &influxql.NumberLiteral{Val: 3},
						},
						RHS: &influxql.NumberLiteral{Val: 30},
					},
					RHS: &influxql.BinaryExpr{
						Op:  influxql.ADD,
						LHS: &influxql.NumberLiteral{Val: 1},
						RHS: &influxql.NumberLiteral{Val: 2},
					},
				},
				RHS: &influxql.BooleanLiteral{Val: true},
			},
		},

		// Complex binary expression.
		{
			s: `time > now() - 1d AND time < now() + 1d`,
			expr: &influxql.BinaryExpr{
				Op: influxql.AND,
				LHS: &influxql.BinaryExpr{
					Op:  influxql.GT,
					LHS: &influxql.VarRef{Val: "time"},
					RHS: &influxql.BinaryExpr{
						Op:  influxql.SUB,
						LHS: &influxql.Call{Name: "now"},
						RHS: &influxql.DurationLiteral{Val: mustParseDuration("1d")},
					},
				},
				RHS: &influxql.BinaryExpr{
					Op:  influxql.LT,
					LHS: &influxql.VarRef{Val: "time"},
					RHS: &influxql.BinaryExpr{
						Op:  influxql.ADD,
						LHS: &influxql.Call{Name: "now"},
						RHS: &influxql.DurationLiteral{Val: mustParseDuration("1d")},
					},
				},
			},
		},

		// Function call (empty)
		{
			s: `my_func()`,
			expr: &influxql.Call{
				Name: "my_func",
			},
		},

		// Function call (multi-arg)
		{
			s: `my_func(1, 2 + 3)`,
			expr: &influxql.Call{
				Name: "my_func",
				Args: []influxql.Expr{
					&influxql.NumberLiteral{Val: 1},
					&influxql.BinaryExpr{
						Op:  influxql.ADD,
						LHS: &influxql.NumberLiteral{Val: 2},
						RHS: &influxql.NumberLiteral{Val: 3},
					},
				},
			},
		},
	}

	for i, tt := range tests {
		expr, err := influxql.NewParser(strings.NewReader(tt.s)).ParseExpr()
		if !reflect.DeepEqual(tt.err, errstring(err)) {
			t.Errorf("%d. %q: error mismatch:\n  exp=%s\n  got=%s\n\n", i, tt.s, tt.err, err)
		} else if tt.err == "" && !reflect.DeepEqual(tt.expr, expr) {
			t.Errorf("%d. %q\n\nexpr mismatch:\n\nexp=%#v\n\ngot=%#v\n\n", i, tt.s, tt.expr, expr)
		}
	}
}

// Ensure a time duration can be parsed.
func TestParseDuration(t *testing.T) {
	var tests = []struct {
		s   string
		d   time.Duration
		err string
	}{
		{s: `3`, d: 3 * time.Microsecond},
		{s: `1000`, d: 1000 * time.Microsecond},
		{s: `10u`, d: 10 * time.Microsecond},
		{s: `10µ`, d: 10 * time.Microsecond},
		{s: `15ms`, d: 15 * time.Millisecond},
		{s: `100s`, d: 100 * time.Second},
		{s: `2m`, d: 2 * time.Minute},
		{s: `2h`, d: 2 * time.Hour},
		{s: `2d`, d: 2 * 24 * time.Hour},
		{s: `2w`, d: 2 * 7 * 24 * time.Hour},

		{s: ``, err: "invalid duration"},
		{s: `w`, err: "invalid duration"},
		{s: `1.2w`, err: "invalid duration"},
		{s: `10x`, err: "invalid duration"},
	}

	for i, tt := range tests {
		d, err := influxql.ParseDuration(tt.s)
		if !reflect.DeepEqual(tt.err, errstring(err)) {
			t.Errorf("%d. %q: error mismatch:\n  exp=%s\n  got=%s\n\n", i, tt.s, tt.err, err)
		} else if tt.d != d {
			t.Errorf("%d. %q\n\nduration mismatch:\n\nexp=%#v\n\ngot=%#v\n\n", i, tt.s, tt.d, d)
		}
	}
}

// Ensure a time duration can be formatted.
func TestFormatDuration(t *testing.T) {
	var tests = []struct {
		d time.Duration
		s string
	}{
		{d: 3 * time.Microsecond, s: `3`},
		{d: 1001 * time.Microsecond, s: `1001`},
		{d: 15 * time.Millisecond, s: `15ms`},
		{d: 100 * time.Second, s: `100s`},
		{d: 2 * time.Minute, s: `2m`},
		{d: 2 * time.Hour, s: `2h`},
		{d: 2 * 24 * time.Hour, s: `2d`},
		{d: 2 * 7 * 24 * time.Hour, s: `2w`},
	}

	for i, tt := range tests {
		s := influxql.FormatDuration(tt.d)
		if tt.s != s {
			t.Errorf("%d. %v: mismatch: %s != %s", i, tt.d, tt.s, s)
		}
	}
}

// Ensure a string can be quoted.
func TestQuote(t *testing.T) {
	for i, tt := range []struct {
		in  string
		out string
	}{
		{``, `''`},
		{`foo`, `'foo'`},
		{"foo\nbar", `'foo\nbar'`},
		{`foo bar\\`, `'foo bar\\\\'`},
		{`'foo'`, `'\'foo\''`},
	} {
		if out := influxql.QuoteString(tt.in); tt.out != out {
			t.Errorf("%d. %s: mismatch: %s != %s", i, tt.in, tt.out, out)
		}
	}
}

// Ensure an identifier's segments can be quoted.
func TestQuoteIdent(t *testing.T) {
	for i, tt := range []struct {
		ident []string
		s     string
	}{
		{[]string{``}, ``},
		{[]string{`foo`, `bar`}, `"foo".bar`},
		{[]string{`foo`, ``, `bar`}, `"foo"..bar`},
		{[]string{`foo bar`, `baz`}, `"foo bar".baz`},
		{[]string{`foo.bar`, `baz`}, `"foo.bar".baz`},
		{[]string{`foo.bar`, `rp`, `baz`}, `"foo.bar"."rp".baz`},
		{[]string{`foo.bar`, `rp`, `1baz`}, `"foo.bar"."rp"."1baz"`},
	} {
		if s := influxql.QuoteIdent(tt.ident...); tt.s != s {
			t.Errorf("%d. %s: mismatch: %s != %s", i, tt.ident, tt.s, s)
		}
	}
}

// Ensure DropSeriesStatement can convert to a string
func TestDropSeriesStatement_String(t *testing.T) {
	var tests = []struct {
		s    string
		stmt influxql.Statement
	}{
		{
			s:    `DROP SERIES FROM src`,
			stmt: &influxql.DropSeriesStatement{Sources: []influxql.Source{&influxql.Measurement{Name: "src"}}},
		},
		{
			s: `DROP SERIES FROM src WHERE host = 'hosta.influxdb.org'`,
			stmt: &influxql.DropSeriesStatement{
				Sources: []influxql.Source{&influxql.Measurement{Name: "src"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "host"},
					RHS: &influxql.StringLiteral{Val: "hosta.influxdb.org"},
				},
			},
		},
		{
			s: `DROP SERIES FROM src WHERE host = 'hosta.influxdb.org'`,
			stmt: &influxql.DropSeriesStatement{
				Sources: []influxql.Source{&influxql.Measurement{Name: "src"}},
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "host"},
					RHS: &influxql.StringLiteral{Val: "hosta.influxdb.org"},
				},
			},
		},
		{
			s: `DROP SERIES WHERE host = 'hosta.influxdb.org'`,
			stmt: &influxql.DropSeriesStatement{
				Condition: &influxql.BinaryExpr{
					Op:  influxql.EQ,
					LHS: &influxql.VarRef{Val: "host"},
					RHS: &influxql.StringLiteral{Val: "hosta.influxdb.org"},
				},
			},
		},
	}

	for _, test := range tests {
		s := test.stmt.String()
		if s != test.s {
			t.Errorf("error rendering string. expected %s, actual: %s", test.s, s)
		}
	}
}

func BenchmarkParserParseStatement(b *testing.B) {
	b.ReportAllocs()
	s := `SELECT field FROM "series" WHERE value > 10`
	for i := 0; i < b.N; i++ {
		if stmt, err := influxql.NewParser(strings.NewReader(s)).ParseStatement(); err != nil {
			b.Fatalf("unexpected error: %s", err)
		} else if stmt == nil {
			b.Fatalf("expected statement: %s", stmt)
		}
	}
	b.SetBytes(int64(len(s)))
}

// MustParseSelectStatement parses a select statement. Panic on error.
func MustParseSelectStatement(s string) *influxql.SelectStatement {
	stmt, err := influxql.NewParser(strings.NewReader(s)).ParseStatement()
	panicIfErr(err)
	return stmt.(*influxql.SelectStatement)
}

// MustParseExpr parses an expression. Panic on error.
func MustParseExpr(s string) influxql.Expr {
	expr, err := influxql.NewParser(strings.NewReader(s)).ParseExpr()
	panicIfErr(err)
	return expr
}

// errstring converts an error to its string representation.
func errstring(err error) string {
	if err != nil {
		return err.Error()
	}
	return ""
}

// newAlterRetentionPolicyStatement creates an initialized AlterRetentionPolicyStatement.
func newAlterRetentionPolicyStatement(name string, DB string, d time.Duration, replication int, dfault bool) *influxql.AlterRetentionPolicyStatement {
	stmt := &influxql.AlterRetentionPolicyStatement{
		Name:     name,
		Database: DB,
		Default:  dfault,
	}

	if d > -1 {
		stmt.Duration = &d
	}

	if replication > -1 {
		stmt.Replication = &replication
	}

	return stmt
}

// mustMarshalJSON encodes a value to JSON.
func mustMarshalJSON(v interface{}) []byte {
	b, err := json.Marshal(v)
	panicIfErr(err)
	return b
}

func mustParseDuration(s string) time.Duration {
	d, err := influxql.ParseDuration(s)
	panicIfErr(err)
	return d
}

func panicIfErr(err error) {
	if err != nil {
		panic(err)
	}
}
