package statement

import (
	//	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/influxdata/influxdb/stress/v2/statement"
)

func newParserFromString(s string) *Parser {
	f := strings.NewReader(s)
	p := NewParser(f)

	return p
}

func TestParser_ParseStatement(t *testing.T) {
	var tests = []struct {
		skip bool
		s    string
		stmt statement.Statement
		err  string
	}{

		// QUERY

		{
			s:    "QUERY basicCount\nSELECT count(%f) FROM cpu\nDO 100",
			stmt: &statement.QueryStatement{Name: "basicCount", TemplateString: "SELECT count(%v) FROM cpu", Args: []string{"%f"}, Count: 100},
		},

		{
			s:    "QUERY basicCount\nSELECT count(%f) FROM %m\nDO 100",
			stmt: &statement.QueryStatement{Name: "basicCount", TemplateString: "SELECT count(%v) FROM %v", Args: []string{"%f", "%m"}, Count: 100},
		},

		{
			skip: true, // SHOULD CAUSE AN ERROR
			s:    "QUERY\nSELECT count(%f) FROM %m\nDO 100",
			err:  "Missing Name",
		},

		// INSERT

		{
			s: "INSERT mockCpu\ncpu,\nhost=[us-west|us-east|eu-north],server_id=[str rand(7) 1000]\nbusy=[int rand(1000) 100],free=[float rand(10) 0]\n100000 10s",
			stmt: &statement.InsertStatement{
				Name:           "mockCpu",
				TemplateString: "cpu,host=%v,server_id=%v busy=%v,free=%v %v",
				TagCount:       2,
				Templates: []*statement.Template{
					&statement.Template{
						Tags: []string{"us-west", "us-east", "eu-north"},
					},
					&statement.Template{
						Function: &statement.Function{Type: "str", Fn: "rand", Argument: 7, Count: 1000},
					},
					&statement.Template{
						Function: &statement.Function{Type: "int", Fn: "rand", Argument: 1000, Count: 100},
					},
					&statement.Template{
						Function: &statement.Function{Type: "float", Fn: "rand", Argument: 10, Count: 0},
					},
				},
				Timestamp: &statement.Timestamp{
					Count:    100000,
					Duration: time.Duration(10 * time.Second),
				},
			},
		},

		{
			s: "INSERT mockCpu\ncpu,host=[us-west|us-east|eu-north],server_id=[str rand(7) 1000]\nbusy=[int rand(1000) 100],free=[float rand(10) 0]\n100000 10s",
			stmt: &statement.InsertStatement{
				Name:           "mockCpu",
				TemplateString: "cpu,host=%v,server_id=%v busy=%v,free=%v %v",
				TagCount:       2,
				Templates: []*statement.Template{
					&statement.Template{
						Tags: []string{"us-west", "us-east", "eu-north"},
					},
					&statement.Template{
						Function: &statement.Function{Type: "str", Fn: "rand", Argument: 7, Count: 1000},
					},
					&statement.Template{
						Function: &statement.Function{Type: "int", Fn: "rand", Argument: 1000, Count: 100},
					},
					&statement.Template{
						Function: &statement.Function{Type: "float", Fn: "rand", Argument: 10, Count: 0},
					},
				},
				Timestamp: &statement.Timestamp{
					Count:    100000,
					Duration: time.Duration(10 * time.Second),
				},
			},
		},

		{
			s: "INSERT mockCpu\n[str rand(1000) 10],\nhost=[us-west|us-east|eu-north],server_id=[str rand(7) 1000],other=x\nbusy=[int rand(1000) 100],free=[float rand(10) 0]\n100000 10s",
			stmt: &statement.InsertStatement{
				Name:           "mockCpu",
				TemplateString: "%v,host=%v,server_id=%v,other=x busy=%v,free=%v %v",
				TagCount:       3,
				Templates: []*statement.Template{
					&statement.Template{
						Function: &statement.Function{Type: "str", Fn: "rand", Argument: 1000, Count: 10},
					},
					&statement.Template{
						Tags: []string{"us-west", "us-east", "eu-north"},
					},
					&statement.Template{
						Function: &statement.Function{Type: "str", Fn: "rand", Argument: 7, Count: 1000},
					},
					&statement.Template{
						Function: &statement.Function{Type: "int", Fn: "rand", Argument: 1000, Count: 100},
					},
					&statement.Template{
						Function: &statement.Function{Type: "float", Fn: "rand", Argument: 10, Count: 0},
					},
				},
				Timestamp: &statement.Timestamp{
					Count:    100000,
					Duration: time.Duration(10 * time.Second),
				},
			},
		},

		{
			skip: true, // Expected error not working
			s:    "INSERT\ncpu,\nhost=[us-west|us-east|eu-north],server_id=[str rand(7) 1000]\nbusy=[int rand(1000) 100],free=[float rand(10) 0]\n100000 10s",
			err:  `found ",", expected WS`,
		},

		// EXEC

		{
			s:    `EXEC other_script`,
			stmt: &statement.ExecStatement{Script: "other_script"},
		},

		{
			skip: true, // Implement
			s:    `EXEC other_script.sh`,
			stmt: &statement.ExecStatement{Script: "other_script.sh"},
		},

		{
			skip: true, // Implement
			s:    `EXEC ../other_script.sh`,
			stmt: &statement.ExecStatement{Script: "../other_script.sh"},
		},

		{
			skip: true, // Implement
			s:    `EXEC /path/to/some/other_script.sh`,
			stmt: &statement.ExecStatement{Script: "/path/to/some/other_script.sh"},
		},

		// GO

		{
			skip: true,
			s:    "GO INSERT mockCpu\ncpu,\nhost=[us-west|us-east|eu-north],server_id=[str rand(7) 1000]\nbusy=[int rand(1000) 100],free=[float rand(10) 0]\n100000 10s",
			stmt: &statement.GoStatement{
				Statement: &statement.InsertStatement{
					Name:           "mockCpu",
					TemplateString: "cpu,host=%v,server_id=%v busy=%v,free=%v %v",
					Templates: []*statement.Template{
						&statement.Template{
							Tags: []string{"us-west", "us-east", "eu-north"},
						},
						&statement.Template{
							Function: &statement.Function{Type: "str", Fn: "rand", Argument: 7, Count: 1000},
						},
						&statement.Template{
							Function: &statement.Function{Type: "int", Fn: "rand", Argument: 1000, Count: 100},
						},
						&statement.Template{
							Function: &statement.Function{Type: "float", Fn: "rand", Argument: 10, Count: 0},
						},
					},
					Timestamp: &statement.Timestamp{
						Count:    100000,
						Duration: time.Duration(10 * time.Second),
					},
				},
			},
		},

		{
			skip: true,
			s:    "GO QUERY basicCount\nSELECT count(free) FROM cpu\nDO 100",
			stmt: &statement.GoStatement{
				Statement: &statement.QueryStatement{Name: "basicCount", TemplateString: "SELECT count(free) FROM cpu", Count: 100},
			},
		},

		{
			skip: true,
			s:    `GO EXEC other_script`,
			stmt: &statement.GoStatement{
				Statement: &statement.ExecStatement{Script: "other_script"},
			},
		},

		// SET

		{
			s:    `SET database [stress]`,
			stmt: &statement.SetStatement{Var: "database", Value: "stress"},
		},

		// WAIT

		{
			s:    `Wait`,
			stmt: &statement.WaitStatement{},
		},
	}

	for _, tst := range tests {

		if tst.skip {
			continue
		}

		stmt, err := newParserFromString(tst.s).Parse()
		tst.stmt.SetID("x")

		if err != nil && err.Error() != tst.err {
			t.Errorf("REAL ERROR: %v\nExpected ERROR: %v\n", err, tst.err)
		} else if err != nil && tst.err == err.Error() {
			t.Errorf("REAL ERROR: %v\nExpected ERROR: %v\n", err, tst.err)
		} else if stmt.SetID("x"); !reflect.DeepEqual(stmt, tst.stmt) {
			t.Errorf("Expected\n%#v\n%#v", tst.stmt, stmt)
		}
	}

}
