package influxql_test

import (
	"testing"

	"github.com/influxdata/influxdb/influxql"
)

func TestSanitize(t *testing.T) {
	var tests = []struct {
		s    string
		stmt string
	}{
		// Proper statements that should be redacted.
		{
			s:    `create user "admin" with password 'admin'`,
			stmt: `create user "admin" with password [REDACTED]`,
		},
		{
			s:    `set password for "admin" = 'admin'`,
			stmt: `set password for "admin" = [REDACTED]`,
		},

		// Common invalid statements that should still be redacted.
		{
			s:    `create user "admin" with password "admin"`,
			stmt: `create user "admin" with password [REDACTED]`,
		},
		{
			s:    `set password for "admin" = "admin"`,
			stmt: `set password for "admin" = [REDACTED]`,
		},
	}

	for i, tt := range tests {
		stmt := influxql.Sanitize(tt.s)
		if tt.stmt != stmt {
			t.Errorf("%d. %q\n\nsanitize mismatch:\n\nexp=%#v\n\ngot=%#v\n\n", i, tt.s, tt.stmt, stmt)
		}
	}
}

func BenchmarkSanitize(b *testing.B) {
	b.ReportAllocs()
	q := `create user "admin" with password 'admin'; set password for "admin" = 'admin'`
	for i := 0; i < b.N; i++ {
		influxql.Sanitize(q)
	}
}
