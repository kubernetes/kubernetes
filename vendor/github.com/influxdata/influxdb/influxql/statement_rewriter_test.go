package influxql_test

import (
	"testing"

	"github.com/influxdata/influxdb/influxql"
)

func TestRewriteStatement(t *testing.T) {
	tests := []struct {
		stmt string
		s    string
	}{
		{
			stmt: `SHOW FIELD KEYS`,
			s:    `SELECT fieldKey, fieldType FROM _fieldKeys`,
		},
		{
			stmt: `SHOW FIELD KEYS ON db0`,
			s:    `SELECT fieldKey, fieldType FROM db0.._fieldKeys`,
		},
		{
			stmt: `SHOW FIELD KEYS FROM cpu`,
			s:    `SELECT fieldKey, fieldType FROM _fieldKeys WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW FIELD KEYS ON db0 FROM cpu`,
			s:    `SELECT fieldKey, fieldType FROM db0.._fieldKeys WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW FIELD KEYS FROM /c.*/`,
			s:    `SELECT fieldKey, fieldType FROM _fieldKeys WHERE _name =~ /c.*/`,
		},
		{
			stmt: `SHOW FIELD KEYS ON db0 FROM /c.*/`,
			s:    `SELECT fieldKey, fieldType FROM db0.._fieldKeys WHERE _name =~ /c.*/`,
		},
		{
			stmt: `SHOW FIELD KEYS FROM mydb.myrp2.cpu`,
			s:    `SELECT fieldKey, fieldType FROM mydb.myrp2._fieldKeys WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW FIELD KEYS ON db0 FROM mydb.myrp2.cpu`,
			s:    `SELECT fieldKey, fieldType FROM mydb.myrp2._fieldKeys WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW FIELD KEYS FROM mydb.myrp2./c.*/`,
			s:    `SELECT fieldKey, fieldType FROM mydb.myrp2._fieldKeys WHERE _name =~ /c.*/`,
		},
		{
			stmt: `SHOW FIELD KEYS ON db0 FROM mydb.myrp2./c.*/`,
			s:    `SELECT fieldKey, fieldType FROM mydb.myrp2._fieldKeys WHERE _name =~ /c.*/`,
		},
		{
			stmt: `SHOW SERIES`,
			s:    `SELECT "key" FROM _series`,
		},
		{
			stmt: `SHOW SERIES ON db0`,
			s:    `SELECT "key" FROM db0.._series`,
		},
		{
			stmt: `SHOW SERIES FROM cpu`,
			s:    `SELECT "key" FROM _series WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW SERIES ON db0 FROM cpu`,
			s:    `SELECT "key" FROM db0.._series WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW SERIES FROM mydb.myrp1.cpu`,
			s:    `SELECT "key" FROM mydb.myrp1._series WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW SERIES ON db0 FROM mydb.myrp1.cpu`,
			s:    `SELECT "key" FROM mydb.myrp1._series WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW SERIES FROM mydb.myrp1./c.*/`,
			s:    `SELECT "key" FROM mydb.myrp1._series WHERE _name =~ /c.*/`,
		},
		{
			stmt: `SHOW SERIES ON db0 FROM mydb.myrp1./c.*/`,
			s:    `SELECT "key" FROM mydb.myrp1._series WHERE _name =~ /c.*/`,
		},
		{
			stmt: `SHOW TAG KEYS`,
			s:    `SELECT tagKey FROM _tagKeys`,
		},
		{
			stmt: `SHOW TAG KEYS ON db0`,
			s:    `SELECT tagKey FROM db0.._tagKeys`,
		},
		{
			stmt: `SHOW TAG KEYS FROM cpu`,
			s:    `SELECT tagKey FROM _tagKeys WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW TAG KEYS ON db0 FROM cpu`,
			s:    `SELECT tagKey FROM db0.._tagKeys WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW TAG KEYS FROM /c.*/`,
			s:    `SELECT tagKey FROM _tagKeys WHERE _name =~ /c.*/`,
		},
		{
			stmt: `SHOW TAG KEYS ON db0 FROM /c.*/`,
			s:    `SELECT tagKey FROM db0.._tagKeys WHERE _name =~ /c.*/`,
		},
		{
			stmt: `SHOW TAG KEYS FROM cpu WHERE region = 'uswest'`,
			s:    `SELECT tagKey FROM _tagKeys WHERE (_name = 'cpu') AND (region = 'uswest')`,
		},
		{
			stmt: `SHOW TAG KEYS ON db0 FROM cpu WHERE region = 'uswest'`,
			s:    `SELECT tagKey FROM db0.._tagKeys WHERE (_name = 'cpu') AND (region = 'uswest')`,
		},
		{
			stmt: `SHOW TAG KEYS FROM mydb.myrp1.cpu`,
			s:    `SELECT tagKey FROM mydb.myrp1._tagKeys WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW TAG KEYS ON db0 FROM mydb.myrp1.cpu`,
			s:    `SELECT tagKey FROM mydb.myrp1._tagKeys WHERE _name = 'cpu'`,
		},
		{
			stmt: `SHOW TAG KEYS FROM mydb.myrp1./c.*/`,
			s:    `SELECT tagKey FROM mydb.myrp1._tagKeys WHERE _name =~ /c.*/`,
		},
		{
			stmt: `SHOW TAG KEYS ON db0 FROM mydb.myrp1./c.*/`,
			s:    `SELECT tagKey FROM mydb.myrp1._tagKeys WHERE _name =~ /c.*/`,
		},
		{
			stmt: `SHOW TAG KEYS FROM mydb.myrp1.cpu WHERE region = 'uswest'`,
			s:    `SELECT tagKey FROM mydb.myrp1._tagKeys WHERE (_name = 'cpu') AND (region = 'uswest')`,
		},
		{
			stmt: `SHOW TAG KEYS ON db0 FROM mydb.myrp1.cpu WHERE region = 'uswest'`,
			s:    `SELECT tagKey FROM mydb.myrp1._tagKeys WHERE (_name = 'cpu') AND (region = 'uswest')`,
		},
		{
			stmt: `SELECT value FROM cpu`,
			s:    `SELECT value FROM cpu`,
		},
	}

	for _, test := range tests {
		stmt, err := influxql.ParseStatement(test.stmt)
		if err != nil {
			t.Errorf("error parsing statement: %s", err)
		} else {
			stmt, err = influxql.RewriteStatement(stmt)
			if err != nil {
				t.Errorf("error rewriting statement: %s", err)
			} else if s := stmt.String(); s != test.s {
				t.Errorf("error rendering string. expected %s, actual: %s", test.s, s)
			}
		}
	}
}
