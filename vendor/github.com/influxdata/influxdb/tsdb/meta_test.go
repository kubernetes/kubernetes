package tsdb_test

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/tsdb"
)

// Test comparing SeriesIDs for equality.
func TestSeriesIDs_Equals(t *testing.T) {
	ids1 := tsdb.SeriesIDs([]uint64{1, 2, 3})
	ids2 := tsdb.SeriesIDs([]uint64{1, 2, 3})
	ids3 := tsdb.SeriesIDs([]uint64{4, 5, 6})

	if !ids1.Equals(ids2) {
		t.Fatal("expected ids1 == ids2")
	} else if ids1.Equals(ids3) {
		t.Fatal("expected ids1 != ids3")
	}
}

// Test intersecting sets of SeriesIDs.
func TestSeriesIDs_Intersect(t *testing.T) {
	// Test swaping l & r, all branches of if-else, and exit loop when 'j < len(r)'
	ids1 := tsdb.SeriesIDs([]uint64{1, 3, 4, 5, 6})
	ids2 := tsdb.SeriesIDs([]uint64{1, 2, 3, 7})
	exp := tsdb.SeriesIDs([]uint64{1, 3})
	got := ids1.Intersect(ids2)

	if !exp.Equals(got) {
		t.Fatalf("exp=%v, got=%v", exp, got)
	}

	// Test exit for loop when 'i < len(l)'
	ids1 = tsdb.SeriesIDs([]uint64{1})
	ids2 = tsdb.SeriesIDs([]uint64{1, 2})
	exp = tsdb.SeriesIDs([]uint64{1})
	got = ids1.Intersect(ids2)

	if !exp.Equals(got) {
		t.Fatalf("exp=%v, got=%v", exp, got)
	}
}

// Test union sets of SeriesIDs.
func TestSeriesIDs_Union(t *testing.T) {
	// Test all branches of if-else, exit loop because of 'j < len(r)', and append remainder from left.
	ids1 := tsdb.SeriesIDs([]uint64{1, 2, 3, 7})
	ids2 := tsdb.SeriesIDs([]uint64{1, 3, 4, 5, 6})
	exp := tsdb.SeriesIDs([]uint64{1, 2, 3, 4, 5, 6, 7})
	got := ids1.Union(ids2)

	if !exp.Equals(got) {
		t.Fatalf("exp=%v, got=%v", exp, got)
	}

	// Test exit because of 'i < len(l)' and append remainder from right.
	ids1 = tsdb.SeriesIDs([]uint64{1})
	ids2 = tsdb.SeriesIDs([]uint64{1, 2})
	exp = tsdb.SeriesIDs([]uint64{1, 2})
	got = ids1.Union(ids2)

	if !exp.Equals(got) {
		t.Fatalf("exp=%v, got=%v", exp, got)
	}
}

// Test removing one set of SeriesIDs from another.
func TestSeriesIDs_Reject(t *testing.T) {
	// Test all branches of if-else, exit loop because of 'j < len(r)', and append remainder from left.
	ids1 := tsdb.SeriesIDs([]uint64{1, 2, 3, 7})
	ids2 := tsdb.SeriesIDs([]uint64{1, 3, 4, 5, 6})
	exp := tsdb.SeriesIDs([]uint64{2, 7})
	got := ids1.Reject(ids2)

	if !exp.Equals(got) {
		t.Fatalf("exp=%v, got=%v", exp, got)
	}

	// Test exit because of 'i < len(l)'.
	ids1 = tsdb.SeriesIDs([]uint64{1})
	ids2 = tsdb.SeriesIDs([]uint64{1, 2})
	exp = tsdb.SeriesIDs{}
	got = ids1.Reject(ids2)

	if !exp.Equals(got) {
		t.Fatalf("exp=%v, got=%v", exp, got)
	}
}

func TestMeasurement_AppendSeriesKeysByID_Missing(t *testing.T) {
	m := tsdb.NewMeasurement("cpu")
	var dst []string
	dst = m.AppendSeriesKeysByID(dst, []uint64{1})
	if exp, got := 0, len(dst); exp != got {
		t.Fatalf("series len mismatch: exp %v, got %v", exp, got)
	}
}

func TestMeasurement_AppendSeriesKeysByID_Exists(t *testing.T) {
	m := tsdb.NewMeasurement("cpu")
	s := tsdb.NewSeries("cpu,host=foo", models.Tags{models.Tag{Key: []byte("host"), Value: []byte("foo")}})
	s.ID = 1
	m.AddSeries(s)

	var dst []string
	dst = m.AppendSeriesKeysByID(dst, []uint64{1})
	if exp, got := 1, len(dst); exp != got {
		t.Fatalf("series len mismatch: exp %v, got %v", exp, got)
	}

	if exp, got := "cpu,host=foo", dst[0]; exp != got {
		t.Fatalf("series mismatch: exp %v, got %v", exp, got)
	}
}

func BenchmarkMeasurement_SeriesIDForExp_EQRegex(b *testing.B) {
	m := tsdb.NewMeasurement("cpu")
	for i := 0; i < 100000; i++ {
		s := tsdb.NewSeries("cpu", models.Tags{models.Tag{
			Key:   []byte("host"),
			Value: []byte(fmt.Sprintf("host%d", i))}})
		s.ID = uint64(i)
		m.AddSeries(s)
	}

	if exp, got := 100000, len(m.SeriesKeys()); exp != got {
		b.Fatalf("series count mismatch: exp %v got %v", exp, got)
	}

	stmt, err := influxql.NewParser(strings.NewReader(`SELECT * FROM cpu WHERE host =~ /host\d+/`)).ParseStatement()
	if err != nil {
		b.Fatalf("invalid statement: %s", err)
	}

	selectStmt := stmt.(*influxql.SelectStatement)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ids := m.IDsForExpr(selectStmt.Condition.(*influxql.BinaryExpr))
		if exp, got := 100000, len(ids); exp != got {
			b.Fatalf("series count mismatch: exp %v got %v", exp, got)
		}

	}
}

func BenchmarkMeasurement_SeriesIDForExp_NERegex(b *testing.B) {
	m := tsdb.NewMeasurement("cpu")
	for i := 0; i < 100000; i++ {
		s := tsdb.NewSeries("cpu", models.Tags{models.Tag{
			Key:   []byte("host"),
			Value: []byte(fmt.Sprintf("host%d", i))}})
		s.ID = uint64(i)
		m.AddSeries(s)
	}

	if exp, got := 100000, len(m.SeriesKeys()); exp != got {
		b.Fatalf("series count mismatch: exp %v got %v", exp, got)
	}

	stmt, err := influxql.NewParser(strings.NewReader(`SELECT * FROM cpu WHERE host !~ /foo\d+/`)).ParseStatement()
	if err != nil {
		b.Fatalf("invalid statement: %s", err)
	}

	selectStmt := stmt.(*influxql.SelectStatement)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ids := m.IDsForExpr(selectStmt.Condition.(*influxql.BinaryExpr))
		if exp, got := 100000, len(ids); exp != got {
			b.Fatalf("series count mismatch: exp %v got %v", exp, got)
		}

	}

}

// Ensure tags can be marshaled into a byte slice.
func TestMarshalTags(t *testing.T) {
	for i, tt := range []struct {
		tags   map[string]string
		result []byte
	}{
		{
			tags:   nil,
			result: nil,
		},
		{
			tags:   map[string]string{"foo": "bar"},
			result: []byte(`foo|bar`),
		},
		{
			tags:   map[string]string{"foo": "bar", "baz": "battttt"},
			result: []byte(`baz|foo|battttt|bar`),
		},
		{
			tags:   map[string]string{"baz": "battttt", "foo": "bar"},
			result: []byte(`baz|foo|battttt|bar`),
		},
	} {
		result := tsdb.MarshalTags(tt.tags)
		if !bytes.Equal(result, tt.result) {
			t.Fatalf("%d. unexpected result: exp=%s, got=%s", i, tt.result, result)
		}
	}
}

func BenchmarkMarshalTags_KeyN1(b *testing.B)  { benchmarkMarshalTags(b, 1) }
func BenchmarkMarshalTags_KeyN3(b *testing.B)  { benchmarkMarshalTags(b, 3) }
func BenchmarkMarshalTags_KeyN5(b *testing.B)  { benchmarkMarshalTags(b, 5) }
func BenchmarkMarshalTags_KeyN10(b *testing.B) { benchmarkMarshalTags(b, 10) }

func benchmarkMarshalTags(b *testing.B, keyN int) {
	const keySize, valueSize = 8, 15

	// Generate tag map.
	tags := make(map[string]string)
	for i := 0; i < keyN; i++ {
		tags[fmt.Sprintf("%0*d", keySize, i)] = fmt.Sprintf("%0*d", valueSize, i)
	}

	// Unmarshal map into byte slice.
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		tsdb.MarshalTags(tags)
	}
}

func BenchmarkCreateSeriesIndex_1K(b *testing.B) {
	benchmarkCreateSeriesIndex(b, genTestSeries(38, 3, 3))
}

func BenchmarkCreateSeriesIndex_100K(b *testing.B) {
	benchmarkCreateSeriesIndex(b, genTestSeries(32, 5, 5))
}

func BenchmarkCreateSeriesIndex_1M(b *testing.B) {
	benchmarkCreateSeriesIndex(b, genTestSeries(330, 5, 5))
}

func benchmarkCreateSeriesIndex(b *testing.B, series []*TestSeries) {
	idxs := make([]*tsdb.DatabaseIndex, 0, b.N)
	for i := 0; i < b.N; i++ {
		idxs = append(idxs, tsdb.NewDatabaseIndex(fmt.Sprintf("db%d", i)))
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		idx := idxs[n]
		for _, s := range series {
			idx.CreateSeriesIndexIfNotExists(s.Measurement, s.Series)
		}
	}
}

type TestSeries struct {
	Measurement string
	Series      *tsdb.Series
}

func genTestSeries(mCnt, tCnt, vCnt int) []*TestSeries {
	measurements := genStrList("measurement", mCnt)
	tagSets := NewTagSetGenerator(tCnt, vCnt).AllSets()
	series := []*TestSeries{}
	for _, m := range measurements {
		for _, ts := range tagSets {
			series = append(series, &TestSeries{
				Measurement: m,
				Series:      tsdb.NewSeries(fmt.Sprintf("%s:%s", m, string(tsdb.MarshalTags(ts))), models.NewTags(ts)),
			})
		}
	}
	return series
}

type TagValGenerator struct {
	Key  string
	Vals []string
	idx  int
}

func NewTagValGenerator(tagKey string, nVals int) *TagValGenerator {
	tvg := &TagValGenerator{Key: tagKey}
	for i := 0; i < nVals; i++ {
		tvg.Vals = append(tvg.Vals, fmt.Sprintf("tagValue%d", i))
	}
	return tvg
}

func (tvg *TagValGenerator) First() string {
	tvg.idx = 0
	return tvg.Curr()
}

func (tvg *TagValGenerator) Curr() string {
	return tvg.Vals[tvg.idx]
}

func (tvg *TagValGenerator) Next() string {
	tvg.idx++
	if tvg.idx >= len(tvg.Vals) {
		tvg.idx--
		return ""
	}
	return tvg.Curr()
}

type TagSet map[string]string

type TagSetGenerator struct {
	TagVals []*TagValGenerator
}

func NewTagSetGenerator(nSets int, nTagVals ...int) *TagSetGenerator {
	tsg := &TagSetGenerator{}
	for i := 0; i < nSets; i++ {
		nVals := nTagVals[0]
		if i < len(nTagVals) {
			nVals = nTagVals[i]
		}
		tagKey := fmt.Sprintf("tagKey%d", i)
		tsg.TagVals = append(tsg.TagVals, NewTagValGenerator(tagKey, nVals))
	}
	return tsg
}

func (tsg *TagSetGenerator) First() TagSet {
	for _, tsv := range tsg.TagVals {
		tsv.First()
	}
	return tsg.Curr()
}

func (tsg *TagSetGenerator) Curr() TagSet {
	ts := TagSet{}
	for _, tvg := range tsg.TagVals {
		ts[tvg.Key] = tvg.Curr()
	}
	return ts
}

func (tsg *TagSetGenerator) Next() TagSet {
	val := ""
	for _, tsv := range tsg.TagVals {
		if val = tsv.Next(); val != "" {
			break
		} else {
			tsv.First()
		}
	}

	if val == "" {
		return nil
	}

	return tsg.Curr()
}

func (tsg *TagSetGenerator) AllSets() []TagSet {
	allSets := []TagSet{}
	for ts := tsg.First(); ts != nil; ts = tsg.Next() {
		allSets = append(allSets, ts)
	}
	return allSets
}

func genStrList(prefix string, n int) []string {
	lst := make([]string, 0, n)
	for i := 0; i < n; i++ {
		lst = append(lst, fmt.Sprintf("%s%d", prefix, i))
	}
	return lst
}

// MustParseExpr parses an expression string and returns its AST representation.
func MustParseExpr(s string) influxql.Expr {
	expr, err := influxql.ParseExpr(s)
	if err != nil {
		panic(err.Error())
	}
	return expr
}

func strref(s string) *string {
	return &s
}
