package statement

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestInsertReportString(t *testing.T) {
	ir := newTestInsertReport()
	tmplString := `Write Statement:                    %v
  Points/Sec:                          %v
  Resp Time Average:                   %v
  Resp Time Standard Deviation:        %v
  95th Percentile Write Response:      %v
  Average Request Bytes:               %v
  Successful Write Reqs:               %v
  Retries:                             %v`
	expected := fmt.Sprintf(tmplString,
		ir.name,
		ir.pointsPerSecond,
		ir.avgResponseTime,
		ir.stdDevResponseTime,
		ir.percentile,
		ir.avgRequestBytes,
		ir.successfulWrites,
		ir.numRetries)
	got := ir.String()
	if expected != got {
		t.Fail()
	}
}

func TestInsertReportPoint(t *testing.T) {
	ir := newTestInsertReport()
	expected := "testDefault"
	got := strings.Split(ir.Point().String(), " ")[0]
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func TestQueryReportString(t *testing.T) {
	qr := newTestQueryReport()
	tmplString := `Query Statement:                    %v
  Resp Time Average:                   %v
  Resp Time Standard Deviation:        %v
  95th Percentile Read Response:       %v
  Query Resp Bytes Average:            %v bytes
  Successful Queries:                  %v`
	expected := fmt.Sprintf(tmplString,
		qr.name,
		qr.avgResponseTime,
		qr.stdDevResponseTime,
		qr.percentile,
		qr.responseBytes,
		qr.successfulReads)
	got := qr.String()
	if expected != got {
		t.Fail()
	}
}

func TestQueryReportPoint(t *testing.T) {
	qr := newTestQueryReport()
	expected := "testDefault"
	got := strings.Split(qr.Point().String(), " ")[0]
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func TestInfluxQLReportString(t *testing.T) {
	iqlr := newTestInfluxQLReport()
	expected := fmt.Sprintf("[X] '%v' -> %v", iqlr.statement, iqlr.responseTime)
	got := iqlr.String()
	if expected != got {
		t.Fail()
	}
}

func TestInfluxQLReportPoint(t *testing.T) {
	iqlr := newTestInfluxQLReport()
	expected := "testDefault"
	got := strings.Split(iqlr.Point().String(), " ")[0]
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func newTestInsertReport() *insertReport {
	return &insertReport{
		name:               "foo_name",
		numRetries:         0,
		pointsPerSecond:    500000,
		successfulWrites:   20000,
		avgRequestBytes:    18932,
		avgResponseTime:    time.Duration(int64(20000)),
		stdDevResponseTime: time.Duration(int64(20000)),
		percentile:         time.Duration(int64(20000)),
	}
}

func newTestQueryReport() *queryReport {
	return &queryReport{
		name:                "foo_name",
		successfulReads:     2000,
		responseBytes:       39049,
		stddevResponseBytes: 9091284,
		avgResponseTime:     139082,
		stdDevResponseTime:  29487,
		percentile:          8273491,
	}
}

func newTestInfluxQLReport() *influxQlReport {
	return &influxQlReport{
		statement:    "foo_name",
		responseTime: time.Duration(int64(20000)),
		success:      false,
	}
}

func TestGetColumnIndex(t *testing.T) {
	col := "thing"
	columns := []string{"thing"}
	expected := 0
	got := getColumnIndex(col, columns)
	if expected != got {
		t.Fail()
	}
}

func TestNumberBytes(t *testing.T) {
	columns := []string{"num_bytes"}
	values := [][]interface{}{[]interface{}{json.Number("1")}}
	expected := 1
	got := numberBytes(columns, values)
	if expected != got {
		t.Fail()
	}
}

func TestCountSuccesses(t *testing.T) {
	columns := []string{"status_code"}
	values := [][]interface{}{[]interface{}{json.Number("200")}}
	expected := 1
	got := countSuccesses(columns, values)
	if expected != got {
		t.Fail()
	}
}

func TestCountRetries(t *testing.T) {
	columns := []string{"status_code"}
	values := [][]interface{}{[]interface{}{json.Number("500")}}
	expected := 1
	got := countRetries(columns, values)
	if expected != got {
		t.Fail()
	}
}

func TestResponseTimes(t *testing.T) {
	columns := []string{"response_time_ns"}
	values := [][]interface{}{[]interface{}{json.Number("380")}}
	expected := ResponseTimes([]ResponseTime{NewResponseTime(380)})
	got := responseTimes(columns, values)
	if expected[0].Value != got[0].Value {
		t.Fail()
	}
}

func TestPercentile(t *testing.T) {
	rs := createTestResponseTimes()
	expected := time.Duration(21)
	got := percentile(rs)
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func TestAvgDuration(t *testing.T) {
	rs := createTestResponseTimes()
	expected := time.Duration(11)
	got := avgDuration(rs)
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func TestStddevDuration(t *testing.T) {
	rs := createTestResponseTimes()
	expected := time.Duration(6)
	got := stddevDuration(rs)
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func createTestResponseTimes() ResponseTimes {
	rstms := []int{1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 8, 9, 10, 11, 12, 20, 21, 22}
	rs := []ResponseTime{}
	for _, rst := range rstms {
		rs = append(rs, NewResponseTime(rst))
	}
	return rs
}
