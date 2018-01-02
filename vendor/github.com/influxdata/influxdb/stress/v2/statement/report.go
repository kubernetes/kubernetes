package statement

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sort"
	"time"

	influx "github.com/influxdata/influxdb/client/v2"
)

// TODO: Refactor this file to utilize a common interface
// This will make adding new reports easier in the future

// Runs performance numbers for insert statements
type insertReport struct {
	name               string
	numRetries         int
	pointsPerSecond    int
	successfulWrites   int
	avgRequestBytes    int
	avgResponseTime    time.Duration
	stdDevResponseTime time.Duration
	percentile         time.Duration

	columns []string
	values  [][]interface{}
}

// Returns the version of the report that is output to STDOUT
func (ir *insertReport) String() string {
	tmplString := `Write Statement:                    %v
  Points/Sec:                          %v
  Resp Time Average:                   %v
  Resp Time Standard Deviation:        %v
  95th Percentile Write Response:      %v
  Average Request Bytes:               %v
  Successful Write Reqs:               %v
  Retries:                             %v`

	return fmt.Sprintf(tmplString,
		ir.name,
		ir.pointsPerSecond,
		ir.avgResponseTime,
		ir.stdDevResponseTime,
		ir.percentile,
		ir.avgRequestBytes,
		ir.successfulWrites,
		ir.numRetries)
}

// Returns a point representation of the report to be written to the ResultsDB
func (ir *insertReport) Point() *influx.Point {
	measurement := "testDefault"
	tags := map[string]string{}
	fields := map[string]interface{}{"field": "blank"}
	point, err := influx.NewPoint(measurement, tags, fields, time.Now())
	if err != nil {
		log.Fatalf("Error creating insertReport point\n  measurement: %v\n  tags: %v\n  fields: %v\n  error: %v\n", measurement, tags, fields, err)
	}
	return point
}

// Runs performance numbers for query statements
type queryReport struct {
	name                string
	successfulReads     int
	responseBytes       int
	stddevResponseBytes int
	avgResponseTime     time.Duration
	stdDevResponseTime  time.Duration
	percentile          time.Duration

	columns []string
	values  [][]interface{}
}

// Returns the version of the report that is output to STDOUT
func (qr *queryReport) String() string {
	tmplString := `Query Statement:                    %v
  Resp Time Average:                   %v
  Resp Time Standard Deviation:        %v
  95th Percentile Read Response:       %v
  Query Resp Bytes Average:            %v bytes
  Successful Queries:                  %v`

	return fmt.Sprintf(tmplString,
		qr.name,
		qr.avgResponseTime,
		qr.stdDevResponseTime,
		qr.percentile,
		qr.responseBytes,
		qr.successfulReads)
}

// Returns a point representation of the report to be written to the ResultsDB
func (qr *queryReport) Point() *influx.Point {
	measurement := "testDefault"
	tags := map[string]string{}
	fields := map[string]interface{}{"field": "blank"}
	point, err := influx.NewPoint(measurement, tags, fields, time.Now())
	if err != nil {
		log.Fatalf("Error creating queryReport point\n  measurement: %v\n  tags: %v\n  fields: %v\n  error: %v\n", measurement, tags, fields, err)
	}
	return point
}

// Runs performance numbers for InfluxQL statements
type influxQlReport struct {
	statement    string
	responseTime time.Duration
	success      bool

	columns []string
	values  [][]interface{}
}

// Returns the version of the report that is output to STDOUT
func (iqlr *influxQlReport) String() string {
	// Fancy format success
	var success string
	switch iqlr.success {
	case true:
		success = "[√]"
	case false:
		success = "[X]"
	}
	return fmt.Sprintf("%v '%v' -> %v", success, iqlr.statement, iqlr.responseTime)
}

// Returns a point representation of the report to be written to the ResultsDB
func (iqlr *influxQlReport) Point() *influx.Point {
	measurement := "testDefault"
	tags := map[string]string{}
	fields := map[string]interface{}{"field": "blank"}
	point, err := influx.NewPoint(measurement, tags, fields, time.Now())
	if err != nil {
		log.Fatalf("Error creating influxQL point\n  measurement: %v\n  tags: %v\n  fields: %v\n  error: %v\n", measurement, tags, fields, err)
	}
	return point
}

// Given a field or tag name this function returns the index where the values are found
func getColumnIndex(col string, columns []string) int {
	index := -1
	for i, column := range columns {
		if column == col {
			index = i
		}
	}
	return index
}

// Given a full set of results pulls the average num_bytes
func numberBytes(columns []string, values [][]interface{}) int {
	out := 0
	index := getColumnIndex("num_bytes", columns)
	for _, val := range values {
		reqBytes, err := val[index].(json.Number).Int64()
		if err != nil {
			log.Fatalf("Error coercing json.Number to Int64\n  json.Number:%v\n  error: %v\n", val[index], err)
		}
		out += int(reqBytes)
	}
	return out / len(values)
}

// Counts the number of 200(query) or 204(write) responses and returns them
func countSuccesses(columns []string, values [][]interface{}) (out int) {
	index := getColumnIndex("status_code", columns)
	for _, val := range values {
		status, err := val[index].(json.Number).Int64()
		if err != nil {
			log.Fatalf("Error coercing json.Number to Int64\n  json.Number:%v\n  error: %v\n", val[index], err)
		}
		if status == 204 || status == 200 {
			out++
		}
	}
	return out
}

// Counts number of 500 status codes
func countRetries(columns []string, values [][]interface{}) (out int) {
	index := getColumnIndex("status_code", columns)
	for _, val := range values {
		status, err := val[index].(json.Number).Int64()
		if err != nil {
			log.Fatalf("Error coercing json.Number to Int64\n  json.Number:%v\n  error: %v\n", val[index], err)
		}
		if status == 500 {
			out++
		}
	}
	return out
}

// Pulls out the response_time_ns values and formats them into ResponseTimes for reporting
func responseTimes(columns []string, values [][]interface{}) (rs ResponseTimes) {
	rs = make([]ResponseTime, 0)
	index := getColumnIndex("response_time_ns", columns)
	for _, val := range values {
		respTime, err := val[index].(json.Number).Int64()
		if err != nil {
			log.Fatalf("Error coercing json.Number to Int64\n  json.Number:%v\n  error: %v\n", val[index], err)
		}
		rs = append(rs, NewResponseTime(int(respTime)))
	}
	return rs
}

// Returns the 95th perecntile response time
func percentile(rs ResponseTimes) time.Duration {
	sort.Sort(rs)
	return time.Duration(rs[(len(rs) * 19 / 20)].Value)
}

// Returns the average response time
func avgDuration(rs ResponseTimes) (out time.Duration) {
	for _, t := range rs {
		out += time.Duration(t.Value)
	}
	return out / time.Duration(len(rs))
}

// Returns the standard deviation of a sample of response times
func stddevDuration(rs ResponseTimes) (out time.Duration) {
	avg := avgDuration(rs)

	for _, t := range rs {
		out += (avg - time.Duration(t.Value)) * (avg - time.Duration(t.Value))
	}

	return time.Duration(int64(math.Sqrt(float64(out) / float64(len(rs)))))
}
