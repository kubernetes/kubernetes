package statement

import (
	"bytes"
	"fmt"
	"log"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

// InsertStatement is a Statement Implementation that creates points to be written to the target InfluxDB instance
type InsertStatement struct {
	TestID      string
	StatementID string

	// Statement Name
	Name string

	// Template string for points. Filled by the output of stringers
	TemplateString string

	// TagCount is used to find the number of series in the dataset
	TagCount int

	// The Tracer prevents InsertStatement.Run() from returning early
	Tracer *stressClient.Tracer

	// Timestamp is #points to write and percision
	Timestamp *Timestamp

	// Templates turn into stringers
	Templates Templates
	stringers Stringers

	// Number of series in this insert Statement
	series int

	// Returns the proper time for the next point
	time func() int64

	// Concurrency utiliities
	sync.WaitGroup
	sync.Mutex

	// Timer for runtime and pps calculation
	runtime time.Duration
}

func (i *InsertStatement) tags() map[string]string {
	tags := map[string]string{
		"number_fields":       i.numFields(),
		"number_series":       fmtInt(i.series),
		"number_points_write": fmtInt(i.Timestamp.Count),
	}
	return tags
}

// SetID statisfies the Statement Interface
func (i *InsertStatement) SetID(s string) {
	i.StatementID = s
}

// SetVars sets up the environment for InsertStatement to call it's Run function
func (i *InsertStatement) SetVars(s *stressClient.StressTest) chan<- string {
	// Set the #series at 1 to start
	i.series = 1

	// Num series is the product of the cardinality of the tags
	for _, tmpl := range i.Templates[0:i.TagCount] {
		i.series *= tmpl.numSeries()
	}

	// make stringers from the templates
	i.stringers = i.Templates.Init(i.series)

	// Set the time function, keeps track of 'time' of the points being created
	i.time = i.Timestamp.Time(s.StartDate, i.series, s.Precision)

	// Set a commune on the StressTest
	s.Lock()
	comCh := s.SetCommune(i.Name)
	s.Unlock()

	// Set the tracer
	i.Tracer = stressClient.NewTracer(i.tags())

	return comCh
}

// Run statisfies the Statement Interface
func (i *InsertStatement) Run(s *stressClient.StressTest) {

	// Set variables on the InsertStatement and make the comCh
	comCh := i.SetVars(s)

	// TODO: Refactor to eleminate the ctr
	// Start the counter
	ctr := 0

	// Create the first bytes buffer
	buf := bytes.NewBuffer([]byte{})

	runtime := time.Now()

	for k := 0; k < i.Timestamp.Count; k++ {

		// Increment the counter. ctr == k + 1?
		ctr++

		// Make the point from the template string and the stringers
		point := fmt.Sprintf(i.TemplateString, i.stringers.Eval(i.time)...)

		// Add the string to the buffer
		buf.WriteString(point)
		// Add a newline char to seperate the points
		buf.WriteString("\n")

		// If len(batch) == batchSize then send it
		if ctr%s.BatchSize == 0 && ctr != 0 {
			b := buf.Bytes()
			// Trimming the trailing newline character
			b = b[0 : len(b)-1]

			// Create the package
			p := stressClient.NewPackage(stressClient.Write, b, i.StatementID, i.Tracer)

			// Use Tracer to wait for all operations to finish
			i.Tracer.Add(1)

			// Send the package
			s.SendPackage(p)

			// Reset the bytes Buffer
			temp := bytes.NewBuffer([]byte{})
			buf = temp
		}

		// TODO: Racy
		// Has to do with InsertStatement and QueryStatement communication
		if len(comCh) < cap(comCh) {
			select {
			case comCh <- point:
				break
			default:
				break
			}
		}

	}

	// If There are additional points remaining in the buffer send them before exiting
	if buf.Len() != 0 {
		b := buf.Bytes()
		// Trimming the trailing newline character
		b = b[0 : len(b)-1]

		// Create the package
		p := stressClient.NewPackage(stressClient.Write, b, i.StatementID, i.Tracer)

		// Use Tracer to wait for all operations to finish
		i.Tracer.Add(1)

		// Send the package
		s.SendPackage(p)
	}

	// Wait for all tracers to decrement
	i.Tracer.Wait()

	// Stop the timer
	i.runtime = time.Since(runtime)
}

// Report statisfies the Statement Interface
func (i *InsertStatement) Report(s *stressClient.StressTest) string {
	// Pull data via StressTest client
	allData := s.GetStatementResults(i.StatementID, "write")

	if allData == nil || allData[0].Series == nil {
		log.Fatalf("No data returned for write report\n  Statement Name: %v\n  Statement ID: %v\n", i.Name, i.StatementID)
	}

	ir := &insertReport{
		name:    i.Name,
		columns: allData[0].Series[0].Columns,
		values:  allData[0].Series[0].Values,
	}

	responseTimes := responseTimes(ir.columns, ir.values)

	ir.percentile = percentile(responseTimes)
	ir.avgResponseTime = avgDuration(responseTimes)
	ir.stdDevResponseTime = stddevDuration(responseTimes)
	ir.pointsPerSecond = int(float64(i.Timestamp.Count) / i.runtime.Seconds())
	ir.numRetries = countRetries(ir.columns, ir.values)
	ir.successfulWrites = countSuccesses(ir.columns, ir.values)
	ir.avgRequestBytes = numberBytes(ir.columns, ir.values)

	return ir.String()
}

func (i *InsertStatement) numFields() string {
	pt := strings.Split(i.TemplateString, " ")
	fields := strings.Split(pt[1], ",")
	return fmtInt(len(fields))
}

func fmtInt(i int) string {
	return strconv.FormatInt(int64(i), 10)
}
