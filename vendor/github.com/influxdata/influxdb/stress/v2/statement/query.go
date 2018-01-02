package statement

import (
	"fmt"
	"log"
	"time"

	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

// QueryStatement is a Statement Implementation to run queries on the target InfluxDB instance
type QueryStatement struct {
	StatementID string
	Name        string

	// TemplateString is a query template that can be filled in by Args
	TemplateString string
	Args           []string

	// Number of queries to run
	Count int

	// Tracer for tracking returns
	Tracer *stressClient.Tracer

	// track time for all queries
	runtime time.Duration
}

// This function adds tags to the recording points
func (i *QueryStatement) tags() map[string]string {
	tags := make(map[string]string)
	return tags
}

// SetID statisfies the Statement Interface
func (i *QueryStatement) SetID(s string) {
	i.StatementID = s
}

// Run statisfies the Statement Interface
func (i *QueryStatement) Run(s *stressClient.StressTest) {

	i.Tracer = stressClient.NewTracer(i.tags())

	vals := make(map[string]interface{})

	var point models.Point

	runtime := time.Now()

	for j := 0; j < i.Count; j++ {

		// If the query is a simple query, send it.
		if len(i.Args) == 0 {
			b := []byte(i.TemplateString)

			// Make the package
			p := stressClient.NewPackage(stressClient.Query, b, i.StatementID, i.Tracer)

			// Increment the tracer
			i.Tracer.Add(1)

			// Send the package
			s.SendPackage(p)

		} else {
			// Otherwise cherry pick field values from the commune?

			// TODO: Currently the program lock up here if s.GetPoint
			//       cannot return a value, which can happen.
			// See insert.go
			s.Lock()
			point = s.GetPoint(i.Name, s.Precision)
			s.Unlock()

			setMapValues(vals, point)

			// Set the template string with args from the commune
			b := []byte(fmt.Sprintf(i.TemplateString, setArgs(vals, i.Args)...))

			// Make the package
			p := stressClient.NewPackage(stressClient.Query, b, i.StatementID, i.Tracer)

			// Increment the tracer
			i.Tracer.Add(1)

			// Send the package
			s.SendPackage(p)

		}
	}

	// Wait for all operations to finish
	i.Tracer.Wait()

	// Stop time timer
	i.runtime = time.Since(runtime)
}

// Report statisfies the Statement Interface
func (i *QueryStatement) Report(s *stressClient.StressTest) string {
	// Pull data via StressTest client
	allData := s.GetStatementResults(i.StatementID, "query")

	if len(allData) == 0 || allData[0].Series == nil {
		log.Fatalf("No data returned for query report\n  Statement Name: %v\n  Statement ID: %v\n", i.Name, i.StatementID)
	}

	qr := &queryReport{
		name:    i.Name,
		columns: allData[0].Series[0].Columns,
		values:  allData[0].Series[0].Values,
	}

	responseTimes := responseTimes(qr.columns, qr.values)

	qr.percentile = percentile(responseTimes)
	qr.avgResponseTime = avgDuration(responseTimes)
	qr.stdDevResponseTime = stddevDuration(responseTimes)
	qr.successfulReads = countSuccesses(qr.columns, qr.values)
	qr.responseBytes = numberBytes(qr.columns, qr.values)

	return qr.String()
}

func getRandomTagPair(m models.Tags) string {
	for k, v := range m {
		return fmt.Sprintf("%v='%v'", k, v)
	}

	return ""
}

func getRandomFieldKey(m map[string]interface{}) string {
	for k := range m {
		return fmt.Sprintf("%v", k)
	}

	return ""
}

func setMapValues(m map[string]interface{}, p models.Point) {
	m["%f"] = getRandomFieldKey(p.Fields())
	m["%m"] = p.Name()
	m["%t"] = getRandomTagPair(p.Tags())
	m["%a"] = p.UnixNano()
}

func setArgs(m map[string]interface{}, args []string) []interface{} {
	values := make([]interface{}, len(args))
	for i, arg := range args {
		values[i] = m[arg]
	}
	return values
}
