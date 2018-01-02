package statement

import (
	"log"
	"time"

	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

// InfluxqlStatement is a Statement Implementation that allows statements that parse in InfluxQL to be passed directly to the target instance
type InfluxqlStatement struct {
	StatementID string
	Query       string
	Tracer      *stressClient.Tracer
}

func (i *InfluxqlStatement) tags() map[string]string {
	tags := make(map[string]string)
	return tags
}

// SetID statisfies the Statement Interface
func (i *InfluxqlStatement) SetID(s string) {
	i.StatementID = s
}

// Run statisfies the Statement Interface
func (i *InfluxqlStatement) Run(s *stressClient.StressTest) {

	// Set the tracer
	i.Tracer = stressClient.NewTracer(i.tags())

	// Make the Package
	p := stressClient.NewPackage(stressClient.Query, []byte(i.Query), i.StatementID, i.Tracer)

	// Increment the tracer
	i.Tracer.Add(1)

	// Send the Package
	s.SendPackage(p)

	// Wait for all operations to finish
	i.Tracer.Wait()
}

// Report statisfies the Statement Interface
// No test coverage, fix
func (i *InfluxqlStatement) Report(s *stressClient.StressTest) (out string) {
	allData := s.GetStatementResults(i.StatementID, "query")

	iqlr := &influxQlReport{
		statement: i.Query,
		columns:   allData[0].Series[0].Columns,
		values:    allData[0].Series[0].Values,
	}

	iqlr.responseTime = time.Duration(responseTimes(iqlr.columns, iqlr.values)[0].Value)

	switch countSuccesses(iqlr.columns, iqlr.values) {
	case 0:
		iqlr.success = false
	case 1:
		iqlr.success = true
	default:
		log.Fatal("Error fetching response for InfluxQL statement")
	}

	return iqlr.String()
}
