package stressClient

import (
	"fmt"
	"log"
	"sync"

	influx "github.com/influxdata/influxdb/client/v2"
)

// NewStressTest creates the backend for the stress test
func NewStressTest() *StressTest {

	packageCh := make(chan Package, 0)
	directiveCh := make(chan Directive, 0)
	responseCh := make(chan Response, 0)

	clnt, _ := influx.NewHTTPClient(influx.HTTPConfig{
		Addr: fmt.Sprintf("http://%v/", "localhost:8086"),
	})

	s := &StressTest{
		TestDB:    "_stressTest",
		Precision: "s",
		StartDate: "2016-01-02",
		BatchSize: 5000,

		packageChan:   packageCh,
		directiveChan: directiveCh,

		ResultsClient: clnt,
		ResultsChan:   responseCh,
		communes:      make(map[string]*commune),
		TestID:        randStr(10),
	}

	// Start the client service
	startStressClient(packageCh, directiveCh, responseCh, s.TestID)

	// Listen for Results coming in
	s.resultsListen()

	return s
}

// NewTestStressTest returns a StressTest to be used for testing Statements
func NewTestStressTest() (*StressTest, chan Package, chan Directive) {

	packageCh := make(chan Package, 0)
	directiveCh := make(chan Directive, 0)

	s := &StressTest{
		TestDB:    "_stressTest",
		Precision: "s",
		StartDate: "2016-01-02",
		BatchSize: 5000,

		directiveChan: directiveCh,
		packageChan:   packageCh,

		communes: make(map[string]*commune),
		TestID:   randStr(10),
	}

	return s, packageCh, directiveCh
}

// The StressTest is the Statement facing API that consumes Statement output and coordinates the test results
type StressTest struct {
	TestID string
	TestDB string

	Precision string
	StartDate string
	BatchSize int

	sync.WaitGroup
	sync.Mutex

	packageChan   chan<- Package
	directiveChan chan<- Directive

	ResultsChan   chan Response
	communes      map[string]*commune
	ResultsClient influx.Client
}

// SendPackage is the public facing API for to send Queries and Points
func (st *StressTest) SendPackage(p Package) {
	st.packageChan <- p
}

// SendDirective is the public facing API to set state variables in the test
func (st *StressTest) SendDirective(d Directive) {
	st.directiveChan <- d
}

// Starts a go routine that listens for Results
func (st *StressTest) resultsListen() {
	st.createDatabase(st.TestDB)
	go func() {
		bp := st.NewResultsPointBatch()
		for resp := range st.ResultsChan {
			switch resp.Point.Name() {
			case "done":
				st.ResultsClient.Write(bp)
				resp.Tracer.Done()
			default:
				// Add the StressTest tags
				pt := resp.AddTags(st.tags())
				// Add the point to the batch
				bp = st.batcher(pt, bp)
				resp.Tracer.Done()
			}
		}
	}()
}

// NewResultsPointBatch creates a new batch of points for the results
func (st *StressTest) NewResultsPointBatch() influx.BatchPoints {
	bp, _ := influx.NewBatchPoints(influx.BatchPointsConfig{
		Database:  st.TestDB,
		Precision: "ns",
	})
	return bp
}

// Batches incoming Result.Point and sends them if the batch reaches 5k in size
func (st *StressTest) batcher(pt *influx.Point, bp influx.BatchPoints) influx.BatchPoints {
	if len(bp.Points()) <= 5000 {
		bp.AddPoint(pt)
	} else {
		err := st.ResultsClient.Write(bp)
		if err != nil {
			log.Fatalf("Error writing performance stats\n  error: %v\n", err)
		}
		bp = st.NewResultsPointBatch()
	}
	return bp
}

// Convinence database creation function
func (st *StressTest) createDatabase(db string) {
	query := fmt.Sprintf("CREATE DATABASE %v", db)
	res, err := st.ResultsClient.Query(influx.Query{Command: query})
	if err != nil {
		log.Fatalf("error: no running influx server at localhost:8086")
		if res.Error() != nil {
			log.Fatalf("error: no running influx server at localhost:8086")
		}
	}
}

// GetStatementResults is a convinence function for fetching all results given a StatementID
func (st *StressTest) GetStatementResults(sID, t string) (res []influx.Result) {
	qryStr := fmt.Sprintf(`SELECT * FROM "%v" WHERE statement_id = '%v'`, t, sID)
	return st.queryTestResults(qryStr)
}

//  Runs given qry on the test results database and returns the results or nil in case of error
func (st *StressTest) queryTestResults(qry string) (res []influx.Result) {
	response, err := st.ResultsClient.Query(influx.Query{Command: qry, Database: st.TestDB})
	if err == nil {
		if response.Error() != nil {
			log.Fatalf("Error sending results query\n  error: %v\n", response.Error())
		}
	}
	if response.Results[0].Series == nil {
		return nil
	}
	return response.Results
}
