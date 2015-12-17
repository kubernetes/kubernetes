package continuous_querier

import (
	"errors"
	"expvar"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/tsdb"
)

const (
	// NoChunkingSize specifies when not to chunk results. When planning
	// a select statement, passing zero tells it not to chunk results.
	// Only applies to raw queries.
	NoChunkingSize = 0
)

// Statistics for the CQ service.
const (
	statQueryOK       = "queryOk"
	statQueryFail     = "queryFail"
	statPointsWritten = "pointsWritten"
)

// ContinuousQuerier represents a service that executes continuous queries.
type ContinuousQuerier interface {
	// Run executes the named query in the named database.  Blank database or name matches all.
	Run(database, name string, t time.Time) error
}

// queryExecutor is an internal interface to make testing easier.
type queryExecutor interface {
	ExecuteQuery(query *influxql.Query, database string, chunkSize int, closing chan struct{}) (<-chan *influxql.Result, error)
}

// metaStore is an internal interface to make testing easier.
type metaStore interface {
	IsLeader() bool
	Databases() ([]meta.DatabaseInfo, error)
	Database(name string) (*meta.DatabaseInfo, error)
}

// RunRequest is a request to run one or more CQs.
type RunRequest struct {
	// Now tells the CQ serivce what the current time is.
	Now time.Time
	// CQs tells the CQ service which queries to run.
	// If nil, all queries will be run.
	CQs []string
}

// matches returns true if the CQ matches one of the requested CQs.
func (rr *RunRequest) matches(cq *meta.ContinuousQueryInfo) bool {
	if rr.CQs == nil {
		return true
	}
	for _, q := range rr.CQs {
		if q == cq.Name {
			return true
		}
	}
	return false
}

// Service manages continuous query execution.
type Service struct {
	MetaStore     metaStore
	QueryExecutor queryExecutor
	Config        *Config
	RunInterval   time.Duration
	// RunCh can be used by clients to signal service to run CQs.
	RunCh          chan *RunRequest
	Logger         *log.Logger
	loggingEnabled bool
	statMap        *expvar.Map
	// lastRuns maps CQ name to last time it was run.
	mu       sync.RWMutex
	lastRuns map[string]time.Time
	stop     chan struct{}
	wg       *sync.WaitGroup
}

// NewService returns a new instance of Service.
func NewService(c Config) *Service {
	s := &Service{
		Config:         &c,
		RunInterval:    time.Second,
		RunCh:          make(chan *RunRequest),
		loggingEnabled: c.LogEnabled,
		statMap:        influxdb.NewStatistics("cq", "cq", nil),
		Logger:         log.New(os.Stderr, "[continuous_querier] ", log.LstdFlags),
		lastRuns:       map[string]time.Time{},
	}

	return s
}

// Open starts the service.
func (s *Service) Open() error {
	s.Logger.Println("Starting continuous query service")

	if s.stop != nil {
		return nil
	}

	assert(s.MetaStore != nil, "MetaStore is nil")
	assert(s.QueryExecutor != nil, "QueryExecutor is nil")

	s.stop = make(chan struct{})
	s.wg = &sync.WaitGroup{}
	s.wg.Add(1)
	go s.backgroundLoop()
	return nil
}

// Close stops the service.
func (s *Service) Close() error {
	if s.stop == nil {
		return nil
	}
	close(s.stop)
	s.wg.Wait()
	s.wg = nil
	s.stop = nil
	return nil
}

// SetLogger sets the internal logger to the logger passed in.
func (s *Service) SetLogger(l *log.Logger) {
	s.Logger = l
}

// Run runs the specified continuous query, or all CQs if none is specified.
func (s *Service) Run(database, name string, t time.Time) error {
	var dbs []meta.DatabaseInfo

	if database != "" {
		// Find the requested database.
		db, err := s.MetaStore.Database(database)
		if err != nil {
			return err
		} else if db == nil {
			return tsdb.ErrDatabaseNotFound(database)
		}
		dbs = append(dbs, *db)
	} else {
		// Get all databases.
		var err error
		dbs, err = s.MetaStore.Databases()
		if err != nil {
			return err
		}
	}

	// Loop through databases.
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, db := range dbs {
		// Loop through CQs in each DB executing the ones that match name.
		for _, cq := range db.ContinuousQueries {
			if name == "" || cq.Name == name {
				// Reset the last run time for the CQ.
				s.lastRuns[cq.Name] = time.Time{}
			}
		}
	}

	// Signal the background routine to run CQs.
	s.RunCh <- &RunRequest{Now: t}

	return nil
}

// backgroundLoop runs on a go routine and periodically executes CQs.
func (s *Service) backgroundLoop() {
	defer s.wg.Done()
	for {
		select {
		case <-s.stop:
			s.Logger.Println("continuous query service terminating")
			return
		case req := <-s.RunCh:
			if s.MetaStore.IsLeader() {
				s.Logger.Printf("running continuous queries by request for time: %v", req.Now.UnixNano())
				s.runContinuousQueries(req)
			}
		case <-time.After(s.RunInterval):
			if s.MetaStore.IsLeader() {
				s.runContinuousQueries(&RunRequest{Now: time.Now()})
			}
		}
	}
}

// runContinuousQueries gets CQs from the meta store and runs them.
func (s *Service) runContinuousQueries(req *RunRequest) {
	// Get list of all databases.
	dbs, err := s.MetaStore.Databases()
	if err != nil {
		s.Logger.Println("error getting databases")
		return
	}
	// Loop through all databases executing CQs.
	for _, db := range dbs {
		// TODO: distribute across nodes
		for _, cq := range db.ContinuousQueries {
			if !req.matches(&cq) {
				continue
			}
			if err := s.ExecuteContinuousQuery(&db, &cq, req.Now); err != nil {
				s.Logger.Printf("error executing query: %s: err = %s", cq.Query, err)
				s.statMap.Add(statQueryFail, 1)
			} else {
				s.statMap.Add(statQueryOK, 1)
			}
		}
	}
}

// ExecuteContinuousQuery executes a single CQ.
func (s *Service) ExecuteContinuousQuery(dbi *meta.DatabaseInfo, cqi *meta.ContinuousQueryInfo, now time.Time) error {
	// TODO: re-enable stats
	//s.stats.Inc("continuousQueryExecuted")

	// Local wrapper / helper.
	cq, err := NewContinuousQuery(dbi.Name, cqi)
	if err != nil {
		return err
	}

	// Get the last time this CQ was run from the service's cache.
	s.mu.Lock()
	defer s.mu.Unlock()
	cq.LastRun = s.lastRuns[cqi.Name]

	// Set the retention policy to default if it wasn't specified in the query.
	if cq.intoRP() == "" {
		cq.setIntoRP(dbi.DefaultRetentionPolicy)
	}

	// See if this query needs to be run.
	computeNoMoreThan := time.Duration(s.Config.ComputeNoMoreThan)
	run, err := cq.shouldRunContinuousQuery(s.Config.ComputeRunsPerInterval, computeNoMoreThan)
	if err != nil {
		return err
	} else if !run {
		return nil
	}

	// We're about to run the query so store the time.
	lastRun := time.Now()
	cq.LastRun = lastRun
	s.lastRuns[cqi.Name] = lastRun

	// Get the group by interval.
	interval, err := cq.q.GroupByInterval()
	if err != nil {
		return err
	} else if interval == 0 {
		return nil
	}

	// Calculate and set the time range for the query.
	startTime := now.Round(interval)
	if startTime.UnixNano() > now.UnixNano() {
		startTime = startTime.Add(-interval)
	}

	if err := cq.q.SetTimeRange(startTime, startTime.Add(interval)); err != nil {
		s.Logger.Printf("error setting time range: %s\n", err)
	}

	if s.loggingEnabled {
		s.Logger.Printf("executing continuous query %s", cq.Info.Name)
	}

	// Do the actual processing of the query & writing of results.
	if err := s.runContinuousQueryAndWriteResult(cq); err != nil {
		s.Logger.Printf("error: %s. running: %s\n", err, cq.q.String())
		return err
	}

	recomputeNoOlderThan := time.Duration(s.Config.RecomputeNoOlderThan)

	for i := 0; i < s.Config.RecomputePreviousN; i++ {
		// if we're already more time past the previous window than we're going to look back, stop
		if now.Sub(startTime) > recomputeNoOlderThan {
			return nil
		}
		newStartTime := startTime.Add(-interval)

		if err := cq.q.SetTimeRange(newStartTime, startTime); err != nil {
			s.Logger.Printf("error setting time range: %s\n", err)
			return err
		}

		if err := s.runContinuousQueryAndWriteResult(cq); err != nil {
			s.Logger.Printf("error during recompute previous: %s. running: %s\n", err, cq.q.String())
			return err
		}

		startTime = newStartTime
	}
	return nil
}

// runContinuousQueryAndWriteResult will run the query against the cluster and write the results back in
func (s *Service) runContinuousQueryAndWriteResult(cq *ContinuousQuery) error {
	// Wrap the CQ's inner SELECT statement in a Query for the QueryExecutor.
	q := &influxql.Query{
		Statements: influxql.Statements([]influxql.Statement{cq.q}),
	}

	closing := make(chan struct{})
	defer close(closing)

	// Execute the SELECT.
	ch, err := s.QueryExecutor.ExecuteQuery(q, cq.Database, NoChunkingSize, closing)
	if err != nil {
		return err
	}
	// There is only one statement, so we will only ever receive one result
	res, ok := <-ch
	if !ok {
		panic("result channel was closed")
	}
	if res.Err != nil {
		return res.Err
	}
	return nil
}

// ContinuousQuery is a local wrapper / helper around continuous queries.
type ContinuousQuery struct {
	Database string
	Info     *meta.ContinuousQueryInfo
	LastRun  time.Time
	q        *influxql.SelectStatement
}

func (cq *ContinuousQuery) intoRP() string      { return cq.q.Target.Measurement.RetentionPolicy }
func (cq *ContinuousQuery) setIntoRP(rp string) { cq.q.Target.Measurement.RetentionPolicy = rp }

// NewContinuousQuery returns a ContinuousQuery object with a parsed influxql.CreateContinuousQueryStatement
func NewContinuousQuery(database string, cqi *meta.ContinuousQueryInfo) (*ContinuousQuery, error) {
	stmt, err := influxql.NewParser(strings.NewReader(cqi.Query)).ParseStatement()
	if err != nil {
		return nil, err
	}

	q, ok := stmt.(*influxql.CreateContinuousQueryStatement)
	if !ok || q.Source.Target == nil || q.Source.Target.Measurement == nil {
		return nil, errors.New("query isn't a valid continuous query")
	}

	cquery := &ContinuousQuery{
		Database: database,
		Info:     cqi,
		q:        q.Source,
	}

	return cquery, nil
}

// shouldRunContinuousQuery returns true if the CQ should be schedule to run. It will use the
// lastRunTime of the CQ and the rules for when to run set through the config to determine
// if this CQ should be run
func (cq *ContinuousQuery) shouldRunContinuousQuery(runsPerInterval int, noMoreThan time.Duration) (bool, error) {
	// if it's not aggregated we don't run it
	if cq.q.IsRawQuery {
		return false, errors.New("continuous queries must be aggregate queries")
	}

	// since it's aggregated we need to figure how often it should be run
	interval, err := cq.q.GroupByInterval()
	if err != nil {
		return false, err
	}

	// determine how often we should run this continuous query.
	// group by time / the number of times to compute
	computeEvery := time.Duration(interval.Nanoseconds()/int64(runsPerInterval)) * time.Nanosecond
	// make sure we're running no more frequently than the setting in the config
	if computeEvery < noMoreThan {
		computeEvery = noMoreThan
	}

	// if we've passed the amount of time since the last run, do it up
	if cq.LastRun.Add(computeEvery).UnixNano() <= time.Now().UnixNano() {
		return true, nil
	}

	return false, nil
}

// assert will panic with a given formatted message if the given condition is false.
func assert(condition bool, msg string, v ...interface{}) {
	if !condition {
		panic(fmt.Sprintf("assert failed: "+msg, v...))
	}
}
