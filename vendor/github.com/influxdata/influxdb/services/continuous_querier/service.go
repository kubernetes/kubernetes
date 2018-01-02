package continuous_querier // import "github.com/influxdata/influxdb/services/continuous_querier"

import (
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/models"
	"github.com/influxdata/influxdb/services/meta"
)

const (
	// NoChunkingSize specifies when not to chunk results. When planning
	// a select statement, passing zero tells it not to chunk results.
	// Only applies to raw queries.
	NoChunkingSize = 0

	// idDelimiter is used as a delimiter when creating a unique name for a
	// Continuous Query.
	idDelimiter = string(rune(31)) // unit separator
)

// Statistics for the CQ service.
const (
	statQueryOK   = "queryOk"
	statQueryFail = "queryFail"
)

// ContinuousQuerier represents a service that executes continuous queries.
type ContinuousQuerier interface {
	// Run executes the named query in the named database.  Blank database or name matches all.
	Run(database, name string, t time.Time) error
}

// metaClient is an internal interface to make testing easier.
type metaClient interface {
	AcquireLease(name string) (l *meta.Lease, err error)
	Databases() []meta.DatabaseInfo
	Database(name string) *meta.DatabaseInfo
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
	MetaClient    metaClient
	QueryExecutor *influxql.QueryExecutor
	Config        *Config
	RunInterval   time.Duration
	// RunCh can be used by clients to signal service to run CQs.
	RunCh          chan *RunRequest
	Logger         *log.Logger
	loggingEnabled bool
	stats          *Statistics
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
		RunInterval:    time.Duration(c.RunInterval),
		RunCh:          make(chan *RunRequest),
		loggingEnabled: c.LogEnabled,
		Logger:         log.New(os.Stderr, "[continuous_querier] ", log.LstdFlags),
		stats:          &Statistics{},
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

	assert(s.MetaClient != nil, "MetaClient is nil")
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

// SetLogOutput sets the writer to which all logs are written. It must not be
// called after Open is called.
func (s *Service) SetLogOutput(w io.Writer) {
	s.Logger = log.New(w, "[continuous_querier] ", log.LstdFlags)
}

// Statistics maintains the statistics for the continuous query service.
type Statistics struct {
	QueryOK   int64
	QueryFail int64
}

// Statistics returns statistics for periodic monitoring.
func (s *Service) Statistics(tags map[string]string) []models.Statistic {
	return []models.Statistic{{
		Name: "cq",
		Tags: tags,
		Values: map[string]interface{}{
			statQueryOK:   atomic.LoadInt64(&s.stats.QueryOK),
			statQueryFail: atomic.LoadInt64(&s.stats.QueryFail),
		},
	}}
}

// Run runs the specified continuous query, or all CQs if none is specified.
func (s *Service) Run(database, name string, t time.Time) error {
	var dbs []meta.DatabaseInfo

	if database != "" {
		// Find the requested database.
		db := s.MetaClient.Database(database)
		if db == nil {
			return influxql.ErrDatabaseNotFound(database)
		}
		dbs = append(dbs, *db)
	} else {
		// Get all databases.
		dbs = s.MetaClient.Databases()
	}

	// Loop through databases.
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, db := range dbs {
		// Loop through CQs in each DB executing the ones that match name.
		for _, cq := range db.ContinuousQueries {
			if name == "" || cq.Name == name {
				// Remove the last run time for the CQ
				id := fmt.Sprintf("%s%s%s", db.Name, idDelimiter, cq.Name)
				if _, ok := s.lastRuns[id]; ok {
					delete(s.lastRuns, id)
				}
			}
		}
	}

	// Signal the background routine to run CQs.
	s.RunCh <- &RunRequest{Now: t}

	return nil
}

// backgroundLoop runs on a go routine and periodically executes CQs.
func (s *Service) backgroundLoop() {
	leaseName := "continuous_querier"
	defer s.wg.Done()
	for {
		select {
		case <-s.stop:
			s.Logger.Println("continuous query service terminating")
			return
		case req := <-s.RunCh:
			if !s.hasContinuousQueries() {
				continue
			}
			if _, err := s.MetaClient.AcquireLease(leaseName); err == nil {
				s.Logger.Printf("running continuous queries by request for time: %v", req.Now)
				s.runContinuousQueries(req)
			}
		case <-time.After(s.RunInterval):
			if !s.hasContinuousQueries() {
				continue
			}
			if _, err := s.MetaClient.AcquireLease(leaseName); err == nil {
				s.runContinuousQueries(&RunRequest{Now: time.Now()})
			}
		}
	}
}

// hasContinuousQueries returns true if any CQs exist.
func (s *Service) hasContinuousQueries() bool {
	// Get list of all databases.
	dbs := s.MetaClient.Databases()
	// Loop through all databases executing CQs.
	for _, db := range dbs {
		if len(db.ContinuousQueries) > 0 {
			return true
		}
	}
	return false
}

// runContinuousQueries gets CQs from the meta store and runs them.
func (s *Service) runContinuousQueries(req *RunRequest) {
	// Get list of all databases.
	dbs := s.MetaClient.Databases()
	// Loop through all databases executing CQs.
	for _, db := range dbs {
		// TODO: distribute across nodes
		for _, cq := range db.ContinuousQueries {
			if !req.matches(&cq) {
				continue
			}
			if err := s.ExecuteContinuousQuery(&db, &cq, req.Now); err != nil {
				s.Logger.Printf("error executing query: %s: err = %s", cq.Query, err)
				atomic.AddInt64(&s.stats.QueryFail, 1)
			} else {
				atomic.AddInt64(&s.stats.QueryOK, 1)
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
	id := fmt.Sprintf("%s%s%s", dbi.Name, idDelimiter, cqi.Name)
	cq.LastRun, cq.HasRun = s.lastRuns[id]

	// Set the retention policy to default if it wasn't specified in the query.
	if cq.intoRP() == "" {
		cq.setIntoRP(dbi.DefaultRetentionPolicy)
	}

	// See if this query needs to be run.
	run, nextRun, err := cq.shouldRunContinuousQuery(now)
	if err != nil {
		return err
	} else if !run {
		return nil
	}

	// Get the group by interval.
	interval, err := cq.q.GroupByInterval()
	if err != nil {
		return err
	} else if interval == 0 {
		return nil
	}

	// Get the group by offset.
	offset, err := cq.q.GroupByOffset()
	if err != nil {
		return err
	}

	resampleEvery := interval
	if cq.Resample.Every != 0 {
		resampleEvery = cq.Resample.Every
	}

	// We're about to run the query so store the current time closest to the nearest interval.
	// If all is going well, this time should be the same as nextRun.
	cq.LastRun = now.Add(-offset).Truncate(resampleEvery).Add(offset)
	s.lastRuns[id] = cq.LastRun

	// Retrieve the oldest interval we should calculate based on the next time
	// interval. We do this instead of using the current time just in case any
	// time intervals were missed. The start time of the oldest interval is what
	// we use as the start time.
	resampleFor := interval
	if cq.Resample.For != 0 {
		resampleFor = cq.Resample.For
	} else if interval < resampleEvery {
		resampleFor = resampleEvery
	}

	// If the resample interval is greater than the interval of the query, use the
	// query interval instead.
	if interval < resampleEvery {
		resampleEvery = interval
	}

	// Calculate and set the time range for the query.
	startTime := nextRun.Add(interval - resampleFor - offset - 1).Truncate(interval).Add(offset)
	endTime := now.Add(interval - resampleEvery - offset).Truncate(interval).Add(offset)
	if !endTime.After(startTime) {
		// Exit early since there is no time interval.
		return nil
	}

	if err := cq.q.SetTimeRange(startTime, endTime); err != nil {
		s.Logger.Printf("error setting time range: %s\n", err)
		return err
	}

	var start time.Time
	if s.loggingEnabled {
		s.Logger.Printf("executing continuous query %s (%v to %v)", cq.Info.Name, startTime, endTime)
		start = time.Now()
	}

	// Do the actual processing of the query & writing of results.
	if err := s.runContinuousQueryAndWriteResult(cq); err != nil {
		s.Logger.Printf("error: %s. running: %s\n", err, cq.q.String())
		return err
	}

	if s.loggingEnabled {
		s.Logger.Printf("finished continuous query %s (%v to %v) in %s", cq.Info.Name, startTime, endTime, time.Now().Sub(start))
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
	ch := s.QueryExecutor.ExecuteQuery(q, influxql.ExecutionOptions{
		Database: cq.Database,
	}, closing)

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
	HasRun   bool
	LastRun  time.Time
	Resample ResampleOptions
	q        *influxql.SelectStatement
}

func (cq *ContinuousQuery) intoRP() string      { return cq.q.Target.Measurement.RetentionPolicy }
func (cq *ContinuousQuery) setIntoRP(rp string) { cq.q.Target.Measurement.RetentionPolicy = rp }

// Customizes the resampling intervals and duration of this continuous query.
type ResampleOptions struct {
	// The query will be resampled at this time interval. The first query will be
	// performed at this time interval. If this option is not given, the resample
	// interval is set to the group by interval.
	Every time.Duration

	// The query will continue being resampled for this time duration. If this
	// option is not given, the resample duration is the same as the group by
	// interval. A bucket's time is calculated based on the bucket's start time,
	// so a 40m resample duration with a group by interval of 10m will resample
	// the bucket 4 times (using the default time interval).
	For time.Duration
}

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
		Resample: ResampleOptions{
			Every: q.ResampleEvery,
			For:   q.ResampleFor,
		},
		q: q.Source,
	}

	return cquery, nil
}

// shouldRunContinuousQuery returns true if the CQ should be schedule to run. It will use the
// lastRunTime of the CQ and the rules for when to run set through the query to determine
// if this CQ should be run
func (cq *ContinuousQuery) shouldRunContinuousQuery(now time.Time) (bool, time.Time, error) {
	// if it's not aggregated we don't run it
	if cq.q.IsRawQuery {
		return false, cq.LastRun, errors.New("continuous queries must be aggregate queries")
	}

	// since it's aggregated we need to figure how often it should be run
	interval, err := cq.q.GroupByInterval()
	if err != nil {
		return false, cq.LastRun, err
	}

	// allow the interval to be overwritten by the query's resample options
	resampleEvery := interval
	if cq.Resample.Every != 0 {
		resampleEvery = cq.Resample.Every
	}

	// if we've passed the amount of time since the last run, or there was no last run, do it up
	if cq.HasRun {
		nextRun := cq.LastRun.Add(resampleEvery)
		if nextRun.UnixNano() <= now.UnixNano() {
			return true, nextRun, nil
		}
	} else {
		return true, now, nil
	}

	return false, cq.LastRun, nil
}

// assert will panic with a given formatted message if the given condition is false.
func assert(condition bool, msg string, v ...interface{}) {
	if !condition {
		panic(fmt.Sprintf("assert failed: "+msg, v...))
	}
}
