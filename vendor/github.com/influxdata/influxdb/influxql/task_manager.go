package influxql

import (
	"fmt"
	"io/ioutil"
	"log"
	"sync"
	"time"

	"github.com/influxdata/influxdb/models"
)

const (
	// DefaultQueryTimeout is the default timeout for executing a query.
	// A value of zero will have no query timeout.
	DefaultQueryTimeout = time.Duration(0)
)

// TaskManager takes care of all aspects related to managing running queries.
type TaskManager struct {
	// Query execution timeout.
	QueryTimeout time.Duration

	// Log queries if they are slower than this time.
	// If zero, slow queries will never be logged.
	LogQueriesAfter time.Duration

	// Maximum number of concurrent queries.
	MaxConcurrentQueries int

	// Logger to use for all logging.
	// Defaults to discarding all log output.
	Logger *log.Logger

	// Used for managing and tracking running queries.
	queries  map[uint64]*QueryTask
	nextID   uint64
	mu       sync.RWMutex
	shutdown bool
}

// NewTaskManager creates a new TaskManager.
func NewTaskManager() *TaskManager {
	return &TaskManager{
		QueryTimeout: DefaultQueryTimeout,
		Logger:       log.New(ioutil.Discard, "[query] ", log.LstdFlags),
		queries:      make(map[uint64]*QueryTask),
		nextID:       1,
	}
}

// ExecuteStatement executes a statement containing one of the task management queries.
func (t *TaskManager) ExecuteStatement(stmt Statement, ctx ExecutionContext) error {
	switch stmt := stmt.(type) {
	case *ShowQueriesStatement:
		rows, err := t.executeShowQueriesStatement(stmt)
		if err != nil {
			return err
		}

		ctx.Results <- &Result{
			StatementID: ctx.StatementID,
			Series:      rows,
		}
	case *KillQueryStatement:
		var messages []*Message
		if ctx.ReadOnly {
			messages = append(messages, ReadOnlyWarning(stmt.String()))
		}

		if err := t.executeKillQueryStatement(stmt); err != nil {
			return err
		}
		ctx.Results <- &Result{
			StatementID: ctx.StatementID,
			Messages:    messages,
		}
	default:
		return ErrInvalidQuery
	}
	return nil
}

func (t *TaskManager) executeKillQueryStatement(stmt *KillQueryStatement) error {
	return t.KillQuery(stmt.QueryID)
}

func (t *TaskManager) executeShowQueriesStatement(q *ShowQueriesStatement) (models.Rows, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	now := time.Now()

	values := make([][]interface{}, 0, len(t.queries))
	for id, qi := range t.queries {
		d := now.Sub(qi.startTime)

		switch {
		case d >= time.Second:
			d = d - (d % time.Second)
		case d >= time.Millisecond:
			d = d - (d % time.Millisecond)
		case d >= time.Microsecond:
			d = d - (d % time.Microsecond)
		}

		values = append(values, []interface{}{id, qi.query, qi.database, d.String()})
	}

	return []*models.Row{{
		Columns: []string{"qid", "query", "database", "duration"},
		Values:  values,
	}}, nil
}

func (t *TaskManager) query(qid uint64) (*QueryTask, bool) {
	t.mu.RLock()
	query, ok := t.queries[qid]
	t.mu.RUnlock()
	return query, ok
}

// AttachQuery attaches a running query to be managed by the TaskManager.
// Returns the query id of the newly attached query or an error if it was
// unable to assign a query id or attach the query to the TaskManager.
// This function also returns a channel that will be closed when this
// query finishes running.
//
// After a query finishes running, the system is free to reuse a query id.
func (t *TaskManager) AttachQuery(q *Query, database string, interrupt <-chan struct{}) (uint64, *QueryTask, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.shutdown {
		return 0, nil, ErrQueryEngineShutdown
	}

	if t.MaxConcurrentQueries > 0 && len(t.queries) >= t.MaxConcurrentQueries {
		return 0, nil, ErrMaxConcurrentQueriesLimitExceeded(len(t.queries), t.MaxConcurrentQueries)
	}

	qid := t.nextID
	query := &QueryTask{
		query:     q.String(),
		database:  database,
		startTime: time.Now(),
		closing:   make(chan struct{}),
		monitorCh: make(chan error),
	}
	t.queries[qid] = query

	go t.waitForQuery(qid, query.closing, interrupt, query.monitorCh)
	if t.LogQueriesAfter != 0 {
		go query.monitor(func(closing <-chan struct{}) error {
			timer := time.NewTimer(t.LogQueriesAfter)
			defer timer.Stop()

			select {
			case <-timer.C:
				t.Logger.Printf("Detected slow query: %s (qid: %d, database: %s, threshold: %s)",
					query.query, qid, query.database, t.LogQueriesAfter)
			case <-closing:
			}
			return nil
		})
	}
	t.nextID++
	return qid, query, nil
}

// KillQuery stops and removes a query from the TaskManager.
// This method can be used to forcefully terminate a running query.
func (t *TaskManager) KillQuery(qid uint64) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	query, ok := t.queries[qid]
	if !ok {
		return fmt.Errorf("no such query id: %d", qid)
	}

	close(query.closing)
	delete(t.queries, qid)
	return nil
}

// QueryInfo represents the information for a query.
type QueryInfo struct {
	ID       uint64        `json:"id"`
	Query    string        `json:"query"`
	Database string        `json:"database"`
	Duration time.Duration `json:"duration"`
}

// Queries returns a list of all running queries with information about them.
func (t *TaskManager) Queries() []QueryInfo {
	t.mu.RLock()
	defer t.mu.RUnlock()

	now := time.Now()
	queries := make([]QueryInfo, 0, len(t.queries))
	for id, qi := range t.queries {
		queries = append(queries, QueryInfo{
			ID:       id,
			Query:    qi.query,
			Database: qi.database,
			Duration: now.Sub(qi.startTime),
		})
	}
	return queries
}

func (t *TaskManager) waitForQuery(qid uint64, interrupt <-chan struct{}, closing <-chan struct{}, monitorCh <-chan error) {
	var timerCh <-chan time.Time
	if t.QueryTimeout != 0 {
		timer := time.NewTimer(t.QueryTimeout)
		timerCh = timer.C
		defer timer.Stop()
	}

	select {
	case <-closing:
		query, ok := t.query(qid)
		if !ok {
			break
		}
		query.setError(ErrQueryInterrupted)
	case err := <-monitorCh:
		if err == nil {
			break
		}

		query, ok := t.query(qid)
		if !ok {
			break
		}
		query.setError(err)
	case <-timerCh:
		query, ok := t.query(qid)
		if !ok {
			break
		}
		query.setError(ErrQueryTimeoutLimitExceeded)
	case <-interrupt:
		// Query was manually closed so exit the select.
		return
	}
	t.KillQuery(qid)
}

// Close kills all running queries and prevents new queries from being attached.
func (t *TaskManager) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.shutdown = true
	for _, query := range t.queries {
		query.setError(ErrQueryEngineShutdown)
		close(query.closing)
	}
	t.queries = nil
	return nil
}
