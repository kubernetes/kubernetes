package monitor

import (
	"expvar"
	"fmt"
	"log"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/influxdb/influxdb"
	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/models"
)

const leaderWaitTimeout = 30 * time.Second

// Policy constants.
const (
	MonitorRetentionPolicy         = "monitor"
	MonitorRetentionPolicyDuration = 7 * 24 * time.Hour
)

// DiagsClient is the interface modules implement if they register diags with monitor.
type DiagsClient interface {
	Diagnostics() (*Diagnostic, error)
}

// The DiagsClientFunc type is an adapter to allow the use of
// ordinary functions as Diagnostis clients.
type DiagsClientFunc func() (*Diagnostic, error)

// Diagnostics calls f().
func (f DiagsClientFunc) Diagnostics() (*Diagnostic, error) {
	return f()
}

// Diagnostic represents a table of diagnostic information. The first value
// is the name of the columns, the second is a slice of interface slices containing
// the values for each column, by row. This information is never written to an InfluxDB
// system and is display-only. An example showing, say, connections follows:
//
//     source_ip    source_port       dest_ip     dest_port
//     182.1.0.2    2890              127.0.0.1   38901
//     174.33.1.2   2924              127.0.0.1   38902
type Diagnostic struct {
	Columns []string
	Rows    [][]interface{}
}

// NewDiagnostic initialises a new Diagnostic with the specified columns.
func NewDiagnostic(columns []string) *Diagnostic {
	return &Diagnostic{
		Columns: columns,
		Rows:    make([][]interface{}, 0),
	}
}

// AddRow appends the provided row to the Diagnostic's rows.
func (d *Diagnostic) AddRow(r []interface{}) {
	d.Rows = append(d.Rows, r)
}

// Monitor represents an instance of the monitor system.
type Monitor struct {
	// Build information for diagnostics.
	Version   string
	Commit    string
	Branch    string
	BuildTime string

	wg   sync.WaitGroup
	done chan struct{}
	mu   sync.Mutex

	diagRegistrations map[string]DiagsClient

	storeCreated           bool
	storeEnabled           bool
	storeDatabase          string
	storeRetentionPolicy   string
	storeRetentionDuration time.Duration
	storeReplicationFactor int
	storeAddress           string
	storeInterval          time.Duration

	MetaStore interface {
		ClusterID() (uint64, error)
		NodeID() uint64
		WaitForLeader(d time.Duration) error
		IsLeader() bool
		CreateDatabaseIfNotExists(name string) (*meta.DatabaseInfo, error)
		CreateRetentionPolicyIfNotExists(database string, rpi *meta.RetentionPolicyInfo) (*meta.RetentionPolicyInfo, error)
		SetDefaultRetentionPolicy(database, name string) error
		DropRetentionPolicy(database, name string) error
	}

	PointsWriter interface {
		WritePoints(p *cluster.WritePointsRequest) error
	}

	Logger *log.Logger
}

// New returns a new instance of the monitor system.
func New(c Config) *Monitor {
	return &Monitor{
		done:              make(chan struct{}),
		diagRegistrations: make(map[string]DiagsClient),
		storeEnabled:      c.StoreEnabled,
		storeDatabase:     c.StoreDatabase,
		storeInterval:     time.Duration(c.StoreInterval),
		Logger:            log.New(os.Stderr, "[monitor] ", log.LstdFlags),
	}
}

// Open opens the monitoring system, using the given clusterID, node ID, and hostname
// for identification purpose.
func (m *Monitor) Open() error {
	m.Logger.Printf("Starting monitor system")

	// Self-register various stats and diagnostics.
	m.RegisterDiagnosticsClient("build", &build{
		Version: m.Version,
		Commit:  m.Commit,
		Branch:  m.Branch,
		Time:    m.BuildTime,
	})
	m.RegisterDiagnosticsClient("runtime", &goRuntime{})
	m.RegisterDiagnosticsClient("network", &network{})
	m.RegisterDiagnosticsClient("system", &system{})

	// If enabled, record stats in a InfluxDB system.
	if m.storeEnabled {

		// Start periodic writes to system.
		m.wg.Add(1)
		go m.storeStatistics()
	}

	return nil
}

// Close closes the monitor system.
func (m *Monitor) Close() {
	m.Logger.Println("shutting down monitor system")
	close(m.done)
	m.wg.Wait()
	m.done = nil
}

// SetLogger sets the internal logger to the logger passed in.
func (m *Monitor) SetLogger(l *log.Logger) {
	m.Logger = l
}

// RegisterDiagnosticsClient registers a diagnostics client with the given name and tags.
func (m *Monitor) RegisterDiagnosticsClient(name string, client DiagsClient) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.diagRegistrations[name] = client
	m.Logger.Printf(`'%s' registered for diagnostics monitoring`, name)
}

// DeregisterDiagnosticsClient deregisters a diagnostics client by name.
func (m *Monitor) DeregisterDiagnosticsClient(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.diagRegistrations, name)
}

// Statistics returns the combined statistics for all expvar data. The given
// tags are added to each of the returned statistics.
func (m *Monitor) Statistics(tags map[string]string) ([]*Statistic, error) {
	var statistics []*Statistic

	expvar.Do(func(kv expvar.KeyValue) {
		// Skip built-in expvar stats.
		if kv.Key == "memstats" || kv.Key == "cmdline" {
			return
		}

		statistic := &Statistic{
			Tags:   make(map[string]string),
			Values: make(map[string]interface{}),
		}

		// Add any supplied tags.
		for k, v := range tags {
			statistic.Tags[k] = v
		}

		// Every other top-level expvar value is a map.
		m := kv.Value.(*expvar.Map)

		m.Do(func(subKV expvar.KeyValue) {
			switch subKV.Key {
			case "name":
				// straight to string name.
				u, err := strconv.Unquote(subKV.Value.String())
				if err != nil {
					return
				}
				statistic.Name = u
			case "tags":
				// string-string tags map.
				n := subKV.Value.(*expvar.Map)
				n.Do(func(t expvar.KeyValue) {
					u, err := strconv.Unquote(t.Value.String())
					if err != nil {
						return
					}
					statistic.Tags[t.Key] = u
				})
			case "values":
				// string-interface map.
				n := subKV.Value.(*expvar.Map)
				n.Do(func(kv expvar.KeyValue) {
					var f interface{}
					var err error
					switch v := kv.Value.(type) {
					case *expvar.Float:
						f, err = strconv.ParseFloat(v.String(), 64)
						if err != nil {
							return
						}
					case *expvar.Int:
						f, err = strconv.ParseInt(v.String(), 10, 64)
						if err != nil {
							return
						}
					default:
						return
					}
					statistic.Values[kv.Key] = f
				})
			}
		})

		// If a registered client has no field data, don't include it in the results
		if len(statistic.Values) == 0 {
			return
		}

		statistics = append(statistics, statistic)
	})

	// Add Go memstats.
	statistic := &Statistic{
		Name:   "runtime",
		Tags:   make(map[string]string),
		Values: make(map[string]interface{}),
	}

	// Add any supplied tags to Go memstats
	for k, v := range tags {
		statistic.Tags[k] = v
	}

	var rt runtime.MemStats
	runtime.ReadMemStats(&rt)
	statistic.Values = map[string]interface{}{
		"Alloc":        int64(rt.Alloc),
		"TotalAlloc":   int64(rt.TotalAlloc),
		"Sys":          int64(rt.Sys),
		"Lookups":      int64(rt.Lookups),
		"Mallocs":      int64(rt.Mallocs),
		"Frees":        int64(rt.Frees),
		"HeapAlloc":    int64(rt.HeapAlloc),
		"HeapSys":      int64(rt.HeapSys),
		"HeapIdle":     int64(rt.HeapIdle),
		"HeapInUse":    int64(rt.HeapInuse),
		"HeapReleased": int64(rt.HeapReleased),
		"HeapObjects":  int64(rt.HeapObjects),
		"PauseTotalNs": int64(rt.PauseTotalNs),
		"NumGC":        int64(rt.NumGC),
		"NumGoroutine": int64(runtime.NumGoroutine()),
	}
	statistics = append(statistics, statistic)

	return statistics, nil
}

// Diagnostics fetches diagnostic information for each registered
// diagnostic client. It skips any clients that return an error when
// retrieving their diagnostics.
func (m *Monitor) Diagnostics() (map[string]*Diagnostic, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	diags := make(map[string]*Diagnostic, len(m.diagRegistrations))
	for k, v := range m.diagRegistrations {
		d, err := v.Diagnostics()
		if err != nil {
			continue
		}
		diags[k] = d
	}
	return diags, nil
}

// createInternalStorage ensures the internal storage has been created.
func (m *Monitor) createInternalStorage() {
	if !m.MetaStore.IsLeader() || m.storeCreated {
		return
	}

	if _, err := m.MetaStore.CreateDatabaseIfNotExists(m.storeDatabase); err != nil {
		m.Logger.Printf("failed to create database '%s', failed to create storage: %s",
			m.storeDatabase, err.Error())
		return
	}

	rpi := meta.NewRetentionPolicyInfo(MonitorRetentionPolicy)
	rpi.Duration = MonitorRetentionPolicyDuration
	rpi.ReplicaN = 1
	if _, err := m.MetaStore.CreateRetentionPolicyIfNotExists(m.storeDatabase, rpi); err != nil {
		m.Logger.Printf("failed to create retention policy '%s', failed to create internal storage: %s",
			rpi.Name, err.Error())
		return
	}

	if err := m.MetaStore.SetDefaultRetentionPolicy(m.storeDatabase, rpi.Name); err != nil {
		m.Logger.Printf("failed to set default retention policy on '%s', failed to create internal storage: %s",
			m.storeDatabase, err.Error())
		return
	}

	err := m.MetaStore.DropRetentionPolicy(m.storeDatabase, "default")
	if err != nil && err.Error() != influxdb.ErrRetentionPolicyNotFound("default").Error() {
		m.Logger.Printf("failed to delete retention policy 'default', failed to created internal storage: %s", err.Error())
		return
	}

	// Mark storage creation complete.
	m.storeCreated = true
}

// storeStatistics writes the statistics to an InfluxDB system.
func (m *Monitor) storeStatistics() {
	defer m.wg.Done()
	m.Logger.Printf("Storing statistics in database '%s' retention policy '%s', at interval %s",
		m.storeDatabase, m.storeRetentionPolicy, m.storeInterval)

	if err := m.MetaStore.WaitForLeader(leaderWaitTimeout); err != nil {
		m.Logger.Printf("failed to detect a cluster leader, terminating storage: %s", err.Error())
		return
	}

	// Get cluster-level metadata. Nothing different is going to happen if errors occur.
	clusterID, _ := m.MetaStore.ClusterID()
	nodeID := m.MetaStore.NodeID()
	hostname, _ := os.Hostname()
	clusterTags := map[string]string{
		"clusterID": fmt.Sprintf("%d", clusterID),
		"nodeID":    fmt.Sprintf("%d", nodeID),
		"hostname":  hostname,
	}

	tick := time.NewTicker(m.storeInterval)
	defer tick.Stop()
	for {
		select {
		case <-tick.C:
			m.createInternalStorage()

			stats, err := m.Statistics(clusterTags)
			if err != nil {
				m.Logger.Printf("failed to retrieve registered statistics: %s", err)
				continue
			}

			points := make(models.Points, 0, len(stats))
			for _, s := range stats {
				pt, err := models.NewPoint(s.Name, s.Tags, s.Values, time.Now().Truncate(time.Second))
				if err != nil {
					m.Logger.Printf("Dropping point %v: %v", s.Name, err)
					continue
				}
				points = append(points, pt)
			}

			err = m.PointsWriter.WritePoints(&cluster.WritePointsRequest{
				Database:         m.storeDatabase,
				RetentionPolicy:  m.storeRetentionPolicy,
				ConsistencyLevel: cluster.ConsistencyLevelOne,
				Points:           points,
			})
			if err != nil {
				m.Logger.Printf("failed to store statistics: %s", err)
			}
		case <-m.done:
			m.Logger.Printf("terminating storage of statistics")
			return
		}

	}
}

// Statistic represents the information returned by a single monitor client.
type Statistic struct {
	Name   string                 `json:"name"`
	Tags   map[string]string      `json:"tags"`
	Values map[string]interface{} `json:"values"`
}

// newStatistic returns a new statistic object.
func newStatistic(name string, tags map[string]string, values map[string]interface{}) *Statistic {
	return &Statistic{
		Name:   name,
		Tags:   tags,
		Values: values,
	}
}

// valueNames returns a sorted list of the value names, if any.
func (s *Statistic) valueNames() []string {
	a := make([]string, 0, len(s.Values))
	for k := range s.Values {
		a = append(a, k)
	}
	sort.Strings(a)
	return a
}

// DiagnosticFromMap returns a Diagnostic from a map.
func DiagnosticFromMap(m map[string]interface{}) *Diagnostic {
	// Display columns in deterministic order.
	sortedKeys := make([]string, 0, len(m))
	for k := range m {
		sortedKeys = append(sortedKeys, k)
	}
	sort.Strings(sortedKeys)

	d := NewDiagnostic(sortedKeys)
	row := make([]interface{}, len(sortedKeys))
	for i, k := range sortedKeys {
		row[i] = m[k]
	}
	d.AddRow(row)

	return d
}
