package run

import (
	"bytes"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/services/admin"
	"github.com/influxdb/influxdb/services/collectd"
	"github.com/influxdb/influxdb/services/continuous_querier"
	"github.com/influxdb/influxdb/services/graphite"
	"github.com/influxdb/influxdb/services/hh"
	"github.com/influxdb/influxdb/services/httpd"
	"github.com/influxdb/influxdb/services/opentsdb"
	"github.com/influxdb/influxdb/services/precreator"
	"github.com/influxdb/influxdb/services/retention"
	"github.com/influxdb/influxdb/services/snapshotter"
	"github.com/influxdb/influxdb/services/udp"
	"github.com/influxdb/influxdb/tcp"
	"github.com/influxdb/influxdb/tsdb"
)

// Server represents a container for the metadata and storage data and services.
// It is built using a Config and it manages the startup and shutdown of all
// services in the proper order.
type Server struct {
	version string // Build version

	err     chan error
	closing chan struct{}

	Hostname    string
	BindAddress string
	Listener    net.Listener

	MetaStore     *meta.Store
	TSDBStore     *tsdb.Store
	QueryExecutor *tsdb.QueryExecutor
	PointsWriter  *cluster.PointsWriter
	ShardWriter   *cluster.ShardWriter
	ShardMapper   *cluster.ShardMapper
	HintedHandoff *hh.Service

	Services []Service

	// These references are required for the tcp muxer.
	ClusterService     *cluster.Service
	SnapshotterService *snapshotter.Service

	// Server reporting
	reportingDisabled bool

	// Profiling
	CPUProfile string
	MemProfile string
}

// NewServer returns a new instance of Server built from a config.
func NewServer(c *Config, version string) (*Server, error) {
	// Construct base meta store and data store.
	s := &Server{
		version: version,
		err:     make(chan error),
		closing: make(chan struct{}),

		Hostname:    c.Meta.Hostname,
		BindAddress: c.Meta.BindAddress,

		MetaStore: meta.NewStore(c.Meta),
		TSDBStore: tsdb.NewStore(c.Data.Dir),

		reportingDisabled: c.ReportingDisabled,
	}

	// Copy TSDB configuration.
	s.TSDBStore.MaxWALSize = c.Data.MaxWALSize
	s.TSDBStore.WALFlushInterval = time.Duration(c.Data.WALFlushInterval)
	s.TSDBStore.WALPartitionFlushDelay = time.Duration(c.Data.WALPartitionFlushDelay)

	// Set the shard mapper
	s.ShardMapper = cluster.NewShardMapper()
	s.ShardMapper.MetaStore = s.MetaStore
	s.ShardMapper.TSDBStore = s.TSDBStore

	// Initialize query executor.
	s.QueryExecutor = tsdb.NewQueryExecutor(s.TSDBStore)
	s.QueryExecutor.MetaStore = s.MetaStore
	s.QueryExecutor.MetaStatementExecutor = &meta.StatementExecutor{Store: s.MetaStore}
	s.QueryExecutor.ShardMapper = s.ShardMapper

	// Set the shard writer
	s.ShardWriter = cluster.NewShardWriter(time.Duration(c.Cluster.ShardWriterTimeout))
	s.ShardWriter.MetaStore = s.MetaStore

	// Create the hinted handoff service
	s.HintedHandoff = hh.NewService(c.HintedHandoff, s.ShardWriter)

	// Initialize points writer.
	s.PointsWriter = cluster.NewPointsWriter()
	s.PointsWriter.WriteTimeout = time.Duration(c.Cluster.WriteTimeout)
	s.PointsWriter.MetaStore = s.MetaStore
	s.PointsWriter.TSDBStore = s.TSDBStore
	s.PointsWriter.ShardWriter = s.ShardWriter
	s.PointsWriter.HintedHandoff = s.HintedHandoff

	// Append services.
	s.appendClusterService(c.Cluster)
	s.appendPrecreatorService(c.Precreator)
	s.appendSnapshotterService()
	s.appendAdminService(c.Admin)
	s.appendContinuousQueryService(c.ContinuousQuery)
	s.appendHTTPDService(c.HTTPD)
	s.appendCollectdService(c.Collectd)
	if err := s.appendOpenTSDBService(c.OpenTSDB); err != nil {
		return nil, err
	}
	s.appendUDPService(c.UDP)
	s.appendRetentionPolicyService(c.Retention)
	for _, g := range c.Graphites {
		if err := s.appendGraphiteService(g); err != nil {
			return nil, err
		}
	}

	return s, nil
}

func (s *Server) appendClusterService(c cluster.Config) {
	srv := cluster.NewService(c)
	srv.TSDBStore = s.TSDBStore
	srv.MetaStore = s.MetaStore
	s.Services = append(s.Services, srv)
	s.ClusterService = srv
}

func (s *Server) appendSnapshotterService() {
	srv := snapshotter.NewService()
	srv.TSDBStore = s.TSDBStore
	srv.MetaStore = s.MetaStore
	s.Services = append(s.Services, srv)
	s.SnapshotterService = srv
}

func (s *Server) appendRetentionPolicyService(c retention.Config) {
	if !c.Enabled {
		return
	}
	srv := retention.NewService(c)
	srv.MetaStore = s.MetaStore
	srv.TSDBStore = s.TSDBStore
	s.Services = append(s.Services, srv)
}

func (s *Server) appendAdminService(c admin.Config) {
	if !c.Enabled {
		return
	}
	srv := admin.NewService(c)
	s.Services = append(s.Services, srv)
}

func (s *Server) appendHTTPDService(c httpd.Config) {
	if !c.Enabled {
		return
	}
	srv := httpd.NewService(c)
	srv.Handler.MetaStore = s.MetaStore
	srv.Handler.QueryExecutor = s.QueryExecutor
	srv.Handler.TSDBStore = s.TSDBStore
	srv.Handler.PointsWriter = s.PointsWriter
	srv.Handler.Version = s.version

	// If a ContinuousQuerier service has been started, attach it.
	for _, srvc := range s.Services {
		if cqsrvc, ok := srvc.(continuous_querier.ContinuousQuerier); ok {
			srv.Handler.ContinuousQuerier = cqsrvc
		}
	}

	s.Services = append(s.Services, srv)
}

func (s *Server) appendCollectdService(c collectd.Config) {
	if !c.Enabled {
		return
	}
	srv := collectd.NewService(c)
	srv.MetaStore = s.MetaStore
	srv.PointsWriter = s.PointsWriter
	s.Services = append(s.Services, srv)
}

func (s *Server) appendOpenTSDBService(c opentsdb.Config) error {
	if !c.Enabled {
		return nil
	}
	srv, err := opentsdb.NewService(c)
	if err != nil {
		return err
	}
	srv.PointsWriter = s.PointsWriter
	srv.MetaStore = s.MetaStore
	s.Services = append(s.Services, srv)
	return nil
}

func (s *Server) appendGraphiteService(c graphite.Config) error {
	if !c.Enabled {
		return nil
	}
	srv, err := graphite.NewService(c)
	if err != nil {
		return err
	}

	srv.PointsWriter = s.PointsWriter
	srv.MetaStore = s.MetaStore
	s.Services = append(s.Services, srv)
	return nil
}

func (s *Server) appendPrecreatorService(c precreator.Config) error {
	if !c.Enabled {
		return nil
	}
	srv, err := precreator.NewService(c)
	if err != nil {
		return err
	}

	srv.MetaStore = s.MetaStore
	s.Services = append(s.Services, srv)
	return nil
}

func (s *Server) appendUDPService(c udp.Config) {
	if !c.Enabled {
		return
	}
	srv := udp.NewService(c)
	srv.PointsWriter = s.PointsWriter
	s.Services = append(s.Services, srv)
}

func (s *Server) appendContinuousQueryService(c continuous_querier.Config) {
	if !c.Enabled {
		return
	}
	srv := continuous_querier.NewService(c)
	srv.MetaStore = s.MetaStore
	srv.QueryExecutor = s.QueryExecutor
	srv.PointsWriter = s.PointsWriter
	s.Services = append(s.Services, srv)
}

// Err returns an error channel that multiplexes all out of band errors received from all services.
func (s *Server) Err() <-chan error { return s.err }

// Open opens the meta and data store and all services.
func (s *Server) Open() error {
	if err := func() error {
		// Start profiling, if set.
		startProfile(s.CPUProfile, s.MemProfile)

		// Resolve host to address.
		_, port, err := net.SplitHostPort(s.BindAddress)
		if err != nil {
			return fmt.Errorf("split bind address: %s", err)
		}
		hostport := net.JoinHostPort(s.Hostname, port)
		addr, err := net.ResolveTCPAddr("tcp", hostport)
		if err != nil {
			return fmt.Errorf("resolve tcp: addr=%s, err=%s", hostport, err)
		}
		s.MetaStore.Addr = addr

		// Open shared TCP connection.
		ln, err := net.Listen("tcp", s.BindAddress)
		if err != nil {
			return fmt.Errorf("listen: %s", err)
		}
		s.Listener = ln

		// Multiplex listener.
		mux := tcp.NewMux()
		s.MetaStore.RaftListener = mux.Listen(meta.MuxRaftHeader)
		s.MetaStore.ExecListener = mux.Listen(meta.MuxExecHeader)
		s.ClusterService.Listener = mux.Listen(cluster.MuxHeader)
		s.SnapshotterService.Listener = mux.Listen(snapshotter.MuxHeader)
		go mux.Serve(ln)

		// Open meta store.
		if err := s.MetaStore.Open(); err != nil {
			return fmt.Errorf("open meta store: %s", err)
		}
		go s.monitorErrorChan(s.MetaStore.Err())

		// Wait for the store to initialize.
		<-s.MetaStore.Ready()

		// Open TSDB store.
		if err := s.TSDBStore.Open(); err != nil {
			return fmt.Errorf("open tsdb store: %s", err)
		}

		// Open the hinted handoff service
		if err := s.HintedHandoff.Open(); err != nil {
			return fmt.Errorf("open hinted handoff: %s", err)
		}

		for _, service := range s.Services {
			if err := service.Open(); err != nil {
				return fmt.Errorf("open service: %s", err)
			}
		}

		// Start the reporting service, if not disabled.
		if !s.reportingDisabled {
			go s.startServerReporting()
		}

		return nil

	}(); err != nil {
		s.Close()
		return err
	}

	return nil
}

// Close shuts down the meta and data stores and all services.
func (s *Server) Close() error {
	stopProfile()

	if s.Listener != nil {
		s.Listener.Close()
	}
	if s.MetaStore != nil {
		s.MetaStore.Close()
	}
	if s.TSDBStore != nil {
		s.TSDBStore.Close()
	}
	if s.HintedHandoff != nil {
		s.HintedHandoff.Close()
	}
	for _, service := range s.Services {
		service.Close()
	}

	close(s.closing)
	return nil
}

// startServerReporting starts periodic server reporting.
func (s *Server) startServerReporting() {
	for {
		if err := s.MetaStore.WaitForLeader(30 * time.Second); err != nil {
			log.Printf("no leader available for reporting: %s", err.Error())
			continue
		}
		s.reportServer()
		<-time.After(24 * time.Hour)
	}
}

// reportServer reports anonymous statistics about the system.
func (s *Server) reportServer() {
	dis, err := s.MetaStore.Databases()
	if err != nil {
		log.Printf("failed to retrieve databases for reporting: %s", err.Error())
		return
	}
	numDatabases := len(dis)

	numMeasurements := 0
	numSeries := 0
	for _, di := range dis {
		d := s.TSDBStore.DatabaseIndex(di.Name)
		if d == nil {
			// No data in this store for this database.
			continue
		}
		m, s := d.MeasurementSeriesCounts()
		numMeasurements += m
		numSeries += s
	}

	clusterID, err := s.MetaStore.ClusterID()
	if err != nil {
		log.Printf("failed to retrieve cluster ID for reporting: %s", err.Error())
		return
	}

	json := fmt.Sprintf(`[{
    "name":"reports",
    "columns":["os", "arch", "version", "server_id", "cluster_id", "num_series", "num_measurements", "num_databases"],
    "points":[["%s", "%s", "%s", "%x", "%x", "%d", "%d", "%d"]]
  }]`, runtime.GOOS, runtime.GOARCH, s.version, s.MetaStore.NodeID(), clusterID, numSeries, numMeasurements, numDatabases)

	data := bytes.NewBufferString(json)

	log.Printf("Sending anonymous usage statistics to m.influxdb.com")

	client := http.Client{Timeout: time.Duration(5 * time.Second)}
	go client.Post("http://m.influxdb.com:8086/db/reporting/series?u=reporter&p=influxdb", "application/json", data)
}

// monitorErrorChan reads an error channel and resends it through the server.
func (s *Server) monitorErrorChan(ch <-chan error) {
	for {
		select {
		case err, ok := <-ch:
			if !ok {
				return
			}
			s.err <- err
		case <-s.closing:
			return
		}
	}
}

// Service represents a service attached to the server.
type Service interface {
	Open() error
	Close() error
}

// prof stores the file locations of active profiles.
var prof struct {
	cpu *os.File
	mem *os.File
}

// StartProfile initializes the cpu and memory profile, if specified.
func startProfile(cpuprofile, memprofile string) {
	if cpuprofile != "" {
		f, err := os.Create(cpuprofile)
		if err != nil {
			log.Fatalf("cpuprofile: %v", err)
		}
		log.Printf("writing CPU profile to: %s\n", cpuprofile)
		prof.cpu = f
		pprof.StartCPUProfile(prof.cpu)
	}

	if memprofile != "" {
		f, err := os.Create(memprofile)
		if err != nil {
			log.Fatalf("memprofile: %v", err)
		}
		log.Printf("writing mem profile to: %s\n", memprofile)
		prof.mem = f
		runtime.MemProfileRate = 4096
	}

}

// StopProfile closes the cpu and memory profiles if they are running.
func stopProfile() {
	if prof.cpu != nil {
		pprof.StopCPUProfile()
		prof.cpu.Close()
		log.Println("CPU profile stopped")
	}
	if prof.mem != nil {
		pprof.Lookup("heap").WriteTo(prof.mem, 0)
		prof.mem.Close()
		log.Println("mem profile stopped")
	}
}
