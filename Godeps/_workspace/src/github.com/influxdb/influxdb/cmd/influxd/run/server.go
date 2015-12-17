package run

import (
	"fmt"
	"log"
	"net"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/monitor"
	"github.com/influxdb/influxdb/services/admin"
	"github.com/influxdb/influxdb/services/collectd"
	"github.com/influxdb/influxdb/services/continuous_querier"
	"github.com/influxdb/influxdb/services/copier"
	"github.com/influxdb/influxdb/services/graphite"
	"github.com/influxdb/influxdb/services/hh"
	"github.com/influxdb/influxdb/services/httpd"
	"github.com/influxdb/influxdb/services/opentsdb"
	"github.com/influxdb/influxdb/services/precreator"
	"github.com/influxdb/influxdb/services/retention"
	"github.com/influxdb/influxdb/services/snapshotter"
	"github.com/influxdb/influxdb/services/subscriber"
	"github.com/influxdb/influxdb/services/udp"
	"github.com/influxdb/influxdb/tcp"
	"github.com/influxdb/influxdb/tsdb"
	"github.com/influxdb/usage-client/v1"
	// Initialize the engine packages
	_ "github.com/influxdb/influxdb/tsdb/engine"
)

// BuildInfo represents the build details for the server code.
type BuildInfo struct {
	Version string
	Commit  string
	Branch  string
	Time    string
}

// Server represents a container for the metadata and storage data and services.
// It is built using a Config and it manages the startup and shutdown of all
// services in the proper order.
type Server struct {
	buildInfo BuildInfo

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
	Subscriber    *subscriber.Service

	Services []Service

	// These references are required for the tcp muxer.
	ClusterService     *cluster.Service
	SnapshotterService *snapshotter.Service
	CopierService      *copier.Service

	Monitor *monitor.Monitor

	// Server reporting and registration
	reportingDisabled bool

	// Profiling
	CPUProfile string
	MemProfile string
}

// NewServer returns a new instance of Server built from a config.
func NewServer(c *Config, buildInfo *BuildInfo) (*Server, error) {
	// Construct base meta store and data store.
	tsdbStore := tsdb.NewStore(c.Data.Dir)
	tsdbStore.EngineOptions.Config = c.Data

	s := &Server{
		buildInfo: *buildInfo,
		err:       make(chan error),
		closing:   make(chan struct{}),

		Hostname:    c.Meta.Hostname,
		BindAddress: c.Meta.BindAddress,

		MetaStore: meta.NewStore(c.Meta),
		TSDBStore: tsdbStore,

		Monitor: monitor.New(c.Monitor),

		reportingDisabled: c.ReportingDisabled,
	}

	// Copy TSDB configuration.
	s.TSDBStore.EngineOptions.EngineVersion = c.Data.Engine
	s.TSDBStore.EngineOptions.MaxWALSize = c.Data.MaxWALSize
	s.TSDBStore.EngineOptions.WALFlushInterval = time.Duration(c.Data.WALFlushInterval)
	s.TSDBStore.EngineOptions.WALPartitionFlushDelay = time.Duration(c.Data.WALPartitionFlushDelay)

	// Set the shard mapper
	s.ShardMapper = cluster.NewShardMapper(time.Duration(c.Cluster.ShardMapperTimeout))
	s.ShardMapper.ForceRemoteMapping = c.Cluster.ForceRemoteShardMapping
	s.ShardMapper.MetaStore = s.MetaStore
	s.ShardMapper.TSDBStore = s.TSDBStore

	// Initialize query executor.
	s.QueryExecutor = tsdb.NewQueryExecutor(s.TSDBStore)
	s.QueryExecutor.MetaStore = s.MetaStore
	s.QueryExecutor.MetaStatementExecutor = &meta.StatementExecutor{Store: s.MetaStore}
	s.QueryExecutor.MonitorStatementExecutor = &monitor.StatementExecutor{Monitor: s.Monitor}
	s.QueryExecutor.ShardMapper = s.ShardMapper
	s.QueryExecutor.QueryLogEnabled = c.Data.QueryLogEnabled

	// Set the shard writer
	s.ShardWriter = cluster.NewShardWriter(time.Duration(c.Cluster.ShardWriterTimeout))
	s.ShardWriter.MetaStore = s.MetaStore

	// Create the hinted handoff service
	s.HintedHandoff = hh.NewService(c.HintedHandoff, s.ShardWriter, s.MetaStore)
	s.HintedHandoff.Monitor = s.Monitor

	// Create the Subscriber service
	s.Subscriber = subscriber.NewService(c.Subscriber)
	s.Subscriber.MetaStore = s.MetaStore

	// Initialize points writer.
	s.PointsWriter = cluster.NewPointsWriter()
	s.PointsWriter.WriteTimeout = time.Duration(c.Cluster.WriteTimeout)
	s.PointsWriter.MetaStore = s.MetaStore
	s.PointsWriter.TSDBStore = s.TSDBStore
	s.PointsWriter.ShardWriter = s.ShardWriter
	s.PointsWriter.HintedHandoff = s.HintedHandoff
	s.PointsWriter.Subscriber = s.Subscriber

	// needed for executing INTO queries.
	s.QueryExecutor.IntoWriter = s.PointsWriter

	// Initialize the monitor
	s.Monitor.Version = s.buildInfo.Version
	s.Monitor.Commit = s.buildInfo.Commit
	s.Monitor.Branch = s.buildInfo.Branch
	s.Monitor.BuildTime = s.buildInfo.Time
	s.Monitor.MetaStore = s.MetaStore
	s.Monitor.PointsWriter = s.PointsWriter

	// Append services.
	s.appendClusterService(c.Cluster)
	s.appendPrecreatorService(c.Precreator)
	s.appendSnapshotterService()
	s.appendCopierService()
	s.appendAdminService(c.Admin)
	s.appendContinuousQueryService(c.ContinuousQuery)
	s.appendHTTPDService(c.HTTPD)
	s.appendCollectdService(c.Collectd)
	if err := s.appendOpenTSDBService(c.OpenTSDB); err != nil {
		return nil, err
	}
	for _, g := range c.UDPs {
		s.appendUDPService(g)
	}
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

func (s *Server) appendCopierService() {
	srv := copier.NewService()
	srv.TSDBStore = s.TSDBStore
	s.Services = append(s.Services, srv)
	s.CopierService = srv
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
	srv.Handler.PointsWriter = s.PointsWriter
	srv.Handler.Version = s.buildInfo.Version

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
	srv.Monitor = s.Monitor
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
	srv.MetaStore = s.MetaStore
	s.Services = append(s.Services, srv)
}

func (s *Server) appendContinuousQueryService(c continuous_querier.Config) {
	if !c.Enabled {
		return
	}
	srv := continuous_querier.NewService(c)
	srv.MetaStore = s.MetaStore
	srv.QueryExecutor = s.QueryExecutor
	s.Services = append(s.Services, srv)
}

// Err returns an error channel that multiplexes all out of band errors received from all services.
func (s *Server) Err() <-chan error { return s.err }

// Open opens the meta and data store and all services.
func (s *Server) Open() error {
	if err := func() error {
		// Start profiling, if set.
		startProfile(s.CPUProfile, s.MemProfile)

		host, port, err := s.hostAddr()
		if err != nil {
			return err
		}

		hostport := net.JoinHostPort(host, port)
		addr, err := net.ResolveTCPAddr("tcp", hostport)
		if err != nil {
			return fmt.Errorf("resolve tcp: addr=%s, err=%s", hostport, err)
		}
		s.MetaStore.Addr = addr
		s.MetaStore.RemoteAddr = &tcpaddr{hostport}

		// Open shared TCP connection.
		ln, err := net.Listen("tcp", s.BindAddress)
		if err != nil {
			return fmt.Errorf("listen: %s", err)
		}
		s.Listener = ln

		// The port 0 is used, we need to retrieve the port assigned by the kernel
		if strings.HasSuffix(s.BindAddress, ":0") {
			s.MetaStore.Addr = ln.Addr()
			s.MetaStore.RemoteAddr = ln.Addr()
		}

		// Multiplex listener.
		mux := tcp.NewMux()
		s.MetaStore.RaftListener = mux.Listen(meta.MuxRaftHeader)
		s.MetaStore.ExecListener = mux.Listen(meta.MuxExecHeader)
		s.MetaStore.RPCListener = mux.Listen(meta.MuxRPCHeader)

		s.ClusterService.Listener = mux.Listen(cluster.MuxHeader)
		s.SnapshotterService.Listener = mux.Listen(snapshotter.MuxHeader)
		s.CopierService.Listener = mux.Listen(copier.MuxHeader)
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

		// Open the subcriber service
		if err := s.Subscriber.Open(); err != nil {
			return fmt.Errorf("open subscriber: %s", err)
		}

		// Open the points writer service
		if err := s.PointsWriter.Open(); err != nil {
			return fmt.Errorf("open points writer: %s", err)
		}

		// Open the monitor service
		if err := s.Monitor.Open(); err != nil {
			return fmt.Errorf("open monitor: %v", err)
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

	// Close the listener first to stop any new connections
	if s.Listener != nil {
		s.Listener.Close()
	}

	// Close services to allow any inflight requests to complete
	// and prevent new requests from being accepted.
	for _, service := range s.Services {
		service.Close()
	}

	if s.Monitor != nil {
		s.Monitor.Close()
	}

	if s.PointsWriter != nil {
		s.PointsWriter.Close()
	}

	if s.HintedHandoff != nil {
		s.HintedHandoff.Close()
	}

	// Close the TSDBStore, no more reads or writes at this point
	if s.TSDBStore != nil {
		s.TSDBStore.Close()
	}

	if s.Subscriber != nil {
		s.Subscriber.Close()
	}

	// Finally close the meta-store since everything else depends on it
	if s.MetaStore != nil {
		s.MetaStore.Close()
	}

	close(s.closing)
	return nil
}

// startServerReporting starts periodic server reporting.
func (s *Server) startServerReporting() {
	for {
		select {
		case <-s.closing:
			return
		default:
		}
		if err := s.MetaStore.WaitForLeader(30 * time.Second); err != nil {
			log.Printf("no leader available for reporting: %s", err.Error())
			time.Sleep(time.Second)
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

	cl := client.New("")
	usage := client.Usage{
		Product: "influxdb",
		Data: []client.UsageData{
			{
				Values: client.Values{
					"os":               runtime.GOOS,
					"arch":             runtime.GOARCH,
					"version":          s.buildInfo.Version,
					"server_id":        fmt.Sprintf("%v", s.MetaStore.NodeID()),
					"cluster_id":       fmt.Sprintf("%v", clusterID),
					"num_series":       numSeries,
					"num_measurements": numMeasurements,
					"num_databases":    numDatabases,
				},
			},
		},
	}

	log.Printf("Sending anonymous usage statistics to m.influxdb.com")

	go cl.Save(usage)
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

// hostAddr returns the host and port that remote nodes will use to reach this
// node.
func (s *Server) hostAddr() (string, string, error) {
	// Resolve host to address.
	_, port, err := net.SplitHostPort(s.BindAddress)
	if err != nil {
		return "", "", fmt.Errorf("split bind address: %s", err)
	}

	host := s.Hostname

	// See if we might have a port that will override the BindAddress port
	if host != "" && host[len(host)-1] >= '0' && host[len(host)-1] <= '9' && strings.Contains(host, ":") {
		hostArg, portArg, err := net.SplitHostPort(s.Hostname)
		if err != nil {
			return "", "", err
		}

		if hostArg != "" {
			host = hostArg
		}

		if portArg != "" {
			port = portArg
		}
	}
	return host, port, nil
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

type tcpaddr struct{ host string }

func (a *tcpaddr) Network() string { return "tcp" }
func (a *tcpaddr) String() string  { return a.host }
