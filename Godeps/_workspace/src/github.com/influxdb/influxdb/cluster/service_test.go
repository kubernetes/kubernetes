package cluster_test

import (
	"fmt"
	"net"
	"time"

	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/influxql"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/models"
	"github.com/influxdb/influxdb/tcp"
	"github.com/influxdb/influxdb/tsdb"
)

type metaStore struct {
	host string
}

func (m *metaStore) Node(nodeID uint64) (*meta.NodeInfo, error) {
	return &meta.NodeInfo{
		ID:   nodeID,
		Host: m.host,
	}, nil
}

type testService struct {
	nodeID           uint64
	ln               net.Listener
	muxln            net.Listener
	writeShardFunc   func(shardID uint64, points []models.Point) error
	createShardFunc  func(database, policy string, shardID uint64) error
	createMapperFunc func(shardID uint64, stmt influxql.Statement, chunkSize int) (tsdb.Mapper, error)
}

func newTestWriteService(f func(shardID uint64, points []models.Point) error) testService {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}

	mux := tcp.NewMux()
	muxln := mux.Listen(cluster.MuxHeader)
	go mux.Serve(ln)

	return testService{
		writeShardFunc: f,
		ln:             ln,
		muxln:          muxln,
	}
}

func (ts *testService) Close() {
	if ts.ln != nil {
		ts.ln.Close()
	}
}

type serviceResponses []serviceResponse
type serviceResponse struct {
	shardID uint64
	ownerID uint64
	points  []models.Point
}

func (t testService) WriteToShard(shardID uint64, points []models.Point) error {
	return t.writeShardFunc(shardID, points)
}

func (t testService) CreateShard(database, policy string, shardID uint64) error {
	return t.createShardFunc(database, policy, shardID)
}

func (t testService) CreateMapper(shardID uint64, stmt influxql.Statement, chunkSize int) (tsdb.Mapper, error) {
	return t.createMapperFunc(shardID, stmt, chunkSize)
}

func writeShardSuccess(shardID uint64, points []models.Point) error {
	responses <- &serviceResponse{
		shardID: shardID,
		points:  points,
	}
	return nil
}

func writeShardFail(shardID uint64, points []models.Point) error {
	return fmt.Errorf("failed to write")
}

var responses = make(chan *serviceResponse, 1024)

func (testService) ResponseN(n int) ([]*serviceResponse, error) {
	var a []*serviceResponse
	for {
		select {
		case r := <-responses:
			a = append(a, r)
			if len(a) == n {
				return a, nil
			}
		case <-time.After(time.Second):
			return a, fmt.Errorf("unexpected response count: expected: %d, actual: %d", n, len(a))
		}
	}
}
