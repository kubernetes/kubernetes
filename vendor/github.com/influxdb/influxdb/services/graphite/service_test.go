package graphite_test

import (
	"fmt"
	"net"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/services/graphite"
	"github.com/influxdb/influxdb/toml"
	"github.com/influxdb/influxdb/tsdb"
)

func Test_ServerGraphiteTCP(t *testing.T) {
	t.Parallel()

	now := time.Now().UTC().Round(time.Second)

	config := graphite.NewConfig()
	config.Database = "graphitedb"
	config.BatchSize = 0 // No batching.
	config.BatchTimeout = toml.Duration(time.Second)
	config.BindAddress = ":0"

	service, err := graphite.NewService(config)
	if err != nil {
		t.Fatalf("failed to create Graphite service: %s", err.Error())
	}

	// Allow test to wait until points are written.
	var wg sync.WaitGroup
	wg.Add(1)

	pointsWriter := PointsWriter{
		WritePointsFn: func(req *cluster.WritePointsRequest) error {
			defer wg.Done()

			if req.Database != "graphitedb" {
				t.Fatalf("unexpected database: %s", req.Database)
			} else if req.RetentionPolicy != "" {
				t.Fatalf("unexpected retention policy: %s", req.RetentionPolicy)
			} else if !reflect.DeepEqual(req.Points, []tsdb.Point{
				tsdb.NewPoint(
					"cpu",
					map[string]string{},
					map[string]interface{}{"value": 23.456},
					time.Unix(now.Unix(), 0),
				),
			}) {
				spew.Dump(req.Points)
				t.Fatalf("unexpected points: %#v", req.Points)
			}
			return nil
		},
	}
	service.PointsWriter = &pointsWriter
	dbCreator := DatabaseCreator{}
	service.MetaStore = &dbCreator

	if err := service.Open(); err != nil {
		t.Fatalf("failed to open Graphite service: %s", err.Error())
	}

	if !dbCreator.Created {
		t.Fatalf("failed to create target database")
	}

	// Connect to the graphite endpoint we just spun up
	_, port, _ := net.SplitHostPort(service.Addr().String())
	conn, err := net.Dial("tcp", "127.0.0.1:"+port)
	if err != nil {
		t.Fatal(err)
	}
	data := []byte(`cpu 23.456 `)
	data = append(data, []byte(fmt.Sprintf("%d", now.Unix()))...)
	data = append(data, '\n')
	_, err = conn.Write(data)
	conn.Close()
	if err != nil {
		t.Fatal(err)
	}

	wg.Wait()
}

func Test_ServerGraphiteUDP(t *testing.T) {
	t.Parallel()

	now := time.Now().UTC().Round(time.Second)

	config := graphite.NewConfig()
	config.Database = "graphitedb"
	config.BatchSize = 0 // No batching.
	config.BatchTimeout = toml.Duration(time.Second)
	config.BindAddress = ":10000"
	config.Protocol = "udp"

	service, err := graphite.NewService(config)
	if err != nil {
		t.Fatalf("failed to create Graphite service: %s", err.Error())
	}

	// Allow test to wait until points are written.
	var wg sync.WaitGroup
	wg.Add(1)

	pointsWriter := PointsWriter{
		WritePointsFn: func(req *cluster.WritePointsRequest) error {
			defer wg.Done()

			if req.Database != "graphitedb" {
				t.Fatalf("unexpected database: %s", req.Database)
			} else if req.RetentionPolicy != "" {
				t.Fatalf("unexpected retention policy: %s", req.RetentionPolicy)
			} else if !reflect.DeepEqual(req.Points, []tsdb.Point{
				tsdb.NewPoint(
					"cpu",
					map[string]string{},
					map[string]interface{}{"value": 23.456},
					time.Unix(now.Unix(), 0),
				),
			}) {
				spew.Dump(req.Points)
				t.Fatalf("unexpected points: %#v", req.Points)
			}
			return nil
		},
	}
	service.PointsWriter = &pointsWriter
	dbCreator := DatabaseCreator{}
	service.MetaStore = &dbCreator

	if err := service.Open(); err != nil {
		t.Fatalf("failed to open Graphite service: %s", err.Error())
	}

	if !dbCreator.Created {
		t.Fatalf("failed to create target database")
	}

	// Connect to the graphite endpoint we just spun up
	_, port, _ := net.SplitHostPort(service.Addr().String())
	conn, err := net.Dial("udp", "127.0.0.1:"+port)
	if err != nil {
		t.Fatal(err)
	}
	data := []byte(`cpu 23.456 `)
	data = append(data, []byte(fmt.Sprintf("%d", now.Unix()))...)
	data = append(data, '\n')
	_, err = conn.Write(data)
	if err != nil {
		t.Fatal(err)
	}

	wg.Wait()
	conn.Close()
}

// PointsWriter represents a mock impl of PointsWriter.
type PointsWriter struct {
	WritePointsFn func(*cluster.WritePointsRequest) error
}

func (w *PointsWriter) WritePoints(p *cluster.WritePointsRequest) error {
	return w.WritePointsFn(p)
}

type DatabaseCreator struct {
	Created bool
}

func (d *DatabaseCreator) CreateDatabaseIfNotExists(name string) (*meta.DatabaseInfo, error) {
	d.Created = true
	return nil, nil
}

func (d *DatabaseCreator) WaitForLeader(t time.Duration) error {
	return nil
}

// Test Helpers
func errstr(err error) string {
	if err != nil {
		return err.Error()
	}
	return ""
}
