package zk

import (
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

type TestServer struct {
	Port int
	Path string
	Srv  *Server
}

type TestCluster struct {
	Path    string
	Servers []TestServer
}

func StartTestCluster(size int, stdout, stderr io.Writer) (*TestCluster, error) {
	tmpPath, err := ioutil.TempDir("", "gozk")
	if err != nil {
		return nil, err
	}
	success := false
	startPort := int(rand.Int31n(6000) + 10000)
	cluster := &TestCluster{Path: tmpPath}
	defer func() {
		if !success {
			cluster.Stop()
		}
	}()
	for serverN := 0; serverN < size; serverN++ {
		srvPath := filepath.Join(tmpPath, fmt.Sprintf("srv%d", serverN))
		if err := os.Mkdir(srvPath, 0700); err != nil {
			return nil, err
		}
		port := startPort + serverN*3
		cfg := ServerConfig{
			ClientPort: port,
			DataDir:    srvPath,
		}
		for i := 0; i < size; i++ {
			cfg.Servers = append(cfg.Servers, ServerConfigServer{
				ID:                 i + 1,
				Host:               "127.0.0.1",
				PeerPort:           startPort + i*3 + 1,
				LeaderElectionPort: startPort + i*3 + 2,
			})
		}
		cfgPath := filepath.Join(srvPath, "zoo.cfg")
		fi, err := os.Create(cfgPath)
		if err != nil {
			return nil, err
		}
		err = cfg.Marshall(fi)
		fi.Close()
		if err != nil {
			return nil, err
		}

		fi, err = os.Create(filepath.Join(srvPath, "myid"))
		if err != nil {
			return nil, err
		}
		_, err = fmt.Fprintf(fi, "%d\n", serverN+1)
		fi.Close()
		if err != nil {
			return nil, err
		}

		srv := &Server{
			ConfigPath: cfgPath,
			Stdout:     stdout,
			Stderr:     stderr,
		}
		if err := srv.Start(); err != nil {
			return nil, err
		}
		cluster.Servers = append(cluster.Servers, TestServer{
			Path: srvPath,
			Port: cfg.ClientPort,
			Srv:  srv,
		})
	}
	success = true
	time.Sleep(time.Second) // Give the server time to become active. Should probably actually attempt to connect to verify.
	return cluster, nil
}

func (ts *TestCluster) Connect(idx int) (*Conn, error) {
	zk, _, err := Connect([]string{fmt.Sprintf("127.0.0.1:%d", ts.Servers[idx].Port)}, time.Second*15)
	return zk, err
}

func (ts *TestCluster) ConnectAll() (*Conn, <-chan Event, error) {
	return ts.ConnectAllTimeout(time.Second * 15)
}

func (ts *TestCluster) ConnectAllTimeout(sessionTimeout time.Duration) (*Conn, <-chan Event, error) {
	hosts := make([]string, len(ts.Servers))
	for i, srv := range ts.Servers {
		hosts[i] = fmt.Sprintf("127.0.0.1:%d", srv.Port)
	}
	zk, ch, err := Connect(hosts, sessionTimeout)
	return zk, ch, err
}

func (ts *TestCluster) Stop() error {
	for _, srv := range ts.Servers {
		srv.Srv.Stop()
	}
	defer os.RemoveAll(ts.Path)
	return nil
}
