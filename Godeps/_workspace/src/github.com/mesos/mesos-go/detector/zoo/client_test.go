package zoo

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	util "github.com/mesos/mesos-go/mesosutil"
	"github.com/samuel/go-zookeeper/zk"
	"github.com/stretchr/testify/assert"
)

var test_zk_hosts = []string{"localhost:2181"}

const (
	test_zk_path = "/test"
)

func TestClientNew(t *testing.T) {
	path := "/mesos"
	chEvent := make(chan zk.Event)
	connector := makeMockConnector(path, chEvent)

	c, err := newClient(test_zk_hosts, path)
	assert.NoError(t, err)
	assert.NotNil(t, c)
	assert.False(t, c.isConnected())
	c.conn = connector

}

// This test requires zookeeper to be running.
// You must also set env variable ZK_HOSTS to point to zk hosts.
// The zk package does not offer a way to mock its connection function.
func TestClientConnectIntegration(t *testing.T) {
	if os.Getenv("ZK_HOSTS") == "" {
		t.Skip("Skipping zk-server connection test: missing env ZK_HOSTS.")
	}
	hosts := strings.Split(os.Getenv("ZK_HOSTS"), ",")
	c, err := newClient(hosts, "/mesos")
	assert.NoError(t, err)
	c.errorHandler = ErrorHandler(func(c *Client, e error) {
		err = e
	})
	c.connect()
	assert.NoError(t, err)

	c.connect()
	assert.NoError(t, err)
	assert.True(t, c.isConnected())
}

func TestClientConnect(t *testing.T) {
	c, err := makeClient()
	assert.NoError(t, err)
	assert.False(t, c.isConnected())
	c.connect()
	assert.True(t, c.isConnected())
	assert.False(t, c.isConnecting())
}

func TestClient_FlappingConnection(t *testing.T) {
	c, err := newClient(test_zk_hosts, test_zk_path)
	c.reconnDelay = 10 * time.Millisecond // we don't want this test to take forever
	defer c.stop()

	assert.NoError(t, err)

	attempts := 0
	c.setFactory(asFactory(func() (Connector, <-chan zk.Event, error) {
		log.V(2).Infof("**** Using zk.Conn adapter ****")
		ch0 := make(chan zk.Event, 10) // session chan
		ch1 := make(chan zk.Event)     // watch chan
		go func() {
			if attempts > 1 {
				t.Fatalf("only one connector instance is expected")
			}
			attempts++
			for i := 0; i < 4; i++ {
				ch0 <- zk.Event{
					Type:  zk.EventSession,
					State: zk.StateConnecting,
					Path:  test_zk_path,
				}
				ch0 <- zk.Event{
					Type:  zk.EventSession,
					State: zk.StateConnected,
					Path:  test_zk_path,
				}
				time.Sleep(200 * time.Millisecond)
				ch0 <- zk.Event{
					Type:  zk.EventSession,
					State: zk.StateDisconnected,
					Path:  test_zk_path,
				}
			}
		}()
		connector := makeMockConnector(test_zk_path, ch1)
		return connector, ch0, nil
	}))

	go c.connect()
	time.Sleep(2 * time.Second)
	assert.True(t, c.isConnected())
	assert.Equal(t, 1, attempts)
}

func TestClientWatchChildren(t *testing.T) {
	c, err := makeClient()
	assert.NoError(t, err)
	c.errorHandler = ErrorHandler(func(c *Client, e error) {
		err = e
	})
	c.connect()
	assert.NoError(t, err)
	wCh := make(chan struct{}, 1)
	childrenWatcher := ChildWatcher(func(zkc *Client, path string) {
		log.V(4).Infoln("Path", path, "changed!")
		children, err := c.list(path)
		assert.NoError(t, err)
		assert.Equal(t, 3, len(children))
		assert.Equal(t, "info_0", children[0])
		assert.Equal(t, "info_5", children[1])
		assert.Equal(t, "info_10", children[2])
		wCh <- struct{}{}
	})

	_, err = c.watchChildren(currentPath, childrenWatcher)
	assert.NoError(t, err)

	select {
	case <-wCh:
	case <-time.After(time.Millisecond * 700):
		panic("Waited too long...")
	}
}

func TestClientWatchErrors(t *testing.T) {
	path := "/test"
	ch := make(chan zk.Event, 1)
	ch <- zk.Event{
		Type: zk.EventNotWatching,
		Err:  errors.New("Event Error"),
	}

	c, err := makeClient()
	c.state = connectedState

	assert.NoError(t, err)
	c.conn = makeMockConnector(path, (<-chan zk.Event)(ch))
	wCh := make(chan struct{}, 1)
	c.errorHandler = ErrorHandler(func(zkc *Client, err error) {
		assert.Error(t, err)
		wCh <- struct{}{}
	})

	c.watchChildren(currentPath, ChildWatcher(func(*Client, string) {}))

	select {
	case <-wCh:
	case <-time.After(time.Millisecond * 700):
		t.Fatalf("timed out waiting for error message")
	}

}

func TestWatchChildren_flappy(t *testing.T) {
	c, err := newClient(test_zk_hosts, test_zk_path)
	c.reconnDelay = 10 * time.Millisecond // we don't want this test to take forever

	assert.NoError(t, err)

	attempts := 0
	conn := NewMockConnector()
	defer func() {
		if !t.Failed() {
			conn.AssertExpectations(t)
		}
	}()
	defer func() {
		// stop client and give it time to shut down the connector
		c.stop()
		time.Sleep(100 * time.Millisecond)
	}()
	c.setFactory(asFactory(func() (Connector, <-chan zk.Event, error) {
		log.V(2).Infof("**** Using zk.Conn adapter ****")
		ch0 := make(chan zk.Event, 10) // session chan
		ch1 := make(chan zk.Event)     // watch chan
		go func() {
			if attempts > 1 {
				t.Fatalf("only one connector instance is expected")
			}
			attempts++
			for i := 0; i < 4; i++ {
				ch0 <- zk.Event{
					Type:  zk.EventSession,
					State: zk.StateConnecting,
					Path:  test_zk_path,
				}
				ch0 <- zk.Event{
					Type:  zk.EventSession,
					State: zk.StateConnected,
					Path:  test_zk_path,
				}
				time.Sleep(200 * time.Millisecond)
				ch0 <- zk.Event{
					Type:  zk.EventSession,
					State: zk.StateDisconnected,
					Path:  test_zk_path,
				}
			}
			ch0 <- zk.Event{
				Type:  zk.EventSession,
				State: zk.StateConnecting,
				Path:  test_zk_path,
			}
			ch0 <- zk.Event{
				Type:  zk.EventSession,
				State: zk.StateConnected,
				Path:  test_zk_path,
			}
			ch1 <- zk.Event{
				Type: zk.EventNodeChildrenChanged,
				Path: test_zk_path,
			}
		}()
		simulatedErr := errors.New("simulated watch error")
		conn.On("ChildrenW", test_zk_path).Return(nil, nil, nil, simulatedErr).Times(4)
		conn.On("ChildrenW", test_zk_path).Return([]string{test_zk_path}, &zk.Stat{}, (<-chan zk.Event)(ch1), nil)
		conn.On("Close").Return(nil)
		return conn, ch0, nil
	}))

	go c.connect()
	var watchChildrenCount uint64
	watcherFunc := ChildWatcher(func(zkc *Client, path string) {
		log.V(1).Infof("ChildWatcher invoked %d", atomic.LoadUint64(&watchChildrenCount))
	})
	startTime := time.Now()
	endTime := startTime.Add(2 * time.Second)
watcherLoop:
	for time.Now().Before(endTime) {
		log.V(1).Infof("entered watcherLoop")
		select {
		case <-c.connections():
			log.V(1).Infof("invoking watchChildren")
			if _, err := c.watchChildren(currentPath, watcherFunc); err == nil {
				// watching children succeeded!!
				t.Logf("child watch success")
				atomic.AddUint64(&watchChildrenCount, 1)
			} else {
				// setting the watch failed
				t.Logf("setting child watch failed: %v", err)
				continue watcherLoop
			}
		case <-c.stopped():
			t.Logf("detected client termination")
			break watcherLoop
		case <-time.After(endTime.Sub(time.Now())):
		}
	}

	wantChildrenCount := atomic.LoadUint64(&watchChildrenCount)
	assert.Equal(t, uint64(5), wantChildrenCount, "expected watchChildrenCount = 5 instead of %d, should be reinvoked upon initial ChildrenW failures", wantChildrenCount)
}

func makeClient() (*Client, error) {
	ch0 := make(chan zk.Event, 2)
	ch1 := make(chan zk.Event, 1)

	ch0 <- zk.Event{
		State: zk.StateConnected,
		Path:  test_zk_path,
	}
	ch1 <- zk.Event{
		Type: zk.EventNodeChildrenChanged,
		Path: test_zk_path,
	}
	go func() {
		time.Sleep(1 * time.Second)
		ch0 <- zk.Event{
			State: zk.StateDisconnected,
		}
		close(ch0)
		close(ch1)
	}()

	c, err := newClient(test_zk_hosts, test_zk_path)
	if err != nil {
		return nil, err
	}

	// only allow a single connection
	first := true
	c.setFactory(asFactory(func() (Connector, <-chan zk.Event, error) {
		if !first {
			return nil, nil, errors.New("only a single connection attempt allowed for mock connector")
		} else {
			first = false
		}
		log.V(2).Infof("**** Using zk.Conn adapter ****")
		connector := makeMockConnector(test_zk_path, ch1)
		return connector, ch0, nil
	}))

	return c, nil
}

func makeMockConnector(path string, chEvent <-chan zk.Event) *MockConnector {
	log.V(2).Infoln("Making Connector mock.")
	conn := NewMockConnector()
	conn.On("Close").Return(nil)
	conn.On("ChildrenW", path).Return([]string{path}, &zk.Stat{}, chEvent, nil)
	conn.On("Children", path).Return([]string{"info_0", "info_5", "info_10"}, &zk.Stat{}, nil)
	conn.On("Get", fmt.Sprintf("%s/info_0", path)).Return(makeTestMasterInfo(), &zk.Stat{}, nil)

	return conn
}

func newTestMasterInfo(id int) []byte {
	miPb := util.NewMasterInfo(fmt.Sprintf("master(%d)@localhost:5050", id), 123456789, 400)
	data, err := proto.Marshal(miPb)
	if err != nil {
		panic(err)
	}
	return data
}

func makeTestMasterInfo() []byte {
	miPb := util.NewMasterInfo("master@localhost:5050", 123456789, 400)
	data, err := proto.Marshal(miPb)
	if err != nil {
		panic(err)
	}
	return data
}
