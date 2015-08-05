package zoo

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	log "github.com/golang/glog"
	"github.com/samuel/go-zookeeper/zk"
)

const (
	defaultSessionTimeout   = 60 * time.Second
	defaultReconnectTimeout = 5 * time.Second
	currentPath             = "."
	defaultRewatchDelay     = 200 * time.Millisecond
)

type stateType int32

const (
	disconnectedState stateType = iota
	connectionRequestedState
	connectionAttemptState
	connectedState
)

func (s stateType) String() string {
	switch s {
	case disconnectedState:
		return "DISCONNECTED"
	case connectionRequestedState:
		return "REQUESTED"
	case connectionAttemptState:
		return "ATTEMPT"
	case connectedState:
		return "CONNECTED"
	default:
		panic(fmt.Sprintf("unrecognized state: %d", int32(s)))
	}
}

type Client struct {
	conn           Connector
	defaultFactory Factory
	factory        Factory // must never be nil, use setFactory to update
	state          stateType
	reconnCount    uint64
	reconnDelay    time.Duration
	rootPath       string
	errorHandler   ErrorHandler // must never be nil
	connectOnce    sync.Once
	stopOnce       sync.Once
	shouldStop     chan struct{} // signal chan
	shouldReconn   chan struct{} // message chan
	connLock       sync.Mutex
	hasConnected   chan struct{} // message chan
	rewatchDelay   time.Duration
}

func newClient(hosts []string, path string) (*Client, error) {
	zkc := &Client{
		reconnDelay:  defaultReconnectTimeout,
		rewatchDelay: defaultRewatchDelay,
		rootPath:     path,
		shouldStop:   make(chan struct{}),
		shouldReconn: make(chan struct{}, 1),
		hasConnected: make(chan struct{}, 1),
		errorHandler: ErrorHandler(func(*Client, error) {}),
		defaultFactory: asFactory(func() (Connector, <-chan zk.Event, error) {
			return zk.Connect(hosts, defaultSessionTimeout)
		}),
	}
	zkc.setFactory(zkc.defaultFactory)
	// TODO(vlad): validate  URIs
	return zkc, nil
}

func (zkc *Client) setFactory(f Factory) {
	if f == nil {
		f = zkc.defaultFactory
	}
	zkc.factory = asFactory(func() (c Connector, ch <-chan zk.Event, err error) {
		select {
		case <-zkc.shouldStop:
			err = errors.New("client stopping")
		default:
			zkc.connLock.Lock()
			defer zkc.connLock.Unlock()
			if zkc.conn != nil {
				zkc.conn.Close()
			}
			c, ch, err = f.create()
			zkc.conn = c
		}
		return
	})
}

// return true only if the client's state was changed from `from` to `to`
func (zkc *Client) stateChange(from, to stateType) (result bool) {
	defer func() {
		log.V(3).Infof("stateChange: from=%v to=%v result=%v", from, to, result)
	}()
	result = atomic.CompareAndSwapInt32((*int32)(&zkc.state), int32(from), int32(to))
	return
}

// connect to zookeeper, blocks on the initial call to doConnect()
func (zkc *Client) connect() {
	select {
	case <-zkc.shouldStop:
		return
	default:
		zkc.connectOnce.Do(func() {
			if zkc.stateChange(disconnectedState, connectionRequestedState) {
				if err := zkc.doConnect(); err != nil {
					log.Error(err)
					zkc.errorHandler(zkc, err)
				}
			}
			go func() {
				for {
					select {
					case <-zkc.shouldStop:
						zkc.connLock.Lock()
						defer zkc.connLock.Unlock()
						if zkc.conn != nil {
							zkc.conn.Close()
						}
						return
					case <-zkc.shouldReconn:
						if err := zkc.reconnect(); err != nil {
							log.Error(err)
							zkc.errorHandler(zkc, err)
						}
					}
				}
			}()
		})
	}
	return
}

// attempt to reconnect to zookeeper. will ignore attempts to reconnect
// if not disconnected. if reconnection is attempted then this func will block
// for at least reconnDelay before actually attempting to connect to zookeeper.
func (zkc *Client) reconnect() error {
	if !zkc.stateChange(disconnectedState, connectionRequestedState) {
		log.V(4).Infoln("Ignoring reconnect, currently connected/connecting.")
		return nil
	}

	defer func() { zkc.reconnCount++ }()

	log.V(4).Infoln("Delaying reconnection for ", zkc.reconnDelay)
	<-time.After(zkc.reconnDelay)

	return zkc.doConnect()
}

func (zkc *Client) doConnect() error {
	if !zkc.stateChange(connectionRequestedState, connectionAttemptState) {
		log.V(4).Infoln("aborting doConnect, connection attempt already in progress or else disconnected")
		return nil
	}

	// if we're not connected by the time we return then we failed.
	defer func() {
		zkc.stateChange(connectionAttemptState, disconnectedState)
	}()

	// create Connector instance
	conn, sessionEvents, err := zkc.factory.create()
	if err != nil {
		// once the factory stops producing connectors, it's time to stop
		zkc.stop()
		return err
	}

	zkc.connLock.Lock()
	zkc.conn = conn
	zkc.connLock.Unlock()

	log.V(4).Infof("Created connection object of type %T\n", conn)
	connected := make(chan struct{})
	sessionExpired := make(chan struct{})
	go func() {
		defer close(sessionExpired)
		zkc.monitorSession(sessionEvents, connected)
	}()

	// wait for connected confirmation
	select {
	case <-connected:
		if !zkc.stateChange(connectionAttemptState, connectedState) {
			log.V(4).Infoln("failed to transition to connected state")
			// we could be:
			// - disconnected        ... reconnect() will try to connect again, otherwise;
			// - connected           ... another goroutine already established a connection
			// - connectionRequested ... another goroutine is already trying to connect
			zkc.requestReconnect()
		}
		log.Infoln("zookeeper client connected")
	case <-sessionExpired:
		// connection was disconnected before it was ever really 'connected'
		if !zkc.stateChange(connectionAttemptState, disconnectedState) {
			//programming error
			panic("failed to transition from connection-attempt to disconnected state")
		}
		zkc.requestReconnect()
	case <-zkc.shouldStop:
		// noop
	}
	return nil
}

// signal for reconnect unless we're shutting down
func (zkc *Client) requestReconnect() {
	select {
	case <-zkc.shouldStop:
		// abort reconnect request, client is shutting down
	default:
		select {
		case zkc.shouldReconn <- struct{}{}:
			// reconnect request successful
		default:
			// reconnect chan is full: reconnect has already
			// been requested. move on.
		}
	}
}

// monitor a zookeeper session event channel, closes the 'connected' channel once
// a zookeeper connection has been established. errors are forwarded to the client's
// errorHandler. the closing of the sessionEvents chan triggers a call to client.onDisconnected.
// this func blocks until either the client's shouldStop or sessionEvents chan are closed.
func (zkc *Client) monitorSession(sessionEvents <-chan zk.Event, connected chan struct{}) {
	firstConnected := true
	for {
		select {
		case <-zkc.shouldStop:
			return
		case e, ok := <-sessionEvents:
			if !ok {
				// once sessionEvents is closed, the embedded ZK client will
				// no longer attempt to reconnect.
				zkc.onDisconnected()
				return
			} else if e.Err != nil {
				log.Errorf("received state error: %s", e.Err.Error())
				zkc.errorHandler(zkc, e.Err)
			}
			switch e.State {
			case zk.StateConnecting:
				log.Infoln("connecting to zookeeper..")

			case zk.StateConnected:
				log.V(2).Infoln("received StateConnected")
				if firstConnected {
					close(connected) // signal session listener
					firstConnected = false
				}
				// let any listeners know about the change
				select {
				case <-zkc.shouldStop: // noop
				case zkc.hasConnected <- struct{}{}: // noop
				default: // message buf full, this becomes a non-blocking noop
				}

			case zk.StateDisconnected:
				log.Infoln("zookeeper client disconnected")

			case zk.StateExpired:
				log.Infoln("zookeeper client session expired")
			}
		}
	}
}

// watch the child nodes for changes, at the specified path.
// callers that specify a path of `currentPath` will watch the currently set rootPath,
// otherwise the watchedPath is calculated as rootPath+path.
// this func spawns a go routine to actually do the watching, and so returns immediately.
// in the absense of errors a signalling channel is returned that will close
// upon the termination of the watch (e.g. due to disconnection).
func (zkc *Client) watchChildren(path string, watcher ChildWatcher) (<-chan struct{}, error) {
	watchPath := zkc.rootPath
	if path != "" && path != currentPath {
		watchPath = watchPath + path
	}

	log.V(2).Infoln("Watching children for path", watchPath)
	watchEnded := make(chan struct{})
	go func() {
		defer close(watchEnded)
		zkc._watchChildren(watchPath, watcher)
	}()
	return watchEnded, nil
}

// continuation of watchChildren. blocks until either underlying zk connector terminates, or else this
// client is shut down. continuously renews child watches.
func (zkc *Client) _watchChildren(watchPath string, watcher ChildWatcher) {
	watcher(zkc, watchPath) // prime the listener
	var zkevents <-chan zk.Event
	var err error
	first := true
	for {
		// we really only expect this to happen when zk session has expired,
		// give the connection a little time to re-establish itself
		for {
			//TODO(jdef) it would be better if we could listen for broadcast Connection/Disconnection events,
			//emitted whenever the embedded client cycles (read: when the connection state of this client changes).
			//As it currently stands, if the embedded client cycles fast enough, we may actually not notice it here
			//and keep on watching like nothing bad happened.
			if !zkc.isConnected() {
				log.Warningf("no longer connected to server, exiting child watch")
				return
			}
			if first {
				first = false
			} else {
				select {
				case <-zkc.shouldStop:
					return
				case <-time.After(zkc.rewatchDelay):
				}
			}
			_, _, zkevents, err = zkc.conn.ChildrenW(watchPath)
			if err == nil {
				log.V(2).Infoln("rewatching children for path", watchPath)
				break
			}
			log.V(1).Infof("unable to watch children for path %s: %s", watchPath, err.Error())
			zkc.errorHandler(zkc, err)
		}
		// zkevents is (at most) a one-trick channel
		// (a) a child event happens (no error)
		// (b) the embedded client is shutting down (zk.ErrClosing)
		// (c) the zk session expires (zk.ErrSessionExpired)
		select {
		case <-zkc.shouldStop:
			return
		case e, ok := <-zkevents:
			if !ok {
				log.Warningf("expected a single zk event before channel close")
				break // the select
			}
			switch e.Type {
			//TODO(jdef) should we not also watch for EventNode{Created,Deleted,DataChanged}?
			case zk.EventNodeChildrenChanged:
				log.V(2).Infoln("Handling: zk.EventNodeChildrenChanged")
				watcher(zkc, e.Path)
				continue
			default:
				if e.Err != nil {
					zkc.errorHandler(zkc, e.Err)
					if e.Type == zk.EventNotWatching && e.State == zk.StateDisconnected {
						if e.Err == zk.ErrClosing {
							log.V(1).Infof("watch invalidated, embedded client terminating")
							return
						}
						log.V(1).Infof("watch invalidated, attempting to watch again: %v", e.Err)
					} else {
						log.Warningf("received error while watching path %s: %s", watchPath, e.Err.Error())
					}
				}
			}
		}
	}
}

func (zkc *Client) onDisconnected() {
	if st := zkc.getState(); st == connectedState && zkc.stateChange(st, disconnectedState) {
		log.Infoln("disconnected from the server, reconnecting...")
		zkc.requestReconnect()
		return
	}
}

// return a channel that gets an empty struct every time a connection happens
func (zkc *Client) connections() <-chan struct{} {
	return zkc.hasConnected
}

func (zkc *Client) getState() stateType {
	return stateType(atomic.LoadInt32((*int32)(&zkc.state)))
}

// convenience function
func (zkc *Client) isConnected() bool {
	return zkc.getState() == connectedState
}

// convenience function
func (zkc *Client) isConnecting() bool {
	state := zkc.getState()
	return state == connectionRequestedState || state == connectionAttemptState
}

// convenience function
func (zkc *Client) isDisconnected() bool {
	return zkc.getState() == disconnectedState
}

func (zkc *Client) list(path string) ([]string, error) {
	if !zkc.isConnected() {
		return nil, errors.New("Unable to list children, client not connected.")
	}

	children, _, err := zkc.conn.Children(path)
	if err != nil {
		return nil, err
	}

	return children, nil
}

func (zkc *Client) data(path string) ([]byte, error) {
	if !zkc.isConnected() {
		return nil, errors.New("Unable to retrieve node data, client not connected.")
	}

	data, _, err := zkc.conn.Get(path)
	if err != nil {
		return nil, err
	}

	return data, nil
}

func (zkc *Client) stop() {
	zkc.stopOnce.Do(func() {
		close(zkc.shouldStop)
	})
}

// when this channel is closed the client is either stopping, or has stopped
func (zkc *Client) stopped() <-chan struct{} {
	return zkc.shouldStop
}
