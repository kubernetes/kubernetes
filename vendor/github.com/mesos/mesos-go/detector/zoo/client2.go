package zoo

import (
	"sync"
	"time"

	"github.com/samuel/go-zookeeper/zk"
)

const (
	defaultSessionTimeout = 60 * time.Second
	CurrentPath           = "."
)

var zkSessionTimeout = defaultSessionTimeout

type client2 struct {
	*zk.Conn
	path     string
	done     chan struct{} // signal chan, closes when the underlying connection terminates
	stopOnce sync.Once
}

func connect2(hosts []string, path string) (*client2, error) {
	c, ev, err := zk.Connect(hosts, zkSessionTimeout)
	if err != nil {
		return nil, err
	}
	done := make(chan struct{})
	go func() {
		// close the 'done' chan when the zk event chan closes (signals termination of zk connection)
		defer close(done)
		for {
			if _, ok := <-ev; !ok {
				return
			}
		}
	}()
	return &client2{
		Conn: c,
		path: path,
		done: done,
	}, nil
}

func (c *client2) Stopped() <-chan struct{} {
	return c.done
}

func (c *client2) Stop() {
	c.stopOnce.Do(c.Close)
}

func (c *client2) Data(path string) (data []byte, err error) {
	data, _, err = c.Get(path)
	return
}

func (c *client2) WatchChildren(path string) (string, <-chan []string, <-chan error) {
	errCh := make(chan error, 1)
	snap := make(chan []string)

	watchPath := c.path
	if path != "" && path != CurrentPath {
		watchPath = watchPath + path
	}
	go func() {
		defer close(errCh)
		for {
			children, _, ev, err := c.ChildrenW(watchPath)
			if err != nil {
				errCh <- err
				return
			}
			select {
			case snap <- children:
			case <-c.done:
				return
			}
			e := <-ev // wait for the next watch-related event
			if e.Err != nil {
				errCh <- e.Err
				return
			}
		}
	}()
	return watchPath, snap, errCh
}
