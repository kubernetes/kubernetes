package zoo

import (
	"github.com/samuel/go-zookeeper/zk"
)

// Connector Interface to facade zk.Conn type
// since github.com/samuel/go-zookeeper/zk does not provide an interface
// for the zk.Conn object, this allows for mocking and easier testing.
type Connector interface {
	Close()
	Children(string) ([]string, *zk.Stat, error)
	ChildrenW(string) ([]string, *zk.Stat, <-chan zk.Event, error)
	Get(string) ([]byte, *zk.Stat, error)
}

//Factory is an adapter to trap the creation of zk.Conn instances
//since the official zk API does not expose an interface for zk.Conn.
type Factory interface {
	create() (Connector, <-chan zk.Event, error)
}

type asFactory func() (Connector, <-chan zk.Event, error)

func (f asFactory) create() (Connector, <-chan zk.Event, error) {
	return f()
}
