package executor

import (
	"github.com/gogo/protobuf/proto"
	"github.com/mesos/mesos-go/messenger"
	"github.com/mesos/mesos-go/upid"
	"golang.org/x/net/context"
)

type TestDriver struct {
	*MesosExecutorDriver
}

func (e *TestDriver) SetConnected(b bool) {
	e.guarded(func() {
		e.connected = b
	})
}

func (e *TestDriver) SetMessenger(m messenger.Messenger) {
	e.messenger = m
}

func (e *TestDriver) Started() <-chan struct{} {
	return e.started
}

func (e *TestDriver) guarded(f func()) {
	e.lock.Lock()
	defer e.lock.Unlock()
	f()
}

func (e *TestDriver) Context() context.Context {
	return e.context()
}

func (e *TestDriver) StatusUpdateAcknowledgement(ctx context.Context, from *upid.UPID, msg proto.Message) {
	e.guarded(func() {
		e.statusUpdateAcknowledgement(ctx, from, msg)
	})
}
