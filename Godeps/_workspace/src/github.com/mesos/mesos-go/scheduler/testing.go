package scheduler

import (
	"github.com/gogo/protobuf/proto"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/upid"
	"golang.org/x/net/context"
)

type TestDriver struct {
	*MesosSchedulerDriver
}

func (t *TestDriver) SetConnected(b bool) {
	t.eventLock.Lock()
	defer t.eventLock.Unlock()
	t.connected = b
}

func (t *TestDriver) Started() <-chan struct{} {
	return t.started
}

func (t *TestDriver) Stopped() <-chan struct{} {
	return t.stopCh
}

func (t *TestDriver) Done() <-chan struct{} {
	return t.done
}

func (t *TestDriver) Framework() *mesos.FrameworkInfo {
	return t.frameworkInfo
}

func (t *TestDriver) UPID() *upid.UPID {
	return t.self
}

func (t *TestDriver) MasterPID() *upid.UPID {
	return t.masterPid
}

func (t *TestDriver) Fatal(ctx context.Context, msg string) {
	t.eventLock.Lock()
	defer t.eventLock.Unlock()
	t.fatal(ctx, msg)
}

func (t *TestDriver) OnDispatch(f func(ctx context.Context, upid *upid.UPID, msg proto.Message) error) {
	t.dispatch = f
}

func (t *TestDriver) HandleMasterChanged(ctx context.Context, from *upid.UPID, msg proto.Message) {
	t.eventLock.Lock()
	defer t.eventLock.Unlock()
	t.handleMasterChanged(ctx, from, msg)
}

func (t *TestDriver) CacheOffer(offer *mesos.Offer, pid *upid.UPID) {
	t.cache.putOffer(offer, pid)
}

func (t *TestDriver) Context() context.Context {
	return t.context()
}

func (t *TestDriver) FrameworkRegistered(ctx context.Context, from *upid.UPID, msg proto.Message) {
	t.eventLock.Lock()
	defer t.eventLock.Unlock()
	t.frameworkRegistered(ctx, from, msg)
}

func (t *TestDriver) FrameworkReregistered(ctx context.Context, from *upid.UPID, msg proto.Message) {
	t.eventLock.Lock()
	defer t.eventLock.Unlock()
	t.frameworkReregistered(ctx, from, msg)
}
