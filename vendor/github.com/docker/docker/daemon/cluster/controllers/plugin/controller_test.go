package plugin

import (
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/docker/distribution/reference"
	enginetypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/swarm/runtime"
	"github.com/docker/docker/pkg/pubsub"
	"github.com/docker/docker/plugin"
	"github.com/docker/docker/plugin/v2"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

const (
	pluginTestName          = "test"
	pluginTestRemote        = "testremote"
	pluginTestRemoteUpgrade = "testremote2"
)

func TestPrepare(t *testing.T) {
	b := newMockBackend()
	c := newTestController(b, false)
	ctx := context.Background()

	if err := c.Prepare(ctx); err != nil {
		t.Fatal(err)
	}

	if b.p == nil {
		t.Fatal("pull not performed")
	}

	c = newTestController(b, false)
	if err := c.Prepare(ctx); err != nil {
		t.Fatal(err)
	}
	if b.p == nil {
		t.Fatal("unexpected nil")
	}
	if b.p.PluginObj.PluginReference != pluginTestRemoteUpgrade {
		t.Fatal("upgrade not performed")
	}

	c = newTestController(b, false)
	c.serviceID = "1"
	if err := c.Prepare(ctx); err == nil {
		t.Fatal("expected error on prepare")
	}
}

func TestStart(t *testing.T) {
	b := newMockBackend()
	c := newTestController(b, false)
	ctx := context.Background()

	if err := c.Prepare(ctx); err != nil {
		t.Fatal(err)
	}

	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}

	if !b.p.IsEnabled() {
		t.Fatal("expected plugin to be enabled")
	}

	c = newTestController(b, true)
	if err := c.Prepare(ctx); err != nil {
		t.Fatal(err)
	}
	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}
	if b.p.IsEnabled() {
		t.Fatal("expected plugin to be disabled")
	}

	c = newTestController(b, false)
	if err := c.Prepare(ctx); err != nil {
		t.Fatal(err)
	}
	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}
	if !b.p.IsEnabled() {
		t.Fatal("expected plugin to be enabled")
	}
}

func TestWaitCancel(t *testing.T) {
	b := newMockBackend()
	c := newTestController(b, true)
	ctx := context.Background()
	if err := c.Prepare(ctx); err != nil {
		t.Fatal(err)
	}
	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}

	ctxCancel, cancel := context.WithCancel(ctx)
	chErr := make(chan error)
	go func() {
		chErr <- c.Wait(ctxCancel)
	}()
	cancel()
	select {
	case err := <-chErr:
		if err != context.Canceled {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for cancelation")
	}
}

func TestWaitDisabled(t *testing.T) {
	b := newMockBackend()
	c := newTestController(b, true)
	ctx := context.Background()
	if err := c.Prepare(ctx); err != nil {
		t.Fatal(err)
	}
	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}

	chErr := make(chan error)
	go func() {
		chErr <- c.Wait(ctx)
	}()

	if err := b.Enable("test", nil); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-chErr:
		if err == nil {
			t.Fatal("expected error")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for event")
	}

	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}

	ctxWaitReady, cancelCtxWaitReady := context.WithTimeout(ctx, 30*time.Second)
	c.signalWaitReady = cancelCtxWaitReady
	defer cancelCtxWaitReady()

	go func() {
		chErr <- c.Wait(ctx)
	}()

	chEvent, cancel := b.SubscribeEvents(1)
	defer cancel()

	if err := b.Disable("test", nil); err != nil {
		t.Fatal(err)
	}

	select {
	case <-chEvent:
		<-ctxWaitReady.Done()
		if err := ctxWaitReady.Err(); err == context.DeadlineExceeded {
			t.Fatal(err)
		}
		select {
		case <-chErr:
			t.Fatal("wait returned unexpectedly")
		default:
			// all good
		}
	case <-chErr:
		t.Fatal("wait returned unexpectedly")
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for event")
	}

	if err := b.Remove("test", nil); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-chErr:
		if err == nil {
			t.Fatal("expected error")
		}
		if !strings.Contains(err.Error(), "removed") {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for event")
	}
}

func TestWaitEnabled(t *testing.T) {
	b := newMockBackend()
	c := newTestController(b, false)
	ctx := context.Background()
	if err := c.Prepare(ctx); err != nil {
		t.Fatal(err)
	}
	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}

	chErr := make(chan error)
	go func() {
		chErr <- c.Wait(ctx)
	}()

	if err := b.Disable("test", nil); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-chErr:
		if err == nil {
			t.Fatal("expected error")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for event")
	}

	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}

	ctxWaitReady, ctxWaitCancel := context.WithCancel(ctx)
	c.signalWaitReady = ctxWaitCancel
	defer ctxWaitCancel()

	go func() {
		chErr <- c.Wait(ctx)
	}()

	chEvent, cancel := b.SubscribeEvents(1)
	defer cancel()

	if err := b.Enable("test", nil); err != nil {
		t.Fatal(err)
	}

	select {
	case <-chEvent:
		<-ctxWaitReady.Done()
		if err := ctxWaitReady.Err(); err == context.DeadlineExceeded {
			t.Fatal(err)
		}
		select {
		case <-chErr:
			t.Fatal("wait returned unexpectedly")
		default:
			// all good
		}
	case <-chErr:
		t.Fatal("wait returned unexpectedly")
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for event")
	}

	if err := b.Remove("test", nil); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-chErr:
		if err == nil {
			t.Fatal("expected error")
		}
		if !strings.Contains(err.Error(), "removed") {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for event")
	}
}

func TestRemove(t *testing.T) {
	b := newMockBackend()
	c := newTestController(b, false)
	ctx := context.Background()

	if err := c.Prepare(ctx); err != nil {
		t.Fatal(err)
	}
	if err := c.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}

	c2 := newTestController(b, false)
	if err := c2.Prepare(ctx); err != nil {
		t.Fatal(err)
	}

	if err := c.Remove(ctx); err != nil {
		t.Fatal(err)
	}
	if b.p == nil {
		t.Fatal("plugin removed unexpectedly")
	}
	if err := c2.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if err := c2.Remove(ctx); err != nil {
		t.Fatal(err)
	}
	if b.p != nil {
		t.Fatal("expected plugin to be removed")
	}
}

func newTestController(b Backend, disabled bool) *Controller {
	return &Controller{
		logger:  &logrus.Entry{Logger: &logrus.Logger{Out: ioutil.Discard}},
		backend: b,
		spec: runtime.PluginSpec{
			Name:     pluginTestName,
			Remote:   pluginTestRemote,
			Disabled: disabled,
		},
	}
}

func newMockBackend() *mockBackend {
	return &mockBackend{
		pub: pubsub.NewPublisher(0, 0),
	}
}

type mockBackend struct {
	p   *v2.Plugin
	pub *pubsub.Publisher
}

func (m *mockBackend) Disable(name string, config *enginetypes.PluginDisableConfig) error {
	m.p.PluginObj.Enabled = false
	m.pub.Publish(plugin.EventDisable{})
	return nil
}

func (m *mockBackend) Enable(name string, config *enginetypes.PluginEnableConfig) error {
	m.p.PluginObj.Enabled = true
	m.pub.Publish(plugin.EventEnable{})
	return nil
}

func (m *mockBackend) Remove(name string, config *enginetypes.PluginRmConfig) error {
	m.p = nil
	m.pub.Publish(plugin.EventRemove{})
	return nil
}

func (m *mockBackend) Pull(ctx context.Context, ref reference.Named, name string, metaHeaders http.Header, authConfig *enginetypes.AuthConfig, privileges enginetypes.PluginPrivileges, outStream io.Writer, opts ...plugin.CreateOpt) error {
	m.p = &v2.Plugin{
		PluginObj: enginetypes.Plugin{
			ID:              "1234",
			Name:            name,
			PluginReference: ref.String(),
		},
	}
	return nil
}

func (m *mockBackend) Upgrade(ctx context.Context, ref reference.Named, name string, metaHeaders http.Header, authConfig *enginetypes.AuthConfig, privileges enginetypes.PluginPrivileges, outStream io.Writer) error {
	m.p.PluginObj.PluginReference = pluginTestRemoteUpgrade
	return nil
}

func (m *mockBackend) Get(name string) (*v2.Plugin, error) {
	if m.p == nil {
		return nil, errors.New("not found")
	}
	return m.p, nil
}

func (m *mockBackend) SubscribeEvents(buffer int, events ...plugin.Event) (eventCh <-chan interface{}, cancel func()) {
	ch := m.pub.SubscribeTopicWithBuffer(nil, buffer)
	cancel = func() { m.pub.Evict(ch) }
	return ch, cancel
}
