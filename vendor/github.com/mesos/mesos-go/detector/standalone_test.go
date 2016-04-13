package detector

import (
	"sync"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/upid"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

const (
	localhost = uint32(2130706433) // packed uint32 for 127.0.0.1 IPv4
)

func TestStandalone_nil(t *testing.T) {
	d := NewStandalone(nil)
	select {
	case <-d.Done(): // expected
		t.Fatalf("expected detector to stay alive since we haven't done anything with it")
	case <-time.After(500 * time.Millisecond):
	}
	d.Detect(nil)
	select {
	case <-d.Done(): // expected
	case <-time.After(1 * time.Second):
		t.Fatalf("expected detector to shutdown since it has no master")
	}
}

func TestStandalone_pollerIncompleteInfo(t *testing.T) {
	d := NewStandalone(&mesos.MasterInfo{})
	f := fetcherFunc(func(context.Context, string) (*upid.UPID, error) {
		return nil, nil
	})
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		d.poller(f)
	}()
	select {
	case <-ch: // expected
	case <-time.After(1 * time.Second):
		t.Fatalf("expected poller to shutdown since master info is incomplete")
	}
	select {
	case <-d.Done(): // expected
	case <-time.After(1 * time.Second):
		t.Fatalf("expected detector to shutdown since it has no master")
	}
}

func TestStandalone_pollerFetched(t *testing.T) {
	assert := assert.New(t)
	// presence of IP address allows fecher to be called
	d := NewStandalone(&mesos.MasterInfo{Ip: proto.Uint32(localhost)})
	defer d.Cancel()

	fetched := make(chan struct{})
	pid := &upid.UPID{
		ID:   "foo@127.0.0.1:5050",
		Host: "127.0.0.1",
		Port: "5050",
	}
	f := fetcherFunc(func(ctx context.Context, addr string) (*upid.UPID, error) {
		defer close(fetched)
		assert.Equal("127.0.0.1:5050", addr)
		return pid, nil
	})

	go d.poller(f)

	// fetch called
	select {
	case <-fetched: // expected
	case <-time.After(1 * time.Second):
		t.Fatalf("expected fetch")
	}

	// read MasterInfo
	select {
	case mi := <-d.ch:
		assert.Equal(mi, CreateMasterInfo(pid))
	case <-time.After(1 * time.Second):
		t.Fatalf("expected poller to send master info")
	}
}

func TestStandalone_pollerFetchedMulti(t *testing.T) {
	assert := assert.New(t)
	// presence of IP address allows fecher to be called
	d := NewStandalone(&mesos.MasterInfo{Ip: proto.Uint32(localhost)})
	defer d.Cancel()
	d.leaderSyncInterval = 500 * time.Millisecond

	i := 0
	var wg sync.WaitGroup
	wg.Add(4)
	f := fetcherFunc(func(ctx context.Context, addr string) (*upid.UPID, error) {
		defer func() { i++ }()
		switch i {
		case 0:
			wg.Done()
			assert.Equal("127.0.0.1:5050", addr)
			return &upid.UPID{ID: "foo@127.0.0.1:5050", Host: "127.0.0.1", Port: "5050"}, nil
		case 1:
			wg.Done()
			assert.Equal("127.0.0.1:5050", addr)
			return &upid.UPID{ID: "foo@127.0.0.2:5050", Host: "127.0.0.2", Port: "5050"}, nil
		case 2:
			wg.Done()
			return nil, context.DeadlineExceeded
		case 3:
			wg.Done()
			assert.Equal("127.0.0.1:5050", addr)
			return &upid.UPID{ID: "foo@127.0.0.3:5050", Host: "127.0.0.3", Port: "5050"}, nil
		default:
			d.Cancel()
			return nil, context.Canceled
		}
	})

	go d.poller(f)

	// fetches complete
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		wg.Wait()
	}()

	changed := make(chan struct{})
	go func() {
		defer close(changed)
		for i := 0; i < 4; i++ {
			if mi, ok := <-d.ch; !ok {
				t.Fatalf("failed to read master info on cycle %v", i)
				break
			} else {
				switch i {
				case 0:
					assert.Equal(CreateMasterInfo(&upid.UPID{ID: "foo@127.0.0.1:5050", Host: "127.0.0.1", Port: "5050"}), mi)
				case 1:
					assert.Equal(CreateMasterInfo(&upid.UPID{ID: "foo@127.0.0.2:5050", Host: "127.0.0.2", Port: "5050"}), mi)
				case 2:
					assert.Nil(mi)
				case 3:
					assert.Equal(CreateMasterInfo(&upid.UPID{ID: "foo@127.0.0.3:5050", Host: "127.0.0.3", Port: "5050"}), mi)
				}
			}
		}
	}()

	started := time.Now()
	select {
	case <-ch: // expected
	case <-time.After(3 * time.Second):
		t.Fatalf("expected fetches all complete")
	}

	select {
	case <-changed: // expected
	case <-time.After((3 * time.Second) - time.Now().Sub(started)):
		t.Fatalf("expected to have received all master info changes")
	}
}
