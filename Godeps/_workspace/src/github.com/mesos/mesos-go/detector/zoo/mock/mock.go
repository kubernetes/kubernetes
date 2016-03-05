package mock

import (
	"sync"

	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/detector/zoo"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/stretchr/testify/mock"
)

type Client struct {
	mock.Mock
}

func (m *Client) Stopped() (a <-chan struct{}) {
	args := m.Called()
	if x := args.Get(0); x != nil {
		a = x.(<-chan struct{})
	}
	return
}

func (m *Client) Stop() {
	m.Called()
}

func (m *Client) Data(path string) (a []byte, b error) {
	args := m.Called(path)
	if x := args.Get(0); x != nil {
		a = x.([]byte)
	}
	b = args.Error(1)
	return
}

func (m *Client) WatchChildren(path string) (a string, b <-chan []string, c <-chan error) {
	args := m.Called(path)
	a = args.String(0)
	if x := args.Get(1); x != nil {
		b = x.(<-chan []string)
	}
	if x := args.Get(2); x != nil {
		c = x.(<-chan error)
	}
	return
}

// newMockZkClient returns a mocked implementation of ZKInterface that implements expectations
// for Stop() and Stopped(); multiple calls to Stop() are safe.
func NewClient(testZkPath string, initialChildren ...string) (mocked *Client, snaps chan []string, errs chan error) {
	var doneOnce sync.Once
	done := make(chan struct{})

	mocked = &Client{}
	mocked.On("Stop").Return().Run(func(_ mock.Arguments) { doneOnce.Do(func() { close(done) }) })
	mocked.On("Stopped").Return((<-chan struct{})(done))

	if initialChildren != nil {
		errs = make(chan error) // this is purposefully unbuffered (some tests depend on this)
		snaps = make(chan []string, 1)
		snaps <- initialChildren[:]
		mocked.On("WatchChildren", zoo.CurrentPath).Return(
			testZkPath, (<-chan []string)(snaps), (<-chan error)(errs)).Run(
			func(_ mock.Arguments) { log.V(1).Infoln("WatchChildren invoked") })
	}
	return
}

// implements MasterChanged and AllMasters extension
type AllMastersListener struct {
	mock.Mock
}

func (a *AllMastersListener) OnMasterChanged(mi *mesos.MasterInfo) {
	a.Called(mi)
}

func (a *AllMastersListener) UpdatedMasters(mi []*mesos.MasterInfo) {
	a.Called(mi)
}
