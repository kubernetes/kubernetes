package zoo

import (
	"errors"
	"fmt"
	"net/url"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	util "github.com/mesos/mesos-go/mesosutil"
	"github.com/samuel/go-zookeeper/zk"
)

type MockMasterDetector struct {
	*MasterDetector
	zkPath string
	conCh  chan zk.Event
	sesCh  chan zk.Event
}

func NewMockMasterDetector(zkurls string) (*MockMasterDetector, error) {
	log.V(4).Infoln("Creating mock zk master detector")
	md, err := NewMasterDetector(zkurls)
	if err != nil {
		return nil, err
	}

	u, _ := url.Parse(zkurls)
	m := &MockMasterDetector{
		MasterDetector: md,
		zkPath:         u.Path,
		conCh:          make(chan zk.Event, 5),
		sesCh:          make(chan zk.Event, 5),
	}

	path := m.zkPath
	connector := NewMockConnector()
	connector.On("Children", path).Return([]string{"info_0", "info_5", "info_10"}, &zk.Stat{}, nil)
	connector.On("Get", fmt.Sprintf("%s/info_0", path)).Return(m.makeMasterInfo(), &zk.Stat{}, nil)
	connector.On("Close").Return(nil)
	connector.On("ChildrenW", m.zkPath).Return([]string{m.zkPath}, &zk.Stat{}, (<-chan zk.Event)(m.sesCh), nil)

	first := true
	m.client.setFactory(asFactory(func() (Connector, <-chan zk.Event, error) {
		if !first {
			return nil, nil, errors.New("only 1 connector allowed")
		} else {
			first = false
		}
		return connector, m.conCh, nil
	}))

	return m, nil
}

func (m *MockMasterDetector) Start() {
	m.client.connect()
}

func (m *MockMasterDetector) ScheduleConnEvent(s zk.State) {
	log.V(4).Infof("Scheduling zk connection event with state: %v\n", s)
	go func() {
		m.conCh <- zk.Event{
			State: s,
			Path:  m.zkPath,
		}
	}()
}

func (m *MockMasterDetector) ScheduleSessEvent(t zk.EventType) {
	log.V(4).Infof("Scheduling zk session event with state: %v\n", t)
	go func() {
		m.sesCh <- zk.Event{
			Type: t,
			Path: m.zkPath,
		}
	}()
}

func (m *MockMasterDetector) makeMasterInfo() []byte {
	miPb := util.NewMasterInfo("master", 123456789, 400)
	miPb.Pid = proto.String("master@127.0.0.1:5050")
	data, err := proto.Marshal(miPb)
	if err != nil {
		panic(err)
	}
	return data
}
