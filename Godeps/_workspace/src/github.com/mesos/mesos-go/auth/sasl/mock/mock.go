package mock

import (
	"github.com/gogo/protobuf/proto"
	mock_messenger "github.com/mesos/mesos-go/messenger/mock"
	"github.com/mesos/mesos-go/upid"
	"github.com/stretchr/testify/mock"
	"golang.org/x/net/context"
)

type Transport struct {
	*mock_messenger.Messenger
}

func (m *Transport) Send(ctx context.Context, upid *upid.UPID, msg proto.Message) error {
	return m.Called(mock.Anything, upid, msg).Error(0)
}
