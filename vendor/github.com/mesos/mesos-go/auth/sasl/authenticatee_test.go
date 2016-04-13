package sasl

import (
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/mesos/mesos-go/auth/callback"
	"github.com/mesos/mesos-go/auth/sasl/mech/crammd5"
	mock_sasl "github.com/mesos/mesos-go/auth/sasl/mock"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/messenger"
	mock_messenger "github.com/mesos/mesos-go/messenger/mock"
	"github.com/mesos/mesos-go/upid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"golang.org/x/net/context"
)

func TestAuthticatee_validLogin(t *testing.T) {
	assert := assert.New(t)
	ctx := context.TODO()
	client := upid.UPID{
		ID:   "someFramework",
		Host: "b.net",
		Port: "789",
	}
	server := upid.UPID{
		ID:   "serv",
		Host: "a.com",
		Port: "123",
	}
	tpid := upid.UPID{
		ID:   "sasl_transport",
		Host: "g.org",
		Port: "456",
	}
	handler := callback.HandlerFunc(func(cb ...callback.Interface) error {
		for _, c := range cb {
			switch c := c.(type) {
			case *callback.Name:
				c.Set("foo")
			case *callback.Password:
				c.Set([]byte("bar"))
			case *callback.Interprocess:
				c.Set(server, client)
			default:
				return &callback.Unsupported{Callback: c}
			}
		}
		return nil
	})
	var transport *mock_sasl.Transport
	factory := transportFactoryFunc(func() messenger.Messenger {
		transport = &mock_sasl.Transport{mock_messenger.NewMessenger()}
		transport.On("Install").Return(nil)
		transport.On("UPID").Return(tpid)
		transport.On("Start").Return(nil)
		transport.On("Stop").Return(nil)

		mechMsg := make(chan struct{})
		stepMsg := make(chan struct{})

		transport.On("Send", mock.Anything, &server, &mesos.AuthenticateMessage{
			Pid: proto.String(client.String()),
		}).Return(nil).Run(func(_ mock.Arguments) {
			defer close(mechMsg)
			transport.Recv(&server, &mesos.AuthenticationMechanismsMessage{
				Mechanisms: []string{crammd5.Name},
			})
		}).Once()

		transport.On("Send", mock.Anything, &server, &mesos.AuthenticationStartMessage{
			Mechanism: proto.String(crammd5.Name),
		}).Return(nil).Run(func(_ mock.Arguments) {
			defer close(stepMsg)
			<-mechMsg
			transport.Recv(&server, &mesos.AuthenticationStepMessage{
				Data: []byte(`lsd;lfkgjs;dlfkgjs;dfklg`),
			})
		}).Once()

		transport.On("Send", mock.Anything, &server, &mesos.AuthenticationStepMessage{
			Data: []byte(`foo cc7fd96cd80123ea844a7dba29a594ed`),
		}).Return(nil).Run(func(_ mock.Arguments) {
			<-stepMsg
			transport.Recv(&server, &mesos.AuthenticationCompletedMessage{})
		}).Once()

		return transport
	})
	login, err := makeAuthenticatee(handler, factory)
	assert.Nil(err)

	err = login.Authenticate(ctx, handler)
	assert.Nil(err)
	assert.NotNil(transport)
	time.Sleep(1 * time.Second) // wait for the authenticator to shut down
	transport.AssertExpectations(t)
}
