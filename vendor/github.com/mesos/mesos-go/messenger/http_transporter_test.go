package messenger

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"testing"
	"time"

	"github.com/mesos/mesos-go/messenger/testmessage"
	"github.com/mesos/mesos-go/upid"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

func TestTransporterNew(t *testing.T) {
	trans := NewHTTPTransporter(upid.UPID{ID: "mesos1", Host: "localhost"}, nil)
	assert.NotNil(t, trans)
	assert.NotNil(t, trans.messageQueue)
	assert.NotNil(t, trans.client)
}

func TestTransporterSend(t *testing.T) {
	idreg := regexp.MustCompile(`[A-Za-z0-9_\-]+@[A-Za-z0-9_\-\.]+:[0-9]+`)
	serverId := "testserver"

	// setup mesos client-side
	protoMsg := testmessage.GenerateSmallMessage()
	msgName := getMessageName(protoMsg)
	msg := &Message{
		Name:         msgName,
		ProtoMessage: protoMsg,
	}
	requestURI := fmt.Sprintf("/%s/%s", serverId, msgName)

	// setup server-side
	msgReceived := make(chan struct{})
	srv := makeMockServer(requestURI, func(rsp http.ResponseWriter, req *http.Request) {
		defer close(msgReceived)
		from := req.Header.Get("Libprocess-From")
		assert.NotEmpty(t, from)
		assert.True(t, idreg.MatchString(from), fmt.Sprintf("regexp failed for '%v'", from))
	})
	defer srv.Close()
	toUpid, err := upid.Parse(fmt.Sprintf("%s@%s", serverId, srv.Listener.Addr().String()))
	assert.NoError(t, err)

	// make transport call.
	transport := NewHTTPTransporter(upid.UPID{ID: "mesos1", Host: "localhost"}, nil)
	_, errch := transport.Start()
	defer transport.Stop(false)

	msg.UPID = toUpid
	err = transport.Send(context.TODO(), msg)
	assert.NoError(t, err)

	select {
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for message receipt")
	case <-msgReceived:
	case err := <-errch:
		if err != nil {
			t.Fatalf(err.Error())
		}
	}
}

func TestTransporter_DiscardedSend(t *testing.T) {
	serverId := "testserver"

	// setup mesos client-side
	protoMsg := testmessage.GenerateSmallMessage()
	msgName := getMessageName(protoMsg)
	msg := &Message{
		Name:         msgName,
		ProtoMessage: protoMsg,
	}
	requestURI := fmt.Sprintf("/%s/%s", serverId, msgName)

	// setup server-side
	msgReceived := make(chan struct{})
	srv := makeMockServer(requestURI, func(rsp http.ResponseWriter, req *http.Request) {
		close(msgReceived)
		time.Sleep(2 * time.Second) // long enough that we should be able to stop it
	})
	defer srv.Close()
	toUpid, err := upid.Parse(fmt.Sprintf("%s@%s", serverId, srv.Listener.Addr().String()))
	assert.NoError(t, err)

	// make transport call.
	transport := NewHTTPTransporter(upid.UPID{ID: "mesos1", Host: "localhost"}, nil)
	_, errch := transport.Start()
	defer transport.Stop(false)

	msg.UPID = toUpid
	senderr := make(chan struct{})
	go func() {
		defer close(senderr)
		err = transport.Send(context.TODO(), msg)
		assert.NotNil(t, err)
		assert.Equal(t, discardOnStopError, err)
	}()

	// wait for message to be received
	select {
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for message receipt")
		return
	case <-msgReceived:
		transport.Stop(false)
	case err := <-errch:
		if err != nil {
			t.Fatalf(err.Error())
			return
		}
	}

	// wait for send() to process discarded-error
	select {
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for aborted send")
		return
	case <-senderr: // continue
	}
}

func TestTransporterStartAndRcvd(t *testing.T) {
	serverId := "testserver"
	serverAddr := "127.0.0.1"
	protoMsg := testmessage.GenerateSmallMessage()
	msgName := getMessageName(protoMsg)
	ctrl := make(chan struct{})

	// setup receiver (server) process
	receiver := NewHTTPTransporter(upid.UPID{ID: serverId, Host: serverAddr}, nil)
	receiver.Install(msgName)

	go func() {
		defer close(ctrl)
		t.Logf("received something...")
		msg, err := receiver.Recv()
		assert.Nil(t, err)
		assert.NotNil(t, msg)
		if msg != nil {
			assert.Equal(t, msgName, msg.Name)
		}
	}()

	rcvPid, errch := receiver.Start()
	defer receiver.Stop(false)
	assert.NotNil(t, errch)

	time.Sleep(time.Millisecond * 7) // time to catchup

	// setup sender (client) process
	sender := NewHTTPTransporter(upid.UPID{ID: "mesos1", Host: "localhost"}, nil)
	msg := &Message{
		UPID:         &rcvPid,
		Name:         msgName,
		ProtoMessage: protoMsg,
	}
	_, errch2 := sender.Start()
	defer sender.Stop(false)

	t.Logf("sending test message")
	sender.Send(context.TODO(), msg)

	select {
	case <-time.After(time.Second * 5):
		t.Fatalf("Timeout")
	case <-ctrl:
	case err := <-errch:
		if err != nil {
			t.Fatalf(err.Error())
		}
	case err := <-errch2:
		if err != nil {
			t.Fatalf(err.Error())
		}
	}
}

func TestTransporterStartAndStop(t *testing.T) {
	serverId := "testserver"

	// setup receiver (server) process
	receiver := NewHTTPTransporter(upid.UPID{ID: serverId, Host: "127.0.0.1"}, nil)

	_, errch := receiver.Start()
	assert.NotNil(t, errch)

	time.Sleep(1 * time.Second)
	receiver.Stop(false)

	select {
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for transport to stop")
	case err := <-errch:
		if err != nil {
			t.Fatalf(err.Error())
		}
	}
}

func TestMutatedHostUPid(t *testing.T) {
	serverId := "testserver"
	// NOTE(tsenart): This static port can cause conflicts if multiple instances
	// of this test run concurrently or else if this port is already bound by
	// another socket.
	serverPort := 12345
	serverHost := "127.0.0.1"
	serverAddr := serverHost + ":" + strconv.Itoa(serverPort)

	// override the upid.Host with this listener IP
	addr := net.ParseIP("0.0.0.0")

	// setup receiver (server) process
	uPid, err := upid.Parse(fmt.Sprintf("%s@%s", serverId, serverAddr))
	assert.NoError(t, err)
	receiver := NewHTTPTransporter(*uPid, addr)

	err = receiver.listen()
	assert.NoError(t, err)

	if receiver.upid.Host != "127.0.0.1" {
		t.Fatalf("reciever.upid.Host was expected to return %s, got %s\n", serverHost, receiver.upid.Host)
	}

	if receiver.upid.Port != strconv.Itoa(serverPort) {
		t.Fatalf("receiver.upid.Port was expected to return %d, got %s\n", serverPort, receiver.upid.Port)
	}
}

func TestEmptyHostPortUPid(t *testing.T) {
	uPid := upid.UPID{ID: "testserver"}

	// override the upid.Host with this listener IP
	addr := net.ParseIP("0.0.0.0")
	receiver := NewHTTPTransporter(uPid, addr)

	err := receiver.listen()
	assert.NoError(t, err)

	// This should be the host that overrides as uPid.Host is empty
	if receiver.upid.Host != "0.0.0.0" {
		t.Fatalf("reciever.upid.Host was expected to return 0.0.0.0, got %q", receiver.upid.Host)
	}

	if receiver.upid.Port == "0" {
		t.Fatalf("receiver.upid.Port was not expected to return 0, got %q", receiver.upid.Port)
	}
}

func makeMockServer(path string, handler func(rsp http.ResponseWriter, req *http.Request)) *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc(path, handler)
	return httptest.NewServer(mux)
}

func TestProcessOneRequest(t *testing.T) {
	ht := &HTTPTransporter{
		messageQueue: make(chan *Message, 1),
		shouldQuit:   make(chan struct{}),
	}
	testfunc := func(expectProceed bool) {
		rchan := make(chan Response, 1)
		proceed := ht.processOneRequest(&upid.UPID{ID: "james"}, &Request{
			response: rchan,
			Request: &http.Request{
				Method:     "foo",
				RequestURI: "a/z/bar",
				Body:       ioutil.NopCloser(&bytes.Reader{}),
			},
		})
		// expecting to get a 202 response since the request doesn't have libprocess headers
		if proceed != expectProceed {
			t.Fatalf("expected proceed signal %t instead of %t", expectProceed, proceed)
		}
		select {
		case resp := <-rchan:
			if resp.code != 202 {
				t.Fatalf("expected a 202 response for all libprocess requests")
			}
		default:
			t.Fatalf("expected a response since we're not a libprocess agent")
		}
		select {
		case m := <-ht.messageQueue:
			// From, Name, Data
			assert.Equal(t, "james", m.UPID.ID)
			assert.Equal(t, "bar", m.Name)
		default:
			t.Fatalf("expected a message for the request that was processed")
		}
	}
	t.Log("testing w/o shouldQuit signal")
	testfunc(true)

	t.Log("testing w/ shouldQuit signal")
	close(ht.shouldQuit)
	for i := 0; i < 100; i++ {
		testfunc(false) // do this in a loop to test determinism
	}
}
