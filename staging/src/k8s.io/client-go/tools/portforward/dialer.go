package portforward

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"

	v1 "k8s.io/api/core/v1"

	"k8s.io/client-go/tools/remotecommand"

	"k8s.io/client-go/transport/websocket"

	restclient "k8s.io/client-go/rest"
)

var (
	streamType3streamID = map[string]byte{
		v1.StreamTypeData:  0,
		v1.StreamTypeError: 1,
	}
)

type PortForwardWSDialer struct {
	transport http.RoundTripper
	upgrader  websocket.ConnectionHolder
	u         *url.URL
	method    string
}

func NewDialer(config *restclient.Config, u *url.URL, method string) (*PortForwardWSDialer, error) {
	transport, upgrader, err := websocket.RoundTripperFor(config)
	if err != nil {
		return nil, err
	}

	return &PortForwardWSDialer{
		transport: transport,
		upgrader:  upgrader,
		u:         u,
		method:    method,
	}, nil
}

func (d *PortForwardWSDialer) Dial(port uint16, protocols ...string) (*remotecommand.WSStreamCreator, string, error) {
	req, err := http.NewRequestWithContext(context.TODO(), d.method, d.u.String(), nil)
	if err != nil {
		return nil, "", err
	}

	query := req.URL.Query()
	query.Set("ports", strconv.Itoa(int(port)))
	req.URL.RawQuery = query.Encode()
	conn, err := websocket.Negotiate(d.transport, d.upgrader, req, protocols...)
	if err != nil {
		return nil, "", err
	}
	if conn == nil {
		panic(fmt.Errorf("websocket connection is nil"))
	}

	return remotecommand.NewWSStreamCreator(conn, streamType3streamID), conn.Subprotocol(), nil
}
