package client // import "github.com/docker/docker/client"

import (
	"context"
	"net"
	"net/http"
)

// DialSession returns a connection that can be used communication with daemon
func (cli *Client) DialSession(ctx context.Context, proto string, meta map[string][]string) (net.Conn, error) {
	req, err := http.NewRequest("POST", "/session", nil)
	if err != nil {
		return nil, err
	}
	req = cli.addHeaders(req, meta)

	return cli.setupHijackConn(ctx, req, proto)
}
