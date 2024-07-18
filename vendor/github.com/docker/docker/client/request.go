package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/versions"
	"github.com/docker/docker/errdefs"
	"github.com/pkg/errors"
)

// serverResponse is a wrapper for http API responses.
type serverResponse struct {
	body       io.ReadCloser
	header     http.Header
	statusCode int
	reqURL     *url.URL
}

// head sends an http request to the docker API using the method HEAD.
func (cli *Client) head(ctx context.Context, path string, query url.Values, headers map[string][]string) (serverResponse, error) {
	return cli.sendRequest(ctx, http.MethodHead, path, query, nil, headers)
}

// get sends an http request to the docker API using the method GET with a specific Go context.
func (cli *Client) get(ctx context.Context, path string, query url.Values, headers map[string][]string) (serverResponse, error) {
	return cli.sendRequest(ctx, http.MethodGet, path, query, nil, headers)
}

// post sends an http request to the docker API using the method POST with a specific Go context.
func (cli *Client) post(ctx context.Context, path string, query url.Values, obj interface{}, headers map[string][]string) (serverResponse, error) {
	body, headers, err := encodeBody(obj, headers)
	if err != nil {
		return serverResponse{}, err
	}
	return cli.sendRequest(ctx, http.MethodPost, path, query, body, headers)
}

func (cli *Client) postRaw(ctx context.Context, path string, query url.Values, body io.Reader, headers map[string][]string) (serverResponse, error) {
	return cli.sendRequest(ctx, http.MethodPost, path, query, body, headers)
}

// putRaw sends an http request to the docker API using the method PUT.
func (cli *Client) putRaw(ctx context.Context, path string, query url.Values, body io.Reader, headers map[string][]string) (serverResponse, error) {
	return cli.sendRequest(ctx, http.MethodPut, path, query, body, headers)
}

// delete sends an http request to the docker API using the method DELETE.
func (cli *Client) delete(ctx context.Context, path string, query url.Values, headers map[string][]string) (serverResponse, error) {
	return cli.sendRequest(ctx, http.MethodDelete, path, query, nil, headers)
}

type headers map[string][]string

func encodeBody(obj interface{}, headers headers) (io.Reader, headers, error) {
	if obj == nil {
		return nil, headers, nil
	}

	body, err := encodeData(obj)
	if err != nil {
		return nil, headers, err
	}
	if headers == nil {
		headers = make(map[string][]string)
	}
	headers["Content-Type"] = []string{"application/json"}
	return body, headers, nil
}

func (cli *Client) buildRequest(method, path string, body io.Reader, headers headers) (*http.Request, error) {
	expectedPayload := (method == http.MethodPost || method == http.MethodPut)
	if expectedPayload && body == nil {
		body = bytes.NewReader([]byte{})
	}

	req, err := http.NewRequest(method, path, body)
	if err != nil {
		return nil, err
	}
	req = cli.addHeaders(req, headers)
	req.URL.Scheme = cli.scheme
	req.URL.Host = cli.addr

	if cli.proto == "unix" || cli.proto == "npipe" {
		// Override host header for non-tcp connections.
		req.Host = DummyHost
	}

	if expectedPayload && req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "text/plain")
	}
	return req, nil
}

func (cli *Client) sendRequest(ctx context.Context, method, path string, query url.Values, body io.Reader, headers headers) (serverResponse, error) {
	req, err := cli.buildRequest(method, cli.getAPIPath(ctx, path, query), body, headers)
	if err != nil {
		return serverResponse{}, err
	}
	resp, err := cli.doRequest(ctx, req)
	if err != nil {
		return resp, errdefs.FromStatusCode(err, resp.statusCode)
	}
	err = cli.checkResponseErr(resp)
	return resp, errdefs.FromStatusCode(err, resp.statusCode)
}

func (cli *Client) doRequest(ctx context.Context, req *http.Request) (serverResponse, error) {
	serverResp := serverResponse{statusCode: -1, reqURL: req.URL}

	req = req.WithContext(ctx)
	resp, err := cli.client.Do(req)
	if err != nil {
		if cli.scheme != "https" && strings.Contains(err.Error(), "malformed HTTP response") {
			return serverResp, fmt.Errorf("%v.\n* Are you trying to connect to a TLS-enabled daemon without TLS?", err)
		}

		if cli.scheme == "https" && strings.Contains(err.Error(), "bad certificate") {
			return serverResp, errors.Wrap(err, "the server probably has client authentication (--tlsverify) enabled; check your TLS client certification settings")
		}

		// Don't decorate context sentinel errors; users may be comparing to
		// them directly.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return serverResp, err
		}

		if nErr, ok := err.(*url.Error); ok {
			if nErr, ok := nErr.Err.(*net.OpError); ok {
				if os.IsPermission(nErr.Err) {
					return serverResp, errors.Wrapf(err, "permission denied while trying to connect to the Docker daemon socket at %v", cli.host)
				}
			}
		}

		if err, ok := err.(net.Error); ok {
			if err.Timeout() {
				return serverResp, ErrorConnectionFailed(cli.host)
			}
			if strings.Contains(err.Error(), "connection refused") || strings.Contains(err.Error(), "dial unix") {
				return serverResp, ErrorConnectionFailed(cli.host)
			}
		}

		// Although there's not a strongly typed error for this in go-winio,
		// lots of people are using the default configuration for the docker
		// daemon on Windows where the daemon is listening on a named pipe
		// `//./pipe/docker_engine, and the client must be running elevated.
		// Give users a clue rather than the not-overly useful message
		// such as `error during connect: Get http://%2F%2F.%2Fpipe%2Fdocker_engine/v1.26/info:
		// open //./pipe/docker_engine: The system cannot find the file specified.`.
		// Note we can't string compare "The system cannot find the file specified" as
		// this is localised - for example in French the error would be
		// `open //./pipe/docker_engine: Le fichier spécifié est introuvable.`
		if strings.Contains(err.Error(), `open //./pipe/docker_engine`) {
			// Checks if client is running with elevated privileges
			if f, elevatedErr := os.Open("\\\\.\\PHYSICALDRIVE0"); elevatedErr == nil {
				err = errors.Wrap(err, "in the default daemon configuration on Windows, the docker client must be run with elevated privileges to connect")
			} else {
				f.Close()
				err = errors.Wrap(err, "this error may indicate that the docker daemon is not running")
			}
		}

		return serverResp, errors.Wrap(err, "error during connect")
	}

	if resp != nil {
		serverResp.statusCode = resp.StatusCode
		serverResp.body = resp.Body
		serverResp.header = resp.Header
	}
	return serverResp, nil
}

func (cli *Client) checkResponseErr(serverResp serverResponse) error {
	if serverResp.statusCode >= 200 && serverResp.statusCode < 400 {
		return nil
	}

	var body []byte
	var err error
	if serverResp.body != nil {
		bodyMax := 1 * 1024 * 1024 // 1 MiB
		bodyR := &io.LimitedReader{
			R: serverResp.body,
			N: int64(bodyMax),
		}
		body, err = io.ReadAll(bodyR)
		if err != nil {
			return err
		}
		if bodyR.N == 0 {
			return fmt.Errorf("request returned %s with a message (> %d bytes) for API route and version %s, check if the server supports the requested API version", http.StatusText(serverResp.statusCode), bodyMax, serverResp.reqURL)
		}
	}
	if len(body) == 0 {
		return fmt.Errorf("request returned %s for API route and version %s, check if the server supports the requested API version", http.StatusText(serverResp.statusCode), serverResp.reqURL)
	}

	var ct string
	if serverResp.header != nil {
		ct = serverResp.header.Get("Content-Type")
	}

	var errorMessage string
	if (cli.version == "" || versions.GreaterThan(cli.version, "1.23")) && ct == "application/json" {
		var errorResponse types.ErrorResponse
		if err := json.Unmarshal(body, &errorResponse); err != nil {
			return errors.Wrap(err, "Error reading JSON")
		}
		errorMessage = strings.TrimSpace(errorResponse.Message)
	} else {
		errorMessage = strings.TrimSpace(string(body))
	}

	return errors.Wrap(errors.New(errorMessage), "Error response from daemon")
}

func (cli *Client) addHeaders(req *http.Request, headers headers) *http.Request {
	// Add CLI Config's HTTP Headers BEFORE we set the Docker headers
	// then the user can't change OUR headers
	for k, v := range cli.customHTTPHeaders {
		if versions.LessThan(cli.version, "1.25") && k == "User-Agent" {
			continue
		}
		req.Header.Set(k, v)
	}

	for k, v := range headers {
		req.Header[k] = v
	}
	return req
}

func encodeData(data interface{}) (*bytes.Buffer, error) {
	params := bytes.NewBuffer(nil)
	if data != nil {
		if err := json.NewEncoder(params).Encode(data); err != nil {
			return nil, err
		}
	}
	return params, nil
}

func ensureReaderClosed(response serverResponse) {
	if response.body != nil {
		// Drain up to 512 bytes and close the body to let the Transport reuse the connection
		io.CopyN(io.Discard, response.body, 512)
		response.body.Close()
	}
}
