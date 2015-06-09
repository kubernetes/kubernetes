package etcd

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
	"time"
)

// See SetConsistency for how to use these constants.
const (
	// Using strings rather than iota because the consistency level
	// could be persisted to disk, so it'd be better to use
	// human-readable values.
	STRONG_CONSISTENCY = "STRONG"
	WEAK_CONSISTENCY   = "WEAK"
)

const (
	defaultBufferSize = 10
)

func init() {
	rand.Seed(int64(time.Now().Nanosecond()))
}

type Config struct {
	CertFile    string        `json:"certFile"`
	KeyFile     string        `json:"keyFile"`
	CaCertFile  []string      `json:"caCertFiles"`
	DialTimeout time.Duration `json:"timeout"`
	Consistency string        `json:"consistency"`
}

type credentials struct {
	username string
	password string
}

type Client struct {
	config      Config   `json:"config"`
	cluster     *Cluster `json:"cluster"`
	httpClient  *http.Client
	credentials *credentials
	transport   *http.Transport
	persistence io.Writer
	cURLch      chan string
	// CheckRetry can be used to control the policy for failed requests
	// and modify the cluster if needed.
	// The client calls it before sending requests again, and
	// stops retrying if CheckRetry returns some error. The cases that
	// this function needs to handle include no response and unexpected
	// http status code of response.
	// If CheckRetry is nil, client will call the default one
	// `DefaultCheckRetry`.
	// Argument cluster is the etcd.Cluster object that these requests have been made on.
	// Argument numReqs is the number of http.Requests that have been made so far.
	// Argument lastResp is the http.Responses from the last request.
	// Argument err is the reason of the failure.
	CheckRetry func(cluster *Cluster, numReqs int,
		lastResp http.Response, err error) error
}

// NewClient create a basic client that is configured to be used
// with the given machine list.
func NewClient(machines []string) *Client {
	config := Config{
		// default timeout is one second
		DialTimeout: time.Second,
		Consistency: WEAK_CONSISTENCY,
	}

	client := &Client{
		cluster: NewCluster(machines),
		config:  config,
	}

	client.initHTTPClient()
	client.saveConfig()

	return client
}

// NewTLSClient create a basic client with TLS configuration
func NewTLSClient(machines []string, cert, key, caCert string) (*Client, error) {
	// overwrite the default machine to use https
	if len(machines) == 0 {
		machines = []string{"https://127.0.0.1:4001"}
	}

	config := Config{
		// default timeout is one second
		DialTimeout: time.Second,
		Consistency: WEAK_CONSISTENCY,
		CertFile:    cert,
		KeyFile:     key,
		CaCertFile:  make([]string, 0),
	}

	client := &Client{
		cluster: NewCluster(machines),
		config:  config,
	}

	err := client.initHTTPSClient(cert, key)
	if err != nil {
		return nil, err
	}

	err = client.AddRootCA(caCert)

	client.saveConfig()

	return client, nil
}

// NewClientFromFile creates a client from a given file path.
// The given file is expected to use the JSON format.
func NewClientFromFile(fpath string) (*Client, error) {
	fi, err := os.Open(fpath)
	if err != nil {
		return nil, err
	}

	defer func() {
		if err := fi.Close(); err != nil {
			panic(err)
		}
	}()

	return NewClientFromReader(fi)
}

// NewClientFromReader creates a Client configured from a given reader.
// The configuration is expected to use the JSON format.
func NewClientFromReader(reader io.Reader) (*Client, error) {
	c := new(Client)

	b, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(b, c)
	if err != nil {
		return nil, err
	}
	if c.config.CertFile == "" {
		c.initHTTPClient()
	} else {
		err = c.initHTTPSClient(c.config.CertFile, c.config.KeyFile)
	}

	if err != nil {
		return nil, err
	}

	for _, caCert := range c.config.CaCertFile {
		if err := c.AddRootCA(caCert); err != nil {
			return nil, err
		}
	}

	return c, nil
}

// Override the Client's HTTP Transport object
func (c *Client) SetTransport(tr *http.Transport) {
	c.httpClient.Transport = tr
	c.transport = tr
}

func (c *Client) SetCredentials(username, password string) {
	c.credentials = &credentials{username, password}
}

func (c *Client) Close() {
	c.transport.DisableKeepAlives = true
	c.transport.CloseIdleConnections()
}

// initHTTPClient initializes a HTTP client for etcd client
func (c *Client) initHTTPClient() {
	c.transport = &http.Transport{
		Dial: c.dial,
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: true,
		},
	}
	c.httpClient = &http.Client{Transport: c.transport}
}

// initHTTPClient initializes a HTTPS client for etcd client
func (c *Client) initHTTPSClient(cert, key string) error {
	if cert == "" || key == "" {
		return errors.New("Require both cert and key path")
	}

	tlsCert, err := tls.LoadX509KeyPair(cert, key)
	if err != nil {
		return err
	}

	tlsConfig := &tls.Config{
		Certificates:       []tls.Certificate{tlsCert},
		InsecureSkipVerify: true,
	}

	tr := &http.Transport{
		TLSClientConfig: tlsConfig,
		Dial:            c.dial,
	}

	c.httpClient = &http.Client{Transport: tr}
	return nil
}

// SetPersistence sets a writer to which the config will be
// written every time it's changed.
func (c *Client) SetPersistence(writer io.Writer) {
	c.persistence = writer
}

// SetConsistency changes the consistency level of the client.
//
// When consistency is set to STRONG_CONSISTENCY, all requests,
// including GET, are sent to the leader.  This means that, assuming
// the absence of leader failures, GET requests are guaranteed to see
// the changes made by previous requests.
//
// When consistency is set to WEAK_CONSISTENCY, other requests
// are still sent to the leader, but GET requests are sent to a
// random server from the server pool.  This reduces the read
// load on the leader, but it's not guaranteed that the GET requests
// will see changes made by previous requests (they might have not
// yet been committed on non-leader servers).
func (c *Client) SetConsistency(consistency string) error {
	if !(consistency == STRONG_CONSISTENCY || consistency == WEAK_CONSISTENCY) {
		return errors.New("The argument must be either STRONG_CONSISTENCY or WEAK_CONSISTENCY.")
	}
	c.config.Consistency = consistency
	return nil
}

// Sets the DialTimeout value
func (c *Client) SetDialTimeout(d time.Duration) {
	c.config.DialTimeout = d
}

// AddRootCA adds a root CA cert for the etcd client
func (c *Client) AddRootCA(caCert string) error {
	if c.httpClient == nil {
		return errors.New("Client has not been initialized yet!")
	}

	certBytes, err := ioutil.ReadFile(caCert)
	if err != nil {
		return err
	}

	tr, ok := c.httpClient.Transport.(*http.Transport)

	if !ok {
		panic("AddRootCA(): Transport type assert should not fail")
	}

	if tr.TLSClientConfig.RootCAs == nil {
		caCertPool := x509.NewCertPool()
		ok = caCertPool.AppendCertsFromPEM(certBytes)
		if ok {
			tr.TLSClientConfig.RootCAs = caCertPool
		}
		tr.TLSClientConfig.InsecureSkipVerify = false
	} else {
		ok = tr.TLSClientConfig.RootCAs.AppendCertsFromPEM(certBytes)
	}

	if !ok {
		err = errors.New("Unable to load caCert")
	}

	c.config.CaCertFile = append(c.config.CaCertFile, caCert)
	c.saveConfig()

	return err
}

// SetCluster updates cluster information using the given machine list.
func (c *Client) SetCluster(machines []string) bool {
	success := c.internalSyncCluster(machines)
	return success
}

func (c *Client) GetCluster() []string {
	return c.cluster.Machines
}

// SyncCluster updates the cluster information using the internal machine list.
// If no members are found, the intenral machine list is left untouched.
func (c *Client) SyncCluster() bool {
	return c.internalSyncCluster(c.cluster.Machines)
}

// internalSyncCluster syncs cluster information using the given machine list.
func (c *Client) internalSyncCluster(machines []string) bool {
	// comma-separated list of machines in the cluster.
	members := ""

	for _, machine := range machines {
		httpPath := c.createHttpPath(machine, path.Join(version, "members"))
		resp, err := c.httpClient.Get(httpPath)
		if err != nil {
			// try another machine in the cluster
			continue
		}

		if resp.StatusCode != http.StatusOK { // fall-back to old endpoint
			httpPath := c.createHttpPath(machine, path.Join(version, "machines"))
			resp, err := c.httpClient.Get(httpPath)
			if err != nil {
				// try another machine in the cluster
				continue
			}
			b, err := ioutil.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				// try another machine in the cluster
				continue
			}
			members = string(b)
		} else {
			b, err := ioutil.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				// try another machine in the cluster
				continue
			}

			var mCollection memberCollection
			if err := json.Unmarshal(b, &mCollection); err != nil {
				// try another machine
				continue
			}

			urls := make([]string, 0)
			for _, m := range mCollection {
				urls = append(urls, m.ClientURLs...)
			}

			members = strings.Join(urls, ",")
		}

		// We should never do an empty cluster update.
		if members == "" {
			continue
		}

		// update Machines List
		c.cluster.updateFromStr(members)
		logger.Debug("sync.machines ", c.cluster.Machines)
		c.saveConfig()
		return true
	}

	return false
}

// createHttpPath creates a complete HTTP URL.
// serverName should contain both the host name and a port number, if any.
func (c *Client) createHttpPath(serverName string, _path string) string {
	u, err := url.Parse(serverName)
	if err != nil {
		panic(err)
	}

	u.Path = path.Join(u.Path, _path)

	if u.Scheme == "" {
		u.Scheme = "http"
	}
	return u.String()
}

// dial attempts to open a TCP connection to the provided address, explicitly
// enabling keep-alives with a one-second interval.
func (c *Client) dial(network, addr string) (net.Conn, error) {
	conn, err := net.DialTimeout(network, addr, c.config.DialTimeout)
	if err != nil {
		return nil, err
	}

	tcpConn, ok := conn.(*net.TCPConn)
	if !ok {
		return nil, errors.New("Failed type-assertion of net.Conn as *net.TCPConn")
	}

	// Keep TCP alive to check whether or not the remote machine is down
	if err = tcpConn.SetKeepAlive(true); err != nil {
		return nil, err
	}

	if err = tcpConn.SetKeepAlivePeriod(time.Second); err != nil {
		return nil, err
	}

	return tcpConn, nil
}

func (c *Client) OpenCURL() {
	c.cURLch = make(chan string, defaultBufferSize)
}

func (c *Client) CloseCURL() {
	c.cURLch = nil
}

func (c *Client) sendCURL(command string) {
	go func() {
		select {
		case c.cURLch <- command:
		default:
		}
	}()
}

func (c *Client) RecvCURL() string {
	return <-c.cURLch
}

// saveConfig saves the current config using c.persistence.
func (c *Client) saveConfig() error {
	if c.persistence != nil {
		b, err := json.Marshal(c)
		if err != nil {
			return err
		}

		_, err = c.persistence.Write(b)
		if err != nil {
			return err
		}
	}

	return nil
}

// MarshalJSON implements the Marshaller interface
// as defined by the standard JSON package.
func (c *Client) MarshalJSON() ([]byte, error) {
	b, err := json.Marshal(struct {
		Config  Config   `json:"config"`
		Cluster *Cluster `json:"cluster"`
	}{
		Config:  c.config,
		Cluster: c.cluster,
	})

	if err != nil {
		return nil, err
	}

	return b, nil
}

// UnmarshalJSON implements the Unmarshaller interface
// as defined by the standard JSON package.
func (c *Client) UnmarshalJSON(b []byte) error {
	temp := struct {
		Config  Config   `json:"config"`
		Cluster *Cluster `json:"cluster"`
	}{}
	err := json.Unmarshal(b, &temp)
	if err != nil {
		return err
	}

	c.cluster = temp.Cluster
	c.config = temp.Config
	return nil
}
