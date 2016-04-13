package trust

import (
	"crypto/x509"
	"errors"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"sync"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/libtrust/trustgraph"
)

type TrustStore struct {
	path          string
	caPool        *x509.CertPool
	graph         trustgraph.TrustGraph
	expiration    time.Time
	fetcher       *time.Timer
	fetchTime     time.Duration
	autofetch     bool
	httpClient    *http.Client
	baseEndpoints map[string]*url.URL

	sync.RWMutex
}

// defaultFetchtime represents the starting duration to wait between
// fetching sections of the graph.  Unsuccessful fetches should
// increase time between fetching.
const defaultFetchtime = 45 * time.Second

var baseEndpoints = map[string]string{"official": "https://dvjy3tqbc323p.cloudfront.net/trust/official.json"}

func NewTrustStore(path string) (*TrustStore, error) {
	abspath, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}

	// Create base graph url map
	endpoints := map[string]*url.URL{}
	for name, endpoint := range baseEndpoints {
		u, err := url.Parse(endpoint)
		if err != nil {
			return nil, err
		}
		endpoints[name] = u
	}

	// Load grant files
	t := &TrustStore{
		path:          abspath,
		caPool:        nil,
		httpClient:    &http.Client{},
		fetchTime:     time.Millisecond,
		baseEndpoints: endpoints,
	}

	if err := t.reload(); err != nil {
		return nil, err
	}

	return t, nil
}

func (t *TrustStore) reload() error {
	t.Lock()
	defer t.Unlock()

	matches, err := filepath.Glob(filepath.Join(t.path, "*.json"))
	if err != nil {
		return err
	}
	statements := make([]*trustgraph.Statement, len(matches))
	for i, match := range matches {
		f, err := os.Open(match)
		if err != nil {
			return err
		}
		statements[i], err = trustgraph.LoadStatement(f, nil)
		if err != nil {
			f.Close()
			return err
		}
		f.Close()
	}
	if len(statements) == 0 {
		if t.autofetch {
			logrus.Debugf("No grants, fetching")
			t.fetcher = time.AfterFunc(t.fetchTime, t.fetch)
		}
		return nil
	}

	grants, expiration, err := trustgraph.CollapseStatements(statements, true)
	if err != nil {
		return err
	}

	t.expiration = expiration
	t.graph = trustgraph.NewMemoryGraph(grants)
	logrus.Debugf("Reloaded graph with %d grants expiring at %s", len(grants), expiration)

	if t.autofetch {
		nextFetch := expiration.Sub(time.Now())
		if nextFetch < 0 {
			nextFetch = defaultFetchtime
		} else {
			nextFetch = time.Duration(0.8 * (float64)(nextFetch))
		}
		t.fetcher = time.AfterFunc(nextFetch, t.fetch)
	}

	return nil
}

func (t *TrustStore) fetchBaseGraph(u *url.URL) (*trustgraph.Statement, error) {
	req := &http.Request{
		Method:     "GET",
		URL:        u,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     make(http.Header),
		Body:       nil,
		Host:       u.Host,
	}

	resp, err := t.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == 404 {
		return nil, errors.New("base graph does not exist")
	}

	defer resp.Body.Close()

	return trustgraph.LoadStatement(resp.Body, t.caPool)
}

// fetch retrieves updated base graphs.  This function cannot error, it
// should only log errors
func (t *TrustStore) fetch() {
	t.Lock()
	defer t.Unlock()

	if t.autofetch && t.fetcher == nil {
		// Do nothing ??
		return
	}

	fetchCount := 0
	for bg, ep := range t.baseEndpoints {
		statement, err := t.fetchBaseGraph(ep)
		if err != nil {
			logrus.Infof("Trust graph fetch failed: %s", err)
			continue
		}
		b, err := statement.Bytes()
		if err != nil {
			logrus.Infof("Bad trust graph statement: %s", err)
			continue
		}
		// TODO check if value differs
		if err := ioutil.WriteFile(path.Join(t.path, bg+".json"), b, 0600); err != nil {
			logrus.Infof("Error writing trust graph statement: %s", err)
		}
		fetchCount++
	}
	logrus.Debugf("Fetched %d base graphs at %s", fetchCount, time.Now())

	if fetchCount > 0 {
		go func() {
			if err := t.reload(); err != nil {
				logrus.Infof("Reload of trust graph failed: %s", err)
			}
		}()
		t.fetchTime = defaultFetchtime
		t.fetcher = nil
	} else if t.autofetch {
		maxTime := 10 * defaultFetchtime
		t.fetchTime = time.Duration(1.5 * (float64)(t.fetchTime+time.Second))
		if t.fetchTime > maxTime {
			t.fetchTime = maxTime
		}
		t.fetcher = time.AfterFunc(t.fetchTime, t.fetch)
	}
}
