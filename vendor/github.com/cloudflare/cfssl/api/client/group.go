package client

import (
	"crypto/tls"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/cloudflare/cfssl/auth"
	"github.com/cloudflare/cfssl/info"
)

// Strategy is the means by which the server to use as a remote should
// be selected.
type Strategy int

const (
	// StrategyInvalid indicates any strategy that is unsupported
	// or returned when no strategy is applicable.
	StrategyInvalid = iota

	// StrategyOrderedList is a sequential list of servers: if the
	// first server cannot be reached, the next is used. The
	// client will proceed in this manner until the list of
	// servers is exhausted, and then an error is returned.
	StrategyOrderedList
)

var strategyStrings = map[string]Strategy{
	"ordered_list": StrategyOrderedList,
}

// StrategyFromString takes a string describing a
func StrategyFromString(s string) Strategy {
	s = strings.TrimSpace(strings.ToLower(s))
	strategy, ok := strategyStrings[s]
	if !ok {
		return StrategyInvalid
	}
	return strategy
}

// NewGroup will use the collection of remotes specified with the
// given strategy.
func NewGroup(remotes []string, tlsConfig *tls.Config, strategy Strategy) (Remote, error) {
	var servers = make([]*server, len(remotes))
	for i := range remotes {
		u, err := normalizeURL(remotes[i])
		if err != nil {
			return nil, err
		}
		servers[i] = newServer(u, tlsConfig)
	}

	switch strategy {
	case StrategyOrderedList:
		return newOrdererdListGroup(servers)
	default:
		return nil, errors.New("unrecognised strategy")
	}
}

type orderedListGroup struct {
	remotes []*server
}

func (g *orderedListGroup) Hosts() []string {
	var hosts = make([]string, 0, len(g.remotes))
	for _, srv := range g.remotes {
		srvHosts := srv.Hosts()
		hosts = append(hosts, srvHosts[0])
	}
	return hosts
}

func (g *orderedListGroup) SetRequestTimeout(timeout time.Duration) {
	for _, srv := range g.remotes {
		srv.SetRequestTimeout(timeout)
	}
}

func (g *orderedListGroup) SetProxy(proxy func(*http.Request) (*url.URL, error)) {
	for _, srv := range g.remotes {
		srv.SetProxy(proxy)
	}
}

func newOrdererdListGroup(remotes []*server) (Remote, error) {
	return &orderedListGroup{
		remotes: remotes,
	}, nil
}

func (g *orderedListGroup) AuthSign(req, id []byte, provider auth.Provider) (resp []byte, err error) {
	for i := range g.remotes {
		resp, err = g.remotes[i].AuthSign(req, id, provider)
		if err == nil {
			return resp, nil
		}
	}

	return nil, err
}

func (g *orderedListGroup) Sign(jsonData []byte) (resp []byte, err error) {
	for i := range g.remotes {
		resp, err = g.remotes[i].Sign(jsonData)
		if err == nil {
			return resp, nil
		}
	}

	return nil, err
}

func (g *orderedListGroup) Info(jsonData []byte) (resp *info.Resp, err error) {
	for i := range g.remotes {
		resp, err = g.remotes[i].Info(jsonData)
		if err == nil {
			return resp, nil
		}
	}

	return nil, err
}

// SetReqModifier does nothing because there is no request modifier for group
func (g *orderedListGroup) SetReqModifier(mod func(*http.Request, []byte)) {
	// noop
}
