package libnetwork

import (
	"fmt"
	"math/rand"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/docker/libnetwork/types"
	"github.com/miekg/dns"
	"github.com/sirupsen/logrus"
)

// Resolver represents the embedded DNS server in Docker. It operates
// by listening on container's loopback interface for DNS queries.
type Resolver interface {
	// Start starts the name server for the container
	Start() error
	// Stop stops the name server for the container. Stopped resolver
	// can be reused after running the SetupFunc again.
	Stop()
	// SetupFunc() provides the setup function that should be run
	// in the container's network namespace.
	SetupFunc(int) func()
	// NameServer() returns the IP of the DNS resolver for the
	// containers.
	NameServer() string
	// SetExtServers configures the external nameservers the resolver
	// should use to forward queries
	SetExtServers([]extDNSEntry)
	// ResolverOptions returns resolv.conf options that should be set
	ResolverOptions() []string
}

// DNSBackend represents a backend DNS resolver used for DNS name
// resolution. All the queries to the resolver are forwared to the
// backend resolver.
type DNSBackend interface {
	// ResolveName resolves a service name to an IPv4 or IPv6 address by searching
	// the networks the sandbox is connected to. For IPv6 queries, second return
	// value will be true if the name exists in docker domain but doesn't have an
	// IPv6 address. Such queries shouldn't be forwarded to external nameservers.
	ResolveName(name string, iplen int) ([]net.IP, bool)
	// ResolveIP returns the service name for the passed in IP. IP is in reverse dotted
	// notation; the format used for DNS PTR records
	ResolveIP(name string) string
	// ResolveService returns all the backend details about the containers or hosts
	// backing a service. Its purpose is to satisfy an SRV query
	ResolveService(name string) ([]*net.SRV, []net.IP)
	// ExecFunc allows a function to be executed in the context of the backend
	// on behalf of the resolver.
	ExecFunc(f func()) error
	//NdotsSet queries the backends ndots dns option settings
	NdotsSet() bool
	// HandleQueryResp passes the name & IP from a response to the backend. backend
	// can use it to maintain any required state about the resolution
	HandleQueryResp(name string, ip net.IP)
}

const (
	dnsPort         = "53"
	ptrIPv4domain   = ".in-addr.arpa."
	ptrIPv6domain   = ".ip6.arpa."
	respTTL         = 600
	maxExtDNS       = 3 //max number of external servers to try
	extIOTimeout    = 4 * time.Second
	defaultRespSize = 512
	maxConcurrent   = 100
	logInterval     = 2 * time.Second
)

type extDNSEntry struct {
	IPStr        string
	HostLoopback bool
}

// resolver implements the Resolver interface
type resolver struct {
	backend       DNSBackend
	extDNSList    [maxExtDNS]extDNSEntry
	server        *dns.Server
	conn          *net.UDPConn
	tcpServer     *dns.Server
	tcpListen     *net.TCPListener
	err           error
	count         int32
	tStamp        time.Time
	queryLock     sync.Mutex
	listenAddress string
	proxyDNS      bool
	resolverKey   string
	startCh       chan struct{}
}

func init() {
	rand.Seed(time.Now().Unix())
}

// NewResolver creates a new instance of the Resolver
func NewResolver(address string, proxyDNS bool, resolverKey string, backend DNSBackend) Resolver {
	return &resolver{
		backend:       backend,
		proxyDNS:      proxyDNS,
		listenAddress: address,
		resolverKey:   resolverKey,
		err:           fmt.Errorf("setup not done yet"),
		startCh:       make(chan struct{}, 1),
	}
}

func (r *resolver) SetupFunc(port int) func() {
	return (func() {
		var err error

		// DNS operates primarily on UDP
		addr := &net.UDPAddr{
			IP:   net.ParseIP(r.listenAddress),
			Port: port,
		}

		r.conn, err = net.ListenUDP("udp", addr)
		if err != nil {
			r.err = fmt.Errorf("error in opening name server socket %v", err)
			return
		}

		// Listen on a TCP as well
		tcpaddr := &net.TCPAddr{
			IP:   net.ParseIP(r.listenAddress),
			Port: port,
		}

		r.tcpListen, err = net.ListenTCP("tcp", tcpaddr)
		if err != nil {
			r.err = fmt.Errorf("error in opening name TCP server socket %v", err)
			return
		}
		r.err = nil
	})
}

func (r *resolver) Start() error {
	r.startCh <- struct{}{}
	defer func() { <-r.startCh }()

	// make sure the resolver has been setup before starting
	if r.err != nil {
		return r.err
	}

	if err := r.setupIPTable(); err != nil {
		return fmt.Errorf("setting up IP table rules failed: %v", err)
	}

	s := &dns.Server{Handler: r, PacketConn: r.conn}
	r.server = s
	go func() {
		s.ActivateAndServe()
	}()

	tcpServer := &dns.Server{Handler: r, Listener: r.tcpListen}
	r.tcpServer = tcpServer
	go func() {
		tcpServer.ActivateAndServe()
	}()
	return nil
}

func (r *resolver) Stop() {
	r.startCh <- struct{}{}
	defer func() { <-r.startCh }()

	if r.server != nil {
		r.server.Shutdown()
	}
	if r.tcpServer != nil {
		r.tcpServer.Shutdown()
	}
	r.conn = nil
	r.tcpServer = nil
	r.err = fmt.Errorf("setup not done yet")
	r.tStamp = time.Time{}
	r.count = 0
	r.queryLock = sync.Mutex{}
}

func (r *resolver) SetExtServers(extDNS []extDNSEntry) {
	l := len(extDNS)
	if l > maxExtDNS {
		l = maxExtDNS
	}
	for i := 0; i < l; i++ {
		r.extDNSList[i] = extDNS[i]
	}
}

func (r *resolver) NameServer() string {
	return r.listenAddress
}

func (r *resolver) ResolverOptions() []string {
	return []string{"ndots:0"}
}

func setCommonFlags(msg *dns.Msg) {
	msg.RecursionAvailable = true
}

func shuffleAddr(addr []net.IP) []net.IP {
	for i := len(addr) - 1; i > 0; i-- {
		r := rand.Intn(i + 1)
		addr[i], addr[r] = addr[r], addr[i]
	}
	return addr
}

func createRespMsg(query *dns.Msg) *dns.Msg {
	resp := new(dns.Msg)
	resp.SetReply(query)
	setCommonFlags(resp)

	return resp
}

func (r *resolver) handleIPQuery(name string, query *dns.Msg, ipType int) (*dns.Msg, error) {
	var addr []net.IP
	var ipv6Miss bool
	addr, ipv6Miss = r.backend.ResolveName(name, ipType)

	if addr == nil && ipv6Miss {
		// Send a reply without any Answer sections
		logrus.Debugf("[resolver] lookup name %s present without IPv6 address", name)
		resp := createRespMsg(query)
		return resp, nil
	}
	if addr == nil {
		return nil, nil
	}

	logrus.Debugf("[resolver] lookup for %s: IP %v", name, addr)

	resp := createRespMsg(query)
	if len(addr) > 1 {
		addr = shuffleAddr(addr)
	}
	if ipType == types.IPv4 {
		for _, ip := range addr {
			rr := new(dns.A)
			rr.Hdr = dns.RR_Header{Name: name, Rrtype: dns.TypeA, Class: dns.ClassINET, Ttl: respTTL}
			rr.A = ip
			resp.Answer = append(resp.Answer, rr)
		}
	} else {
		for _, ip := range addr {
			rr := new(dns.AAAA)
			rr.Hdr = dns.RR_Header{Name: name, Rrtype: dns.TypeAAAA, Class: dns.ClassINET, Ttl: respTTL}
			rr.AAAA = ip
			resp.Answer = append(resp.Answer, rr)
		}
	}
	return resp, nil
}

func (r *resolver) handlePTRQuery(ptr string, query *dns.Msg) (*dns.Msg, error) {
	parts := []string{}

	if strings.HasSuffix(ptr, ptrIPv4domain) {
		parts = strings.Split(ptr, ptrIPv4domain)
	} else if strings.HasSuffix(ptr, ptrIPv6domain) {
		parts = strings.Split(ptr, ptrIPv6domain)
	} else {
		return nil, fmt.Errorf("invalid PTR query, %v", ptr)
	}

	host := r.backend.ResolveIP(parts[0])

	if len(host) == 0 {
		return nil, nil
	}

	logrus.Debugf("[resolver] lookup for IP %s: name %s", parts[0], host)
	fqdn := dns.Fqdn(host)

	resp := new(dns.Msg)
	resp.SetReply(query)
	setCommonFlags(resp)

	rr := new(dns.PTR)
	rr.Hdr = dns.RR_Header{Name: ptr, Rrtype: dns.TypePTR, Class: dns.ClassINET, Ttl: respTTL}
	rr.Ptr = fqdn
	resp.Answer = append(resp.Answer, rr)
	return resp, nil
}

func (r *resolver) handleSRVQuery(svc string, query *dns.Msg) (*dns.Msg, error) {

	srv, ip := r.backend.ResolveService(svc)

	if len(srv) == 0 {
		return nil, nil
	}
	if len(srv) != len(ip) {
		return nil, fmt.Errorf("invalid reply for SRV query %s", svc)
	}

	resp := createRespMsg(query)

	for i, r := range srv {
		rr := new(dns.SRV)
		rr.Hdr = dns.RR_Header{Name: svc, Rrtype: dns.TypePTR, Class: dns.ClassINET, Ttl: respTTL}
		rr.Port = r.Port
		rr.Target = r.Target
		resp.Answer = append(resp.Answer, rr)

		rr1 := new(dns.A)
		rr1.Hdr = dns.RR_Header{Name: r.Target, Rrtype: dns.TypeA, Class: dns.ClassINET, Ttl: respTTL}
		rr1.A = ip[i]
		resp.Extra = append(resp.Extra, rr1)
	}
	return resp, nil

}

func truncateResp(resp *dns.Msg, maxSize int, isTCP bool) {
	if !isTCP {
		resp.Truncated = true
	}

	srv := resp.Question[0].Qtype == dns.TypeSRV
	// trim the Answer RRs one by one till the whole message fits
	// within the reply size
	for resp.Len() > maxSize {
		resp.Answer = resp.Answer[:len(resp.Answer)-1]

		if srv && len(resp.Extra) > 0 {
			resp.Extra = resp.Extra[:len(resp.Extra)-1]
		}
	}
}

func (r *resolver) ServeDNS(w dns.ResponseWriter, query *dns.Msg) {
	var (
		extConn net.Conn
		resp    *dns.Msg
		err     error
	)

	if query == nil || len(query.Question) == 0 {
		return
	}
	name := query.Question[0].Name

	switch query.Question[0].Qtype {
	case dns.TypeA:
		resp, err = r.handleIPQuery(name, query, types.IPv4)
	case dns.TypeAAAA:
		resp, err = r.handleIPQuery(name, query, types.IPv6)
	case dns.TypePTR:
		resp, err = r.handlePTRQuery(name, query)
	case dns.TypeSRV:
		resp, err = r.handleSRVQuery(name, query)
	}

	if err != nil {
		logrus.Error(err)
		return
	}

	if resp == nil {
		// If the backend doesn't support proxying dns request
		// fail the response
		if !r.proxyDNS {
			resp = new(dns.Msg)
			resp.SetRcode(query, dns.RcodeServerFailure)
			w.WriteMsg(resp)
			return
		}

		// If the user sets ndots > 0 explicitly and the query is
		// in the root domain don't forward it out. We will return
		// failure and let the client retry with the search domain
		// attached
		switch query.Question[0].Qtype {
		case dns.TypeA:
			fallthrough
		case dns.TypeAAAA:
			if r.backend.NdotsSet() && !strings.Contains(strings.TrimSuffix(name, "."), ".") {
				resp = createRespMsg(query)
			}
		}
	}

	proto := w.LocalAddr().Network()
	maxSize := 0
	if proto == "tcp" {
		maxSize = dns.MaxMsgSize - 1
	} else if proto == "udp" {
		optRR := query.IsEdns0()
		if optRR != nil {
			maxSize = int(optRR.UDPSize())
		}
		if maxSize < defaultRespSize {
			maxSize = defaultRespSize
		}
	}

	if resp != nil {
		if resp.Len() > maxSize {
			truncateResp(resp, maxSize, proto == "tcp")
		}
	} else {
		for i := 0; i < maxExtDNS; i++ {
			extDNS := &r.extDNSList[i]
			if extDNS.IPStr == "" {
				break
			}
			extConnect := func() {
				addr := fmt.Sprintf("%s:%d", extDNS.IPStr, 53)
				extConn, err = net.DialTimeout(proto, addr, extIOTimeout)
			}

			if extDNS.HostLoopback {
				extConnect()
			} else {
				execErr := r.backend.ExecFunc(extConnect)
				if execErr != nil {
					logrus.Warn(execErr)
					continue
				}
			}
			if err != nil {
				logrus.Warnf("[resolver] connect failed: %s", err)
				continue
			}

			queryType := dns.TypeToString[query.Question[0].Qtype]
			logrus.Debugf("[resolver] query %s (%s) from %s, forwarding to %s:%s", name, queryType,
				extConn.LocalAddr().String(), proto, extDNS.IPStr)

			// Timeout has to be set for every IO operation.
			extConn.SetDeadline(time.Now().Add(extIOTimeout))
			co := &dns.Conn{
				Conn:    extConn,
				UDPSize: uint16(maxSize),
			}
			defer co.Close()

			// limits the number of outstanding concurrent queries.
			if !r.forwardQueryStart() {
				old := r.tStamp
				r.tStamp = time.Now()
				if r.tStamp.Sub(old) > logInterval {
					logrus.Errorf("[resolver] more than %v concurrent queries from %s", maxConcurrent, extConn.LocalAddr().String())
				}
				continue
			}

			err = co.WriteMsg(query)
			if err != nil {
				r.forwardQueryEnd()
				logrus.Debugf("[resolver] send to DNS server failed, %s", err)
				continue
			}

			resp, err = co.ReadMsg()
			// Truncated DNS replies should be sent to the client so that the
			// client can retry over TCP
			if err != nil && err != dns.ErrTruncated {
				r.forwardQueryEnd()
				logrus.Debugf("[resolver] read from DNS server failed, %s", err)
				continue
			}
			r.forwardQueryEnd()
			if resp != nil {
				answers := 0
				for _, rr := range resp.Answer {
					h := rr.Header()
					switch h.Rrtype {
					case dns.TypeA:
						answers++
						ip := rr.(*dns.A).A
						logrus.Debugf("[resolver] received A record %q for %q from %s:%s", ip, h.Name, proto, extDNS.IPStr)
						r.backend.HandleQueryResp(h.Name, ip)
					case dns.TypeAAAA:
						answers++
						ip := rr.(*dns.AAAA).AAAA
						logrus.Debugf("[resolver] received AAAA record %q for %q from %s:%s", ip, h.Name, proto, extDNS.IPStr)
						r.backend.HandleQueryResp(h.Name, ip)
					}
				}
				if resp.Answer == nil || answers == 0 {
					logrus.Debugf("[resolver] external DNS %s:%s did not return any %s records for %q", proto, extDNS.IPStr, queryType, name)
				}
			} else {
				logrus.Debugf("[resolver] external DNS %s:%s returned empty response for %q", proto, extDNS.IPStr, name)
			}
			resp.Compress = true
			break
		}
		if resp == nil {
			return
		}
	}

	if err = w.WriteMsg(resp); err != nil {
		logrus.Errorf("[resolver] error writing resolver resp, %s", err)
	}
}

func (r *resolver) forwardQueryStart() bool {
	r.queryLock.Lock()
	defer r.queryLock.Unlock()

	if r.count == maxConcurrent {
		return false
	}
	r.count++

	return true
}

func (r *resolver) forwardQueryEnd() {
	r.queryLock.Lock()
	defer r.queryLock.Unlock()

	if r.count == 0 {
		logrus.Error("[resolver] invalid concurrent query count")
	} else {
		r.count--
	}
}
