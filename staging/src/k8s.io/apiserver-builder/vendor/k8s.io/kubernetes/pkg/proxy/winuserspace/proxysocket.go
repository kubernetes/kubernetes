/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package winuserspace

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/ipconfig"
)

const (
	// Kubernetes DNS suffix search list
	// TODO: Get DNS suffix search list from docker containers.
	// --dns-search option doesn't work on Windows containers and has been
	// fixed recently in docker.

	// Kubernetes cluster domain
	clusterDomain = "cluster.local"

	// Kubernetes service domain
	serviceDomain = "svc." + clusterDomain

	// Kubernetes default namespace domain
	namespaceServiceDomain = "default." + serviceDomain

	// Kubernetes DNS service port name
	dnsPortName = "dns"

	// DNS TYPE value A (a host address)
	dnsTypeA uint16 = 0x01

	// DNS TYPE value AAAA (a host IPv6 address)
	dnsTypeAAAA uint16 = 0x1c

	// DNS CLASS value IN (the Internet)
	dnsClassInternet uint16 = 0x01
)

// Abstraction over TCP/UDP sockets which are proxied.
type proxySocket interface {
	// Addr gets the net.Addr for a proxySocket.
	Addr() net.Addr
	// Close stops the proxySocket from accepting incoming connections.
	// Each implementation should comment on the impact of calling Close
	// while sessions are active.
	Close() error
	// ProxyLoop proxies incoming connections for the specified service to the service endpoints.
	ProxyLoop(service ServicePortPortalName, info *serviceInfo, proxier *Proxier)
	// ListenPort returns the host port that the proxySocket is listening on
	ListenPort() int
}

func newProxySocket(protocol api.Protocol, ip net.IP, port int) (proxySocket, error) {
	host := ""
	if ip != nil {
		host = ip.String()
	}

	switch strings.ToUpper(string(protocol)) {
	case "TCP":
		listener, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port)))
		if err != nil {
			return nil, err
		}
		return &tcpProxySocket{Listener: listener, port: port}, nil
	case "UDP":
		addr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(host, strconv.Itoa(port)))
		if err != nil {
			return nil, err
		}
		conn, err := net.ListenUDP("udp", addr)
		if err != nil {
			return nil, err
		}
		return &udpProxySocket{UDPConn: conn, port: port}, nil
	}
	return nil, fmt.Errorf("unknown protocol %q", protocol)
}

// How long we wait for a connection to a backend in seconds
var endpointDialTimeout = []time.Duration{250 * time.Millisecond, 500 * time.Millisecond, 1 * time.Second, 2 * time.Second}

// tcpProxySocket implements proxySocket.  Close() is implemented by net.Listener.  When Close() is called,
// no new connections are allowed but existing connections are left untouched.
type tcpProxySocket struct {
	net.Listener
	port int
}

func (tcp *tcpProxySocket) ListenPort() int {
	return tcp.port
}

func tryConnect(service ServicePortPortalName, srcAddr net.Addr, protocol string, proxier *Proxier) (out net.Conn, err error) {
	sessionAffinityReset := false
	for _, dialTimeout := range endpointDialTimeout {
		servicePortName := proxy.ServicePortName{
			NamespacedName: types.NamespacedName{
				Namespace: service.Namespace,
				Name:      service.Name,
			},
			Port: service.Port,
		}
		endpoint, err := proxier.loadBalancer.NextEndpoint(servicePortName, srcAddr, sessionAffinityReset)
		if err != nil {
			glog.Errorf("Couldn't find an endpoint for %s: %v", service, err)
			return nil, err
		}
		glog.V(3).Infof("Mapped service %q to endpoint %s", service, endpoint)
		// TODO: This could spin up a new goroutine to make the outbound connection,
		// and keep accepting inbound traffic.
		outConn, err := net.DialTimeout(protocol, endpoint, dialTimeout)
		if err != nil {
			if isTooManyFDsError(err) {
				panic("Dial failed: " + err.Error())
			}
			glog.Errorf("Dial failed: %v", err)
			sessionAffinityReset = true
			continue
		}
		return outConn, nil
	}
	return nil, fmt.Errorf("failed to connect to an endpoint.")
}

func (tcp *tcpProxySocket) ProxyLoop(service ServicePortPortalName, myInfo *serviceInfo, proxier *Proxier) {
	for {
		if !myInfo.isAlive() {
			// The service port was closed or replaced.
			return
		}
		// Block until a connection is made.
		inConn, err := tcp.Accept()
		if err != nil {
			if isTooManyFDsError(err) {
				panic("Accept failed: " + err.Error())
			}

			if isClosedError(err) {
				return
			}
			if !myInfo.isAlive() {
				// Then the service port was just closed so the accept failure is to be expected.
				return
			}
			glog.Errorf("Accept failed: %v", err)
			continue
		}
		glog.V(3).Infof("Accepted TCP connection from %v to %v", inConn.RemoteAddr(), inConn.LocalAddr())
		outConn, err := tryConnect(service, inConn.(*net.TCPConn).RemoteAddr(), "tcp", proxier)
		if err != nil {
			glog.Errorf("Failed to connect to balancer: %v", err)
			inConn.Close()
			continue
		}
		// Spin up an async copy loop.
		go proxyTCP(inConn.(*net.TCPConn), outConn.(*net.TCPConn))
	}
}

// proxyTCP proxies data bi-directionally between in and out.
func proxyTCP(in, out *net.TCPConn) {
	var wg sync.WaitGroup
	wg.Add(2)
	glog.V(4).Infof("Creating proxy between %v <-> %v <-> %v <-> %v",
		in.RemoteAddr(), in.LocalAddr(), out.LocalAddr(), out.RemoteAddr())
	go copyBytes("from backend", in, out, &wg)
	go copyBytes("to backend", out, in, &wg)
	wg.Wait()
}

func copyBytes(direction string, dest, src *net.TCPConn, wg *sync.WaitGroup) {
	defer wg.Done()
	glog.V(4).Infof("Copying %s: %s -> %s", direction, src.RemoteAddr(), dest.RemoteAddr())
	n, err := io.Copy(dest, src)
	if err != nil {
		if !isClosedError(err) {
			glog.Errorf("I/O error: %v", err)
		}
	}
	glog.V(4).Infof("Copied %d bytes %s: %s -> %s", n, direction, src.RemoteAddr(), dest.RemoteAddr())
	dest.Close()
	src.Close()
}

// udpProxySocket implements proxySocket.  Close() is implemented by net.UDPConn.  When Close() is called,
// no new connections are allowed and existing connections are broken.
// TODO: We could lame-duck this ourselves, if it becomes important.
type udpProxySocket struct {
	*net.UDPConn
	port int
}

func (udp *udpProxySocket) ListenPort() int {
	return udp.port
}

func (udp *udpProxySocket) Addr() net.Addr {
	return udp.LocalAddr()
}

// Holds all the known UDP clients that have not timed out.
type clientCache struct {
	mu      sync.Mutex
	clients map[string]net.Conn // addr string -> connection
}

func newClientCache() *clientCache {
	return &clientCache{clients: map[string]net.Conn{}}
}

// TODO: use Go net dnsmsg library to walk DNS message format
// DNS packet header
type dnsHeader struct {
	id      uint16
	bits    uint16
	qdCount uint16
	anCount uint16
	nsCount uint16
	arCount uint16
}

// DNS domain name
type dnsDomainName struct {
	name string
}

// DNS packet question section
type dnsQuestion struct {
	qName  dnsDomainName
	qType  uint16
	qClass uint16
}

// DNS message, only interested in question now
type dnsMsg struct {
	header   dnsHeader
	question []dnsQuestion
}

type dnsStruct interface {
	walk(f func(field interface{}) (ok bool)) (ok bool)
}

func (header *dnsHeader) walk(f func(field interface{}) bool) bool {
	return f(&header.id) &&
		f(&header.bits) &&
		f(&header.qdCount) &&
		f(&header.anCount) &&
		f(&header.nsCount) &&
		f(&header.arCount)
}

func (question *dnsQuestion) walk(f func(field interface{}) bool) bool {
	return f(&question.qName) &&
		f(&question.qType) &&
		f(&question.qClass)
}

func packDomainName(name string, buffer []byte, index int) (newIndex int, ok bool) {
	if name == "" {
		buffer[index] = 0
		index++
		return index, true
	}

	// one more dot plus trailing 0
	if index+len(name)+2 > len(buffer) {
		return len(buffer), false
	}

	domains := strings.Split(name, ".")
	for _, domain := range domains {
		domainLen := len(domain)
		if domainLen == 0 {
			return len(buffer), false
		}
		buffer[index] = byte(domainLen)
		index++
		copy(buffer[index:index+domainLen], domain)
		index += domainLen
	}

	buffer[index] = 0
	index++
	return index, true
}

func unpackDomainName(buffer []byte, index int) (name string, newIndex int, ok bool) {
	name = ""

	for index < len(buffer) {
		cnt := int(buffer[index])
		index++
		if cnt == 0 {
			break
		}

		if index+cnt > len(buffer) {
			return "", len(buffer), false
		}
		if name != "" {
			name += "."
		}
		name += string(buffer[index : index+cnt])
		index += cnt
	}

	if index >= len(buffer) {
		return "", len(buffer), false
	}
	return name, index, true
}

func packStruct(any dnsStruct, buffer []byte, index int) (newIndex int, ok bool) {
	ok = any.walk(func(field interface{}) bool {
		switch value := field.(type) {
		case *uint16:
			if index+2 > len(buffer) {
				return false
			}
			binary.BigEndian.PutUint16(buffer[index:index+2], *value)
			index += 2
			return true
		case *dnsDomainName:
			index, ok = packDomainName((*value).name, buffer, index)
			return ok
		default:
			return false
		}
	})

	if !ok {
		return len(buffer), false
	}
	return index, true
}

func unpackStruct(any dnsStruct, buffer []byte, index int) (newIndex int, ok bool) {
	ok = any.walk(func(field interface{}) bool {
		switch value := field.(type) {
		case *uint16:
			if index+2 > len(buffer) {
				return false
			}
			*value = binary.BigEndian.Uint16(buffer[index : index+2])
			index += 2
			return true
		case *dnsDomainName:
			(*value).name, index, ok = unpackDomainName(buffer, index)
			return ok
		default:
			return false
		}
	})

	if !ok {
		return len(buffer), false
	}
	return index, true
}

// Pack the message structure into buffer
func (msg *dnsMsg) packDnsMsg(buffer []byte) (length int, ok bool) {
	index := 0

	if index, ok = packStruct(&msg.header, buffer, index); !ok {
		return len(buffer), false
	}

	for i := 0; i < len(msg.question); i++ {
		if index, ok = packStruct(&msg.question[i], buffer, index); !ok {
			return len(buffer), false
		}
	}
	return index, true
}

// Unpack the buffer into the message structure
func (msg *dnsMsg) unpackDnsMsg(buffer []byte) (ok bool) {
	index := 0

	if index, ok = unpackStruct(&msg.header, buffer, index); !ok {
		return false
	}

	msg.question = make([]dnsQuestion, msg.header.qdCount)
	for i := 0; i < len(msg.question); i++ {
		if index, ok = unpackStruct(&msg.question[i], buffer, index); !ok {
			return false
		}
	}
	return true
}

// DNS query client classified by address and QTYPE
type dnsClientQuery struct {
	clientAddress string
	dnsQType      uint16
}

// Holds DNS client query, the value contains the index in DNS suffix search list,
// the original DNS message and length for the same client and QTYPE
type dnsClientCache struct {
	mu      sync.Mutex
	clients map[dnsClientQuery]*dnsQueryState
}

type dnsQueryState struct {
	searchIndex int32
	msg         *dnsMsg
}

func newDnsClientCache() *dnsClientCache {
	return &dnsClientCache{clients: map[dnsClientQuery]*dnsQueryState{}}
}

func packetRequiresDnsSuffix(dnsType, dnsClass uint16) bool {
	return (dnsType == dnsTypeA || dnsType == dnsTypeAAAA) && dnsClass == dnsClassInternet
}

func isDnsService(portName string) bool {
	return portName == dnsPortName
}

func appendDnsSuffix(msg *dnsMsg, buffer []byte, length int, dnsSuffix string) int {
	if msg == nil || len(msg.question) == 0 {
		glog.Warning("DNS message parameter is invalid.")
		return length
	}

	// Save the original name since it will be reused for next iteration
	origName := msg.question[0].qName.name
	if dnsSuffix != "" {
		msg.question[0].qName.name += "." + dnsSuffix
	}
	len, ok := msg.packDnsMsg(buffer)
	msg.question[0].qName.name = origName

	if !ok {
		glog.Warning("Unable to pack DNS packet.")
		return length
	}

	return len
}

func processUnpackedDnsQueryPacket(dnsClients *dnsClientCache, msg *dnsMsg, host string, dnsQType uint16, buffer []byte, length int, dnsSearch []string) int {
	if dnsSearch == nil || len(dnsSearch) == 0 {
		glog.V(1).Infof("DNS search list is not initialized and is empty.")
		return length
	}

	// TODO: handle concurrent queries from a client
	dnsClients.mu.Lock()
	state, found := dnsClients.clients[dnsClientQuery{host, dnsQType}]
	if !found {
		state = &dnsQueryState{0, msg}
		dnsClients.clients[dnsClientQuery{host, dnsQType}] = state
	}
	dnsClients.mu.Unlock()

	index := atomic.SwapInt32(&state.searchIndex, state.searchIndex+1)
	// Also update message ID if the client retries due to previous query time out
	state.msg.header.id = msg.header.id

	if index < 0 || index >= int32(len(dnsSearch)) {
		glog.V(1).Infof("Search index %d is out of range.", index)
		return length
	}

	length = appendDnsSuffix(msg, buffer, length, dnsSearch[index])
	return length
}

func processUnpackedDnsResponsePacket(svrConn net.Conn, dnsClients *dnsClientCache, rcode uint16, host string, dnsQType uint16, buffer []byte, length int, dnsSearch []string) bool {
	var drop bool
	if dnsSearch == nil || len(dnsSearch) == 0 {
		glog.V(1).Infof("DNS search list is not initialized and is empty.")
		return drop
	}

	dnsClients.mu.Lock()
	state, found := dnsClients.clients[dnsClientQuery{host, dnsQType}]
	dnsClients.mu.Unlock()

	if found {
		index := atomic.SwapInt32(&state.searchIndex, state.searchIndex+1)
		if rcode != 0 && index >= 0 && index < int32(len(dnsSearch)) {
			// If the reponse has failure and iteration through the search list has not
			// reached the end, retry on behalf of the client using the original query message
			drop = true
			length = appendDnsSuffix(state.msg, buffer, length, dnsSearch[index])

			_, err := svrConn.Write(buffer[0:length])
			if err != nil {
				if !logTimeout(err) {
					glog.Errorf("Write failed: %v", err)
				}
			}
		} else {
			dnsClients.mu.Lock()
			delete(dnsClients.clients, dnsClientQuery{host, dnsQType})
			dnsClients.mu.Unlock()
		}
	}

	return drop
}

func processDnsQueryPacket(dnsClients *dnsClientCache, cliAddr net.Addr, buffer []byte, length int, dnsSearch []string) int {
	msg := &dnsMsg{}
	if !msg.unpackDnsMsg(buffer[:length]) {
		glog.Warning("Unable to unpack DNS packet.")
		return length
	}

	// Query - Response bit that specifies whether this message is a query (0) or a response (1).
	qr := msg.header.bits & 0x8000
	if qr != 0 {
		glog.Warning("DNS packet should be a query message.")
		return length
	}

	// QDCOUNT
	if msg.header.qdCount != 1 {
		glog.V(1).Infof("Number of entries in the question section of the DNS packet is: %d", msg.header.qdCount)
		glog.V(1).Infof("DNS suffix appending does not support more than one question.")
		return length
	}

	// ANCOUNT, NSCOUNT, ARCOUNT
	if msg.header.anCount != 0 || msg.header.nsCount != 0 || msg.header.arCount != 0 {
		glog.V(1).Infof("DNS packet contains more than question section.")
		return length
	}

	dnsQType := msg.question[0].qType
	dnsQClass := msg.question[0].qClass
	if packetRequiresDnsSuffix(dnsQType, dnsQClass) {
		host, _, err := net.SplitHostPort(cliAddr.String())
		if err != nil {
			glog.V(1).Infof("Failed to get host from client address: %v", err)
			host = cliAddr.String()
		}

		length = processUnpackedDnsQueryPacket(dnsClients, msg, host, dnsQType, buffer, length, dnsSearch)
	}

	return length
}

func processDnsResponsePacket(svrConn net.Conn, dnsClients *dnsClientCache, cliAddr net.Addr, buffer []byte, length int, dnsSearch []string) bool {
	var drop bool
	msg := &dnsMsg{}
	if !msg.unpackDnsMsg(buffer[:length]) {
		glog.Warning("Unable to unpack DNS packet.")
		return drop
	}

	// Query - Response bit that specifies whether this message is a query (0) or a response (1).
	qr := msg.header.bits & 0x8000
	if qr == 0 {
		glog.Warning("DNS packet should be a response message.")
		return drop
	}

	// QDCOUNT
	if msg.header.qdCount != 1 {
		glog.V(1).Infof("Number of entries in the reponse section of the DNS packet is: %d", msg.header.qdCount)
		return drop
	}

	dnsQType := msg.question[0].qType
	dnsQClass := msg.question[0].qClass
	if packetRequiresDnsSuffix(dnsQType, dnsQClass) {
		host, _, err := net.SplitHostPort(cliAddr.String())
		if err != nil {
			glog.V(1).Infof("Failed to get host from client address: %v", err)
			host = cliAddr.String()
		}

		rcode := msg.header.bits & 0xf
		drop = processUnpackedDnsResponsePacket(svrConn, dnsClients, rcode, host, dnsQType, buffer, length, dnsSearch)
	}

	return drop
}

func (udp *udpProxySocket) ProxyLoop(service ServicePortPortalName, myInfo *serviceInfo, proxier *Proxier) {
	var buffer [4096]byte // 4KiB should be enough for most whole-packets
	var dnsSearch []string
	if isDnsService(service.Port) {
		dnsSearch = []string{"", namespaceServiceDomain, serviceDomain, clusterDomain}
		execer := exec.New()
		ipconfigInterface := ipconfig.New(execer)
		suffixList, err := ipconfigInterface.GetDnsSuffixSearchList()
		if err == nil {
			for _, suffix := range suffixList {
				dnsSearch = append(dnsSearch, suffix)
			}
		}
	}

	for {
		if !myInfo.isAlive() {
			// The service port was closed or replaced.
			break
		}

		// Block until data arrives.
		// TODO: Accumulate a histogram of n or something, to fine tune the buffer size.
		n, cliAddr, err := udp.ReadFrom(buffer[0:])
		if err != nil {
			if e, ok := err.(net.Error); ok {
				if e.Temporary() {
					glog.V(1).Infof("ReadFrom had a temporary failure: %v", err)
					continue
				}
			}
			glog.Errorf("ReadFrom failed, exiting ProxyLoop: %v", err)
			break
		}

		// If this is DNS query packet
		if isDnsService(service.Port) {
			n = processDnsQueryPacket(myInfo.dnsClients, cliAddr, buffer[:], n, dnsSearch)
		}

		// If this is a client we know already, reuse the connection and goroutine.
		svrConn, err := udp.getBackendConn(myInfo.activeClients, myInfo.dnsClients, cliAddr, proxier, service, myInfo.timeout, dnsSearch)
		if err != nil {
			continue
		}
		// TODO: It would be nice to let the goroutine handle this write, but we don't
		// really want to copy the buffer.  We could do a pool of buffers or something.
		_, err = svrConn.Write(buffer[0:n])
		if err != nil {
			if !logTimeout(err) {
				glog.Errorf("Write failed: %v", err)
				// TODO: Maybe tear down the goroutine for this client/server pair?
			}
			continue
		}
		err = svrConn.SetDeadline(time.Now().Add(myInfo.timeout))
		if err != nil {
			glog.Errorf("SetDeadline failed: %v", err)
			continue
		}
	}
}

func (udp *udpProxySocket) getBackendConn(activeClients *clientCache, dnsClients *dnsClientCache, cliAddr net.Addr, proxier *Proxier, service ServicePortPortalName, timeout time.Duration, dnsSearch []string) (net.Conn, error) {
	activeClients.mu.Lock()
	defer activeClients.mu.Unlock()

	svrConn, found := activeClients.clients[cliAddr.String()]
	if !found {
		// TODO: This could spin up a new goroutine to make the outbound connection,
		// and keep accepting inbound traffic.
		glog.V(3).Infof("New UDP connection from %s", cliAddr)
		var err error
		svrConn, err = tryConnect(service, cliAddr, "udp", proxier)
		if err != nil {
			return nil, err
		}
		if err = svrConn.SetDeadline(time.Now().Add(timeout)); err != nil {
			glog.Errorf("SetDeadline failed: %v", err)
			return nil, err
		}
		activeClients.clients[cliAddr.String()] = svrConn
		go func(cliAddr net.Addr, svrConn net.Conn, activeClients *clientCache, dnsClients *dnsClientCache, service ServicePortPortalName, timeout time.Duration, dnsSearch []string) {
			defer runtime.HandleCrash()
			udp.proxyClient(cliAddr, svrConn, activeClients, dnsClients, service, timeout, dnsSearch)
		}(cliAddr, svrConn, activeClients, dnsClients, service, timeout, dnsSearch)
	}
	return svrConn, nil
}

// This function is expected to be called as a goroutine.
// TODO: Track and log bytes copied, like TCP
func (udp *udpProxySocket) proxyClient(cliAddr net.Addr, svrConn net.Conn, activeClients *clientCache, dnsClients *dnsClientCache, service ServicePortPortalName, timeout time.Duration, dnsSearch []string) {
	defer svrConn.Close()
	var buffer [4096]byte
	for {
		n, err := svrConn.Read(buffer[0:])
		if err != nil {
			if !logTimeout(err) {
				glog.Errorf("Read failed: %v", err)
			}
			break
		}

		drop := false
		if isDnsService(service.Port) {
			drop = processDnsResponsePacket(svrConn, dnsClients, cliAddr, buffer[:], n, dnsSearch)
		}

		if !drop {
			err = svrConn.SetDeadline(time.Now().Add(timeout))
			if err != nil {
				glog.Errorf("SetDeadline failed: %v", err)
				break
			}
			n, err = udp.WriteTo(buffer[0:n], cliAddr)
			if err != nil {
				if !logTimeout(err) {
					glog.Errorf("WriteTo failed: %v", err)
				}
				break
			}
		}
	}
	activeClients.mu.Lock()
	delete(activeClients.clients, cliAddr.String())
	activeClients.mu.Unlock()
}
