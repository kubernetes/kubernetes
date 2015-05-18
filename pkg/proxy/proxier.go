/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package proxy

import (
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/iptables"
	"github.com/golang/glog"
)

type serviceInfo struct {
	portalIP            net.IP
	portalPort          int
	protocol            api.Protocol
	proxyPort           int
	socket              proxySocket
	timeout             time.Duration
	loadBalancerStatus  api.LoadBalancerStatus
	sessionAffinityType api.ServiceAffinity
	stickyMaxAgeMinutes int
}

func logTimeout(err error) bool {
	if e, ok := err.(net.Error); ok {
		if e.Timeout() {
			glog.V(3).Infof("connection to endpoint closed due to inactivity")
			return true
		}
	}
	return false
}

// Proxier is a simple proxy for TCP connections between a localhost:lport
// and services that provide the actual implementations.
type Proxier struct {
	loadBalancer  LoadBalancer
	mu            sync.Mutex // protects serviceMap
	serviceMap    map[ServicePortName]*serviceInfo
	numProxyLoops int32 // use atomic ops to access this; mostly for testing
	listenIP      net.IP
	iptables      iptables.Interface
	hostIP        net.IP
}

var (
	// ErrProxyOnLocalhost is returned by NewProxier if the user requests a proxier on
	// the loopback address. May be checked for by callers of NewProxier to know whether
	// the caller provided invalid input.
	ErrProxyOnLocalhost = fmt.Errorf("cannot proxy on localhost")
)

// IsProxyLocked returns true if the proxy could not acquire the lock on iptables.
func IsProxyLocked(err error) bool {
	return strings.Contains(err.Error(), "holding the xtables lock")
}

// NewProxier returns a new Proxier given a LoadBalancer and an address on
// which to listen.  Because of the iptables logic, It is assumed that there
// is only a single Proxier active on a machine. An error will be returned if
// the proxier cannot be started due to an invalid ListenIP (loopback) or
// if iptables fails to update or acquire the initial lock. Once a proxier is
// created, it will keep iptables up to date in the background and will not
// terminate if a particular iptables call fails.
func NewProxier(loadBalancer LoadBalancer, listenIP net.IP, iptables iptables.Interface) (*Proxier, error) {
	if listenIP.Equal(localhostIPv4) || listenIP.Equal(localhostIPv6) {
		return nil, ErrProxyOnLocalhost
	}

	hostIP, err := util.ChooseHostInterface()
	if err != nil {
		return nil, fmt.Errorf("failed to select a host interface: %v", err)
	}

	glog.V(2).Infof("Setting proxy IP to %v and initializing iptables", hostIP)
	return createProxier(loadBalancer, listenIP, iptables, hostIP)
}

func createProxier(loadBalancer LoadBalancer, listenIP net.IP, iptables iptables.Interface, hostIP net.IP) (*Proxier, error) {
	// Set up the iptables foundations we need.
	if err := iptablesInit(iptables); err != nil {
		return nil, fmt.Errorf("failed to initialize iptables: %v", err)
	}
	// Flush old iptables rules (since the bound ports will be invalid after a restart).
	// When OnUpdate() is first called, the rules will be recreated.
	if err := iptablesFlush(iptables); err != nil {
		return nil, fmt.Errorf("failed to flush iptables: %v", err)
	}
	return &Proxier{
		loadBalancer: loadBalancer,
		serviceMap:   make(map[ServicePortName]*serviceInfo),
		listenIP:     listenIP,
		iptables:     iptables,
		hostIP:       hostIP,
	}, nil
}

// The periodic interval for checking the state of things.
const syncInterval = 5 * time.Second

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	t := time.NewTicker(syncInterval)
	defer t.Stop()
	for {
		<-t.C
		glog.V(6).Infof("Periodic sync")
		if err := iptablesInit(proxier.iptables); err != nil {
			glog.Errorf("Failed to ensure iptables: %v", err)
		}
		proxier.ensurePortals()
		proxier.cleanupStaleStickySessions()
	}
}

// Ensure that portals exist for all services.
func (proxier *Proxier) ensurePortals() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	// NB: This does not remove rules that should not be present.
	for name, info := range proxier.serviceMap {
		err := proxier.openPortal(name, info)
		if err != nil {
			glog.Errorf("Failed to ensure portal for %q: %v", name, err)
		}
	}
}

// clean up any stale sticky session records in the hash map.
func (proxier *Proxier) cleanupStaleStickySessions() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	for name := range proxier.serviceMap {
		proxier.loadBalancer.CleanupStaleStickySessions(name)
	}
}

// This assumes proxier.mu is not locked.
func (proxier *Proxier) stopProxy(service ServicePortName, info *serviceInfo) error {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	return proxier.stopProxyInternal(service, info)
}

// This assumes proxier.mu is locked.
func (proxier *Proxier) stopProxyInternal(service ServicePortName, info *serviceInfo) error {
	delete(proxier.serviceMap, service)
	return info.socket.Close()
}

func (proxier *Proxier) getServiceInfo(service ServicePortName) (*serviceInfo, bool) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	info, ok := proxier.serviceMap[service]
	return info, ok
}

func (proxier *Proxier) setServiceInfo(service ServicePortName, info *serviceInfo) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.serviceMap[service] = info
}

// addServiceOnPort starts listening for a new service, returning the serviceInfo.
// Pass proxyPort=0 to allocate a random port. The timeout only applies to UDP
// connections, for now.
func (proxier *Proxier) addServiceOnPort(service ServicePortName, protocol api.Protocol, proxyPort int, timeout time.Duration) (*serviceInfo, error) {
	sock, err := newProxySocket(protocol, proxier.listenIP, proxyPort)
	if err != nil {
		return nil, err
	}
	_, portStr, err := net.SplitHostPort(sock.Addr().String())
	if err != nil {
		sock.Close()
		return nil, err
	}
	portNum, err := strconv.Atoi(portStr)
	if err != nil {
		sock.Close()
		return nil, err
	}
	si := &serviceInfo{
		proxyPort:           portNum,
		protocol:            protocol,
		socket:              sock,
		timeout:             timeout,
		sessionAffinityType: api.ServiceAffinityNone, // default
		stickyMaxAgeMinutes: 180,                     // TODO: paramaterize this in the API.
	}
	proxier.setServiceInfo(service, si)

	glog.V(2).Infof("Proxying for service %q on %s port %d", service, protocol, portNum)
	go func(service ServicePortName, proxier *Proxier) {
		defer util.HandleCrash()
		atomic.AddInt32(&proxier.numProxyLoops, 1)
		sock.ProxyLoop(service, si, proxier)
		atomic.AddInt32(&proxier.numProxyLoops, -1)
	}(service, proxier)

	return si, nil
}

// How long we leave idle UDP connections open.
const udpIdleTimeout = 10 * time.Second

// OnUpdate manages the active set of service proxies.
// Active service proxies are reinitialized if found in the update set or
// shutdown if missing from the update set.
func (proxier *Proxier) OnUpdate(services []api.Service) {
	glog.V(4).Infof("Received update notice: %+v", services)
	activeServices := make(map[ServicePortName]bool) // use a map as a set
	for i := range services {
		service := &services[i]

		// if PortalIP is "None" or empty, skip proxying
		if !api.IsServiceIPSet(service) {
			glog.V(3).Infof("Skipping service %s due to portal IP = %q", types.NamespacedName{service.Namespace, service.Name}, service.Spec.PortalIP)
			continue
		}

		for i := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[i]

			serviceName := ServicePortName{types.NamespacedName{service.Namespace, service.Name}, servicePort.Name}
			activeServices[serviceName] = true
			serviceIP := net.ParseIP(service.Spec.PortalIP)
			info, exists := proxier.getServiceInfo(serviceName)
			// TODO: check health of the socket?  What if ProxyLoop exited?
			if exists && sameConfig(info, service, servicePort) {
				// Nothing changed.
				continue
			}
			if exists {
				glog.V(4).Infof("Something changed for service %q: stopping it", serviceName)
				err := proxier.closePortal(serviceName, info)
				if err != nil {
					glog.Errorf("Failed to close portal for %q: %v", serviceName, err)
				}
				err = proxier.stopProxy(serviceName, info)
				if err != nil {
					glog.Errorf("Failed to stop service %q: %v", serviceName, err)
				}
			}
			glog.V(1).Infof("Adding new service %q at %s:%d/%s", serviceName, serviceIP, servicePort.Port, servicePort.Protocol)
			info, err := proxier.addServiceOnPort(serviceName, servicePort.Protocol, 0, udpIdleTimeout)
			if err != nil {
				glog.Errorf("Failed to start proxy for %q: %v", serviceName, err)
				continue
			}
			info.portalIP = serviceIP
			info.portalPort = servicePort.Port
			info.loadBalancerStatus = *service.Status.LoadBalancer.DeepCopy()
			info.sessionAffinityType = service.Spec.SessionAffinity
			glog.V(4).Infof("info: %+v", info)

			err = proxier.openPortal(serviceName, info)
			if err != nil {
				glog.Errorf("Failed to open portal for %q: %v", serviceName, err)
			}
			proxier.loadBalancer.NewService(serviceName, info.sessionAffinityType, info.stickyMaxAgeMinutes)
		}
	}
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	for name, info := range proxier.serviceMap {
		if !activeServices[name] {
			glog.V(1).Infof("Stopping service %q", name)
			err := proxier.closePortal(name, info)
			if err != nil {
				glog.Errorf("Failed to close portal for %q: %v", name, err)
			}
			err = proxier.stopProxyInternal(name, info)
			if err != nil {
				glog.Errorf("Failed to stop service %q: %v", name, err)
			}
		}
	}
}

func sameConfig(info *serviceInfo, service *api.Service, port *api.ServicePort) bool {
	if info.protocol != port.Protocol || info.portalPort != port.Port {
		return false
	}
	if !info.portalIP.Equal(net.ParseIP(service.Spec.PortalIP)) {
		return false
	}
	if !info.loadBalancerStatus.Equal(&service.Status.LoadBalancer) {
		return false
	}
	if info.sessionAffinityType != service.Spec.SessionAffinity {
		return false
	}
	return true
}

func ipsEqual(lhs, rhs []string) bool {
	if len(lhs) != len(rhs) {
		return false
	}
	for i := range lhs {
		if lhs[i] != rhs[i] {
			return false
		}
	}
	return true
}

func (proxier *Proxier) openPortal(service ServicePortName, info *serviceInfo) error {
	err := proxier.openOnePortal(info.portalIP, info.portalPort, info.protocol, proxier.listenIP, info.proxyPort, service)
	if err != nil {
		return err
	}
	for _, ingress := range info.loadBalancerStatus.Ingress {
		if ingress.IP != "" {
			err = proxier.openOnePortal(net.ParseIP(ingress.IP), info.portalPort, info.protocol, proxier.listenIP, info.proxyPort, service)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (proxier *Proxier) openOnePortal(portalIP net.IP, portalPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, name ServicePortName) error {
	// Handle traffic from containers.
	args := proxier.iptablesContainerPortalArgs(portalIP, portalPort, protocol, proxyIP, proxyPort, name)
	existed, err := proxier.iptables.EnsureRule(iptables.TableNAT, iptablesContainerPortalChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesContainerPortalChain, name)
		return err
	}
	if !existed {
		glog.V(3).Infof("Opened iptables from-containers portal for service %q on %s %s:%d", name, protocol, portalIP, portalPort)
	}

	// Handle traffic from the host.
	args = proxier.iptablesHostPortalArgs(portalIP, portalPort, protocol, proxyIP, proxyPort, name)
	existed, err = proxier.iptables.EnsureRule(iptables.TableNAT, iptablesHostPortalChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesHostPortalChain, name)
		return err
	}
	if !existed {
		glog.V(3).Infof("Opened iptables from-host portal for service %q on %s %s:%d", name, protocol, portalIP, portalPort)
	}
	return nil
}

func (proxier *Proxier) closePortal(service ServicePortName, info *serviceInfo) error {
	// Collect errors and report them all at the end.
	el := proxier.closeOnePortal(info.portalIP, info.portalPort, info.protocol, proxier.listenIP, info.proxyPort, service)
	for _, ingress := range info.loadBalancerStatus.Ingress {
		if ingress.IP != "" {
			el = append(el, proxier.closeOnePortal(net.ParseIP(ingress.IP), info.portalPort, info.protocol, proxier.listenIP, info.proxyPort, service)...)
		}
	}
	if len(el) == 0 {
		glog.V(3).Infof("Closed iptables portals for service %q", service)
	} else {
		glog.Errorf("Some errors closing iptables portals for service %q", service)
	}
	return errors.NewAggregate(el)
}

func (proxier *Proxier) closeOnePortal(portalIP net.IP, portalPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, name ServicePortName) []error {
	el := []error{}

	// Handle traffic from containers.
	args := proxier.iptablesContainerPortalArgs(portalIP, portalPort, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesContainerPortalChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesContainerPortalChain, name)
		el = append(el, err)
	}

	// Handle traffic from the host.
	args = proxier.iptablesHostPortalArgs(portalIP, portalPort, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesHostPortalChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesHostPortalChain, name)
		el = append(el, err)
	}

	return el
}

// See comments in the *PortalArgs() functions for some details about why we
// use two chains.
var iptablesContainerPortalChain iptables.Chain = "KUBE-PORTALS-CONTAINER"
var iptablesHostPortalChain iptables.Chain = "KUBE-PORTALS-HOST"

// Ensure that the iptables infrastructure we use is set up.  This can safely be called periodically.
func iptablesInit(ipt iptables.Interface) error {
	// TODO: There is almost certainly room for optimization here.  E.g. If
	// we knew the portal_net CIDR we could fast-track outbound packets not
	// destined for a service. There's probably more, help wanted.
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesContainerPortalChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.TableNAT, iptables.ChainPrerouting, "-j", string(iptablesContainerPortalChain)); err != nil {
		return err
	}
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesHostPortalChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.TableNAT, iptables.ChainOutput, "-j", string(iptablesHostPortalChain)); err != nil {
		return err
	}
	return nil
}

// Flush all of our custom iptables rules.
func iptablesFlush(ipt iptables.Interface) error {
	el := []error{}
	if err := ipt.FlushChain(iptables.TableNAT, iptablesContainerPortalChain); err != nil {
		el = append(el, err)
	}
	if err := ipt.FlushChain(iptables.TableNAT, iptablesHostPortalChain); err != nil {
		el = append(el, err)
	}
	if len(el) != 0 {
		glog.Errorf("Some errors flushing old iptables portals: %v", el)
	}
	return errors.NewAggregate(el)
}

// Used below.
var zeroIPv4 = net.ParseIP("0.0.0.0")
var localhostIPv4 = net.ParseIP("127.0.0.1")

var zeroIPv6 = net.ParseIP("::0")
var localhostIPv6 = net.ParseIP("::1")

// Build a slice of iptables args that are common to from-container and from-host portal rules.
func iptablesCommonPortalArgs(destIP net.IP, destPort int, protocol api.Protocol, service ServicePortName) []string {
	// This list needs to include all fields as they are eventually spit out
	// by iptables-save.  This is because some systems do not support the
	// 'iptables -C' arg, and so fall back on parsing iptables-save output.
	// If this does not match, it will not pass the check.  For example:
	// adding the /32 on the destination IP arg is not strictly required,
	// but causes this list to not match the final iptables-save output.
	// This is fragile and I hope one day we can stop supporting such old
	// iptables versions.
	args := []string{
		"-m", "comment",
		"--comment", service.String(),
		"-p", strings.ToLower(string(protocol)),
		"-m", strings.ToLower(string(protocol)),
		"-d", fmt.Sprintf("%s/32", destIP.String()),
		"--dport", fmt.Sprintf("%d", destPort),
	}
	return args
}

// Build a slice of iptables args for a from-container portal rule.
func (proxier *Proxier) iptablesContainerPortalArgs(destIP net.IP, destPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service ServicePortName) []string {
	args := iptablesCommonPortalArgs(destIP, destPort, protocol, service)

	// This is tricky.
	//
	// If the proxy is bound (see Proxier.listenIP) to 0.0.0.0 ("any
	// interface") we want to use REDIRECT, which sends traffic to the
	// "primary address of the incoming interface" which means the container
	// bridge, if there is one.  When the response comes, it comes from that
	// same interface, so the NAT matches and the response packet is
	// correct.  This matters for UDP, since there is no per-connection port
	// number.
	//
	// The alternative would be to use DNAT, except that it doesn't work
	// (empirically):
	//   * DNAT to 127.0.0.1 = Packets just disappear - this seems to be a
	//     well-known limitation of iptables.
	//   * DNAT to eth0's IP = Response packets come from the bridge, which
	//     breaks the NAT, and makes things like DNS not accept them.  If
	//     this could be resolved, it would simplify all of this code.
	//
	// If the proxy is bound to a specific IP, then we have to use DNAT to
	// that IP.  Unlike the previous case, this works because the proxy is
	// ONLY listening on that IP, not the bridge.
	//
	// Why would anyone bind to an address that is not inclusive of
	// localhost?  Apparently some cloud environments have their public IP
	// exposed as a real network interface AND do not have firewalling.  We
	// don't want to expose everything out to the world.
	//
	// Unfortunately, I don't know of any way to listen on some (N > 1)
	// interfaces but not ALL interfaces, short of doing it manually, and
	// this is simpler than that.
	//
	// If the proxy is bound to localhost only, all of this is broken.  Not
	// allowed.
	if proxyIP.Equal(zeroIPv4) || proxyIP.Equal(zeroIPv6) {
		// TODO: Can we REDIRECT with IPv6?
		args = append(args, "-j", "REDIRECT", "--to-ports", fmt.Sprintf("%d", proxyPort))
	} else {
		// TODO: Can we DNAT with IPv6?
		args = append(args, "-j", "DNAT", "--to-destination", net.JoinHostPort(proxyIP.String(), strconv.Itoa(proxyPort)))
	}
	return args
}

// Build a slice of iptables args for a from-host portal rule.
func (proxier *Proxier) iptablesHostPortalArgs(destIP net.IP, destPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service ServicePortName) []string {
	args := iptablesCommonPortalArgs(destIP, destPort, protocol, service)

	// This is tricky.
	//
	// If the proxy is bound (see Proxier.listenIP) to 0.0.0.0 ("any
	// interface") we want to do the same as from-container traffic and use
	// REDIRECT.  Except that it doesn't work (empirically).  REDIRECT on
	// localpackets sends the traffic to localhost (special case, but it is
	// documented) but the response comes from the eth0 IP (not sure why,
	// truthfully), which makes DNS unhappy.
	//
	// So we have to use DNAT.  DNAT to 127.0.0.1 can't work for the same
	// reason.
	//
	// So we do our best to find an interface that is not a loopback and
	// DNAT to that.  This works (again, empirically).
	//
	// If the proxy is bound to a specific IP, then we have to use DNAT to
	// that IP.  Unlike the previous case, this works because the proxy is
	// ONLY listening on that IP, not the bridge.
	//
	// If the proxy is bound to localhost only, this should work, but we
	// don't allow it for now.
	if proxyIP.Equal(zeroIPv4) || proxyIP.Equal(zeroIPv6) {
		proxyIP = proxier.hostIP
	}
	// TODO: Can we DNAT with IPv6?
	args = append(args, "-j", "DNAT", "--to-destination", net.JoinHostPort(proxyIP.String(), strconv.Itoa(proxyPort)))
	return args
}

func isTooManyFDsError(err error) bool {
	return strings.Contains(err.Error(), "too many open files")
}

func isClosedError(err error) bool {
	// A brief discussion about handling closed error here:
	// https://code.google.com/p/go/issues/detail?id=4373#c14
	// TODO: maybe create a stoppable TCP listener that returns a StoppedError
	return strings.HasSuffix(err.Error(), "use of closed network connection")
}
