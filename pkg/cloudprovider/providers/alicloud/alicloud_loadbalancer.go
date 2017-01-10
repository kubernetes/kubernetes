/*
Copyright 2014 The Kubernetes Authors.

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

package alicloud

import (
	"errors"
	"fmt"
	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/slb"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"strconv"
	"strings"
)

// ServiceAnnotationLoadBalancerSSLPorts is the annotation used on the service
// to specify a comma-separated list of ports that will use SSL/HTTPS
// listeners. Defaults to '*' (all).
const ServiceAnnotationLoadBalancerProtocolPort = "service.beta.kubernetes.io/alicloud-loadbalancer-ProtocolPort"

const ServiceAnnotationLoadBalancerAddressType = "service.beta.kubernetes.io/alicloud-loadbalancer-AddressType"

const ServiceAnnotationLoadBalancerChargeType = "service.beta.kubernetes.io/alicloud-loadbalancer-ChargeType"

const ServiceAnnotationLoadBalancerRegion = "service.beta.kubernetes.io/alicloud-loadbalancer-Region"

const ServiceAnnotationLoadBalancerBandwidth = "service.beta.kubernetes.io/alicloud-loadbalancer-Bandwidth"

const ServiceAnnotationLoadBalancerCertID = "service.beta.kubernetes.io/alicloud-loadbalancer-CertID"

const ServiceAnnotationLoadBalancerHealthCheckFlag = "service.beta.kubernetes.io/alicloud-loadbalancer-HealthCheckFlag"

const ServiceAnnotationLoadBalancerHealthCheckType = "service.beta.kubernetes.io/alicloud-loadbalancer-HealthCheckType"

const ServiceAnnotationLoadBalancerHealthCheckURI = "service.beta.kubernetes.io/alicloud-loadbalancer-HealthCheckURI"

const ServiceAnnotationLoadBalancerHealthCheckConnectPort = "service.beta.kubernetes.io/alicloud-loadbalancer-HealthCheckConnectPort"

const ServiceAnnotationLoadBalancerHealthCheckHealthyThreshold = "service.beta.kubernetes.io/alicloud-loadbalancer-HealthyThreshold"

const ServiceAnnotationLoadBalancerHealthCheckUnhealthyThreshold = "service.beta.kubernetes.io/alicloud-loadbalancer-UnhealthyThreshold"

const ServiceAnnotationLoadBalancerHealthCheckInterval = "service.beta.kubernetes.io/alicloud-loadbalancer-HealthCheckInterval"

const ServiceAnnotationLoadBalancerHealthCheckConnectTimeout = "service.beta.kubernetes.io/alicloud-loadbalancer-HealthCheckConnectTimeout"

const ServiceAnnotationLoadBalancerHealthCheckTimeout = "service.beta.kubernetes.io/alicloud-loadbalancer-HealthCheckTimeout"

type AnnotationRequest struct {
	SSLPorts    string
	AddressType slb.AddressType
	ChargeType  slb.InternetChargeType
	Region      common.Region
	Bandwidth   int
	CertID      string

	HealthCheck            slb.FlagType
	HealthCheckURI         string
	HealthCheckConnectPort int
	HealthyThreshold       int
	UnhealthyThreshold     int
	HealthCheckInterval    int

	HealthCheckConnectTimeout int                 // for tcp
	HealthCheckType           slb.HealthCheckType // for tcp, Type could be http tcp
	HealthCheckTimeout        int                 // for https and http
}

func ExtractAnnotationRequest(service *v1.Service) *AnnotationRequest {
	ar := &AnnotationRequest{}
	annotation := service.Annotations

	i, err := strconv.Atoi(annotation[ServiceAnnotationLoadBalancerBandwidth])
	if err != nil {
		glog.Errorf("Warining: Annotation bandwidth must be integer,got [%s],use default number 50.",
			annotation[ServiceAnnotationLoadBalancerBandwidth])
		ar.Bandwidth = DEFAULT_BANDWIDTH
	} else {
		ar.Bandwidth = i
	}
	addtype := annotation[ServiceAnnotationLoadBalancerAddressType]
	if addtype != "" {
		ar.AddressType = slb.AddressType(addtype)
	} else {
		ar.AddressType = slb.InternetAddressType
	}

	chargtype := annotation[ServiceAnnotationLoadBalancerChargeType]
	if chargtype != "" {
		ar.ChargeType = slb.InternetChargeType(chargtype)
	} else {
		ar.ChargeType = slb.PayByTraffic
	}

	region := annotation[ServiceAnnotationLoadBalancerRegion]
	if region != "" {
		ar.Region = common.Region(region)
	} else {
		ar.Region = common.Hangzhou
	}

	certid := annotation[ServiceAnnotationLoadBalancerCertID]
	if certid != "" {
		ar.CertID = certid
	}

	hcFlag := annotation[ServiceAnnotationLoadBalancerHealthCheckFlag]
	if hcFlag != "" {
		ar.HealthCheck = slb.FlagType(hcFlag)
	} else {
		ar.HealthCheck = slb.OffFlag
	}

	hcType := annotation[ServiceAnnotationLoadBalancerHealthCheckType]
	if hcType != "" {
		ar.HealthCheckType = slb.HealthCheckType(hcType)
	} else {
		ar.HealthCheckType = slb.TCPHealthCheckType
	}

	hcUri := annotation[ServiceAnnotationLoadBalancerHealthCheckURI]
	if hcUri != "" {
		ar.HealthCheckURI = hcUri
	} else {
		ar.HealthCheckURI = "/"
	}

	port, err := strconv.Atoi(annotation[ServiceAnnotationLoadBalancerHealthCheckConnectPort])
	if err != nil {
		ar.HealthCheckConnectPort = -520
	} else {
		ar.HealthCheckConnectPort = port
	}

	thresh, err := strconv.Atoi(annotation[ServiceAnnotationLoadBalancerHealthCheckHealthyThreshold])
	if err != nil {
		ar.HealthyThreshold = 3
	} else {
		ar.HealthyThreshold = thresh
	}

	unThresh, err := strconv.Atoi(annotation[ServiceAnnotationLoadBalancerHealthCheckUnhealthyThreshold])
	if err != nil {
		ar.UnhealthyThreshold = 3
	} else {
		ar.UnhealthyThreshold = unThresh
	}

	interval, err := strconv.Atoi(annotation[ServiceAnnotationLoadBalancerHealthCheckInterval])
	if err != nil {
		ar.HealthCheckInterval = 2
	} else {
		ar.HealthCheckInterval = interval
	}

	connout, err := strconv.Atoi(annotation[ServiceAnnotationLoadBalancerHealthCheckConnectTimeout])
	if err != nil {
		ar.HealthCheckConnectTimeout = 5
	} else {
		ar.HealthCheckConnectTimeout = connout
	}

	hout, err := strconv.Atoi(annotation[ServiceAnnotationLoadBalancerHealthCheckTimeout])
	if err != nil {
		ar.HealthCheckTimeout = 5
	} else {
		ar.HealthCheckConnectPort = hout
	}
	return ar
}

type SDKClientSLB struct {
	c        *slb.Client
	RegionId common.Region
}

func NewSDKClientSLB(region common.Region, key string, secret string) *SDKClientSLB {
	return &SDKClientSLB{
		RegionId: region,
		c:        slb.NewClient(key, secret),
	}
}
func (s *SDKClientSLB) GetLoadBalancerByName(lbn string) (*slb.LoadBalancerType, bool, error) {
	lbs, err := s.c.DescribeLoadBalancers(
		&slb.DescribeLoadBalancersArgs{
			RegionId:         s.RegionId,
			LoadBalancerName: lbn,
		},
	)

	if err != nil {
		return nil, false, err
	}

	if lbs == nil || len(lbs) == 0 {
		return nil, false, nil
	}
	if len(lbs) > 1 {
		glog.Errorf("Warning: Mutil LoadBalancer returned with name=%s, Using the first one with IP=%s", lbn, lbs[0].Address)
	}
	lb, err := s.c.DescribeLoadBalancerAttribute(lbs[0].LoadBalancerId)
	if err != nil {
		return nil, false, err
	}
	return lb, true, nil
}

func (s *SDKClientSLB) EnsureLoadBalancer(service *v1.Service, nodes []*v1.Node) (*slb.LoadBalancerType, error) {
	lbn := cloudprovider.GetLoadBalancerName(service)
	lb, exists, err := s.GetLoadBalancerByName(lbn)
	if err != nil {
		return nil, err
	}
	opts := s.getLoadBalancerOpts(service)
	opts.LoadBalancerName = lbn
	if !exists {
		lbr, err := s.c.CreateLoadBalancer(opts)
		if err != nil {
			return nil, err
		}
		lb, err = s.c.DescribeLoadBalancerAttribute(lbr.LoadBalancerId)
		if err != nil {
			return nil, err
		}
	} else {

		// Todo: here we need to compare loadbalance
		if opts.InternetChargeType != lb.InternetChargeType ||
			opts.Bandwidth != lb.Bandwidth {
			glog.Infof("Alicloud.EnsureLoadBalancer() InternetChargeType or Bandwidth Changed, update LoadBalancer:[%+v]\n", opts)
			if err := s.c.ModifyLoadBalancerInternetSpec(
				&slb.ModifyLoadBalancerInternetSpecArgs{
					LoadBalancerId:     lb.LoadBalancerId,
					InternetChargeType: opts.InternetChargeType,
					Bandwidth:          opts.Bandwidth,
				}); err != nil {
				return nil, err
			}
		}
		if opts.AddressType != lb.AddressType {
			//fmt.Printf("Alicloud.EnsureLoadBalance(%s): AddressType changed[%s => %s] ,recreate loadbalance!",
			//	lb.AddressType,opts.AddressType,opts.LoadBalancerName,)
			glog.Infof("Alicloud.EnsureLoadBalance(%s): AddressType changed[%s => %s] ,recreate loadbalance!",
				lb.AddressType, opts.AddressType, opts.LoadBalancerName)
			// Can not modify AddressType.  We can only recreate it.
			if err := s.c.DeleteLoadBalancer(lb.LoadBalancerId); err != nil {
				return nil, err
			}
			lbc, err := s.c.CreateLoadBalancer(opts)
			if err != nil {
				return nil, err
			}
			lb, err = s.c.DescribeLoadBalancerAttribute(lbc.LoadBalancerId)
			if err != nil {
				return nil, err
			}
		}
	}
	fmt.Printf("Alicloud.EnsureLoadBalancer() create LoadBalancer step 1:[%+v]\n", lb)
	glog.Infof("Alicloud.EnsureLoadBalancer() create LoadBalancer step 1:[%+v]\n", lb)

	if _, err := s.EnsureLoadBalancerListener(service, lb); err != nil {

		return nil, err
	}

	return s.EnsureBackendServer(service, nodes, lb)
}

func (s *SDKClientSLB) UpdateLoadBalancer(service *v1.Service, nodes []*v1.Node) error {
	lbn := cloudprovider.GetLoadBalancerName(service)
	lb, exists, err := s.GetLoadBalancerByName(lbn)
	if err != nil {
		return err
	}
	if !exists {
		return errors.New(fmt.Sprintf("The loadbalance you specified by name [%s] does not exist!", lbn))
	}
	_, err = s.EnsureBackendServer(service, nodes, lb)
	return err
}

func (s *SDKClientSLB) EnsureLoadBalancerListener(service *v1.Service, lb *slb.LoadBalancerType) (*slb.LoadBalancerType, error) {
	//ssl := service.Annotations["sec_ports"]
	additions, deletions, err := s.diffListeners(service, lb)
	if err != nil {
		return nil, err
	}
	glog.Infof("Alicloud.EnsureLoadBalancerListener() Add additional LoadBalancerListerners:[%+v],  Delete removed LoadBalancerListerners[%+v]", additions, deletions)
	if len(deletions) > 0 {
		for _, p := range deletions {
			// stop first
			// todo: here should retry for none runing status
			if err := s.c.StopLoadBalancerListener(lb.LoadBalancerId, p.Port); err != nil {
				return nil, err
			}
			// deal with port delete
			if err := s.c.DeleteLoadBalancerListener(lb.LoadBalancerId, p.Port); err != nil {
				return nil, err
			}
		}
	}
	if len(additions) > 0 {
		// deal with port add
		for _, p := range additions {
			if err := s.createListener(lb, p); err != nil {
				return nil, err
			}
			// todo : here should retry
			if err := s.c.StartLoadBalancerListener(lb.LoadBalancerId, p.Port); err != nil {
				return nil, err
			}
		}
	}
	return lb, nil
}

type PortListener struct {
	Port     int
	NodePort int
	Protocol string

	Bandwidth int

	Scheduler     slb.SchedulerType
	StickySession slb.FlagType
	CertID        string

	HealthCheck            slb.FlagType
	HealthCheckType        slb.HealthCheckType
	HealthCheckURI         string
	HealthCheckConnectPort int

	HealthyThreshold    int
	UnhealthyThreshold  int
	HealthCheckInterval int

	HealthCheckConnectTimeout int // for tcp
	HealthCheckTimeout        int // for https and http
}

// 1. Modify ListenPort would cause listener to be recreated
// 2. Modify NodePort would cause listener to be recreated
// 3. Modify Protocol would cause listener to be recreated
//
func (s *SDKClientSLB) diffListeners(service *v1.Service, lb *slb.LoadBalancerType) (
	[]PortListener, []PortListener, error) {
	lp := lb.ListenerPortsAndProtocol.ListenerPortAndProtocol
	additions, deletions := []PortListener{}, []PortListener{}

	ar := ExtractAnnotationRequest(service)
	stickSession := slb.OffFlag
	// find additions
	for _, v1 := range service.Spec.Ports {
		found := false
		proto, err := transProtocol(service.Annotations[ServiceAnnotationLoadBalancerProtocolPort], &v1)
		if err != nil {
			return nil, nil, err
		}
		new := PortListener{
			Port:                   int(v1.Port),
			Protocol:               proto,
			NodePort:               int(v1.NodePort),
			Bandwidth:              ar.Bandwidth,
			HealthCheck:            ar.HealthCheck,
			StickySession:          stickSession,
			CertID:                 ar.CertID,
			HealthCheckType:        ar.HealthCheckType,
			HealthCheckConnectPort: ar.HealthCheckConnectPort,
			HealthCheckURI:         ar.HealthCheckURI,

			HealthyThreshold:          ar.HealthyThreshold,
			UnhealthyThreshold:        ar.UnhealthyThreshold,
			HealthCheckInterval:       ar.HealthCheckInterval,
			HealthCheckConnectTimeout: ar.HealthCheckConnectTimeout,
			HealthCheckTimeout:        ar.HealthCheckTimeout,
		}
		for _, v2 := range lp {
			if int64(v1.Port) == int64(v2.ListenerPort) {
				old, err := s.findPortListener(lb, v2.ListenerPort, v2.ListenerProtocol)
				if err != nil {
					return nil, nil, err
				}
				update := false
				if proto != v2.ListenerProtocol {
					update = true
					glog.Infof("Alicloud.diffListeners(%s): protocol changed [ %s => %s]", lb.LoadBalancerId, v2.ListenerProtocol, proto)
				}
				if int(v1.NodePort) != old.NodePort {
					update = true
					glog.Infof("Alicloud.diffListeners(%s): NodePort changed [ %d => %d]", lb.LoadBalancerId, old.NodePort, v1.NodePort)
				}

				if old.Bandwidth != ar.Bandwidth {
					update = true
					glog.Infof("Alicloud.diffListeners(%s): bandwidth changed [ %d => %d]", lb.LoadBalancerId, old.Bandwidth, ar.Bandwidth)
				}
				if old.CertID != ar.CertID && proto == "https" {
					update = true
					glog.Infof("Alicloud.diffListeners(%s): CertID changed [ %s => %s]", lb.LoadBalancerId, old.CertID, ar.CertID)
				}
				if old.HealthCheck != ar.HealthCheck ||
					old.HealthCheckType != ar.HealthCheckType ||
					old.HealthCheckURI != ar.HealthCheckURI ||
					old.HealthCheckConnectPort != ar.HealthCheckConnectPort {
					update = true
					glog.Infof("Alicloud.diffListeners(%s): HealthCheck changed ", lb.LoadBalancerId)
				}
				if update {
					deletions = append(deletions, old)
					additions = append(additions, new)
				}
				found = true
			}
		}
		if !found {
			additions = append(additions, new)
		}
	}

	// Find deletions
	for _, v1 := range lp {
		found := false
		for _, v2 := range service.Spec.Ports {
			if int64(v1.ListenerPort) == int64(v2.Port) {
				found = true
			}
		}
		if !found {
			deletions = append(deletions, PortListener{Port: v1.ListenerPort})
		}
	}

	return additions, deletions, nil
}

func (s *SDKClientSLB) findPortListener(lb *slb.LoadBalancerType, port int, proto string) (PortListener, error) {
	switch proto {
	case "http":
		p, err := s.c.DescribeLoadBalancerHTTPListenerAttribute(lb.LoadBalancerId, port)
		if err != nil {
			return PortListener{}, err
		}
		return PortListener{
			Port:                   p.ListenerPort,
			NodePort:               p.BackendServerPort,
			Protocol:               proto,
			Bandwidth:              p.Bandwidth,
			HealthCheck:            p.HealthCheck,
			Scheduler:              p.Scheduler,
			StickySession:          p.StickySession,
			HealthCheckURI:         p.HealthCheckURI,
			HealthCheckConnectPort: p.HealthCheckConnectPort,

			HealthyThreshold:    p.HealthyThreshold,
			UnhealthyThreshold:  p.UnhealthyThreshold,
			HealthCheckInterval: p.HealthCheckInterval,
			HealthCheckTimeout:  p.HealthCheckTimeout,
		}, nil
	case "tcp":
		p, err := s.c.DescribeLoadBalancerTCPListenerAttribute(lb.LoadBalancerId, port)
		if err != nil {
			return PortListener{}, err
		}
		return PortListener{
			Port:      p.ListenerPort,
			NodePort:  p.BackendServerPort,
			Protocol:  proto,
			Bandwidth: p.Bandwidth,
			Scheduler: p.Scheduler,

			HealthyThreshold:          p.HealthyThreshold,
			UnhealthyThreshold:        p.UnhealthyThreshold,
			HealthCheckInterval:       p.HealthCheckInterval,
			HealthCheckConnectTimeout: p.HealthCheckTimeout,
			HealthCheckTimeout:        p.HealthCheckTimeout,
		}, nil
	case "https":
		p, err := s.c.DescribeLoadBalancerHTTPSListenerAttribute(lb.LoadBalancerId, port)
		if err != nil {
			return PortListener{}, err
		}
		return PortListener{
			Port:          p.ListenerPort,
			NodePort:      p.BackendServerPort,
			Protocol:      proto,
			Bandwidth:     p.Bandwidth,
			HealthCheck:   p.HealthCheck,
			Scheduler:     p.Scheduler,
			StickySession: p.StickySession,
			CertID:        p.ServerCertificateId,

			HealthCheckURI:         p.HealthCheckURI,
			HealthCheckConnectPort: p.HealthCheckConnectPort,

			HealthyThreshold:    p.HealthyThreshold,
			UnhealthyThreshold:  p.UnhealthyThreshold,
			HealthCheckInterval: p.HealthCheckInterval,
			HealthCheckTimeout:  p.HealthCheckTimeout,
		}, nil
	case "udp":
		p, err := s.c.DescribeLoadBalancerUDPListenerAttribute(lb.LoadBalancerId, port)
		if err != nil {
			return PortListener{}, err
		}
		return PortListener{
			Port:      p.ListenerPort,
			NodePort:  p.BackendServerPort,
			Protocol:  proto,
			Bandwidth: p.Bandwidth,
			Scheduler: p.Scheduler,

			HealthCheckConnectPort: p.HealthCheckConnectPort,

			HealthyThreshold:    p.HealthyThreshold,
			UnhealthyThreshold:  p.UnhealthyThreshold,
			HealthCheckInterval: p.HealthCheckInterval,
			HealthCheckTimeout:  p.HealthCheckTimeout,
		}, nil
	}
	return PortListener{}, errors.New(fmt.Sprintf("protocol not match: %s", proto))
}

func (s *SDKClientSLB) EnsureBackendServer(service *v1.Service, nodes []*v1.Node, lb *slb.LoadBalancerType) (*slb.LoadBalancerType, error) {
	additions, deletions := s.diffServers(nodes, lb)
	glog.Infof("Alicloud.EnsureBackendServer() Add additional BackendServers:[%+v],  Delete removed BackendServers[%+v]", additions, deletions)
	if len(additions) > 0 {
		// deal with server add
		if _, err := s.c.AddBackendServers(lb.LoadBalancerId, additions); err != nil {

			return lb, err
		}
	}
	if len(deletions) > 0 {
		servers := []string{}
		for _, v := range deletions {
			servers = append(servers, v.ServerId)
		}
		if _, err := s.c.RemoveBackendServers(lb.LoadBalancerId, servers); err != nil {
			return lb, err
		}
	}
	return lb, nil
}

func (s *SDKClientSLB) EnsureLoadBalanceDeleted(service *v1.Service) error {

	lb, exists, err := s.GetLoadBalancerByName(cloudprovider.GetLoadBalancerName(service))
	if err != nil {
		return err
	}
	if err != nil {
		return err
	}
	if !exists {
		return nil
	}
	return s.c.DeleteLoadBalancer(lb.LoadBalancerId)
}

func (s *SDKClientSLB) EnsureHealthCheck(service *v1.Service, old *PortListener, new *PortListener) (*slb.LoadBalancerType, error) {

	return nil, nil
}

func (s *SDKClientSLB) createListener(lb *slb.LoadBalancerType, pp PortListener) error {
	protocol := pp.Protocol
	if protocol == "https" {
		lis := slb.CreateLoadBalancerHTTPSListenerArgs(
			slb.HTTPSListenerType{
				HTTPListenerType: slb.HTTPListenerType{
					LoadBalancerId:    lb.LoadBalancerId,
					ListenerPort:      pp.Port,
					BackendServerPort: pp.NodePort,
					//Health Check
					HealthCheck:   pp.HealthCheck,
					Bandwidth:     pp.Bandwidth,
					StickySession: pp.StickySession,

					HealthCheckURI:         pp.HealthCheckURI,
					HealthCheckConnectPort: pp.HealthCheckConnectPort,
					HealthyThreshold:       pp.HealthyThreshold,
					UnhealthyThreshold:     pp.UnhealthyThreshold,
					HealthCheckTimeout:     pp.HealthCheckTimeout,
					HealthCheckInterval:    pp.HealthCheckInterval,
				},
				ServerCertificateId: pp.CertID,
			},
		)
		if err := s.c.CreateLoadBalancerHTTPSListener(&lis); err != nil {
			return err
		}
	}
	if protocol == "http" {
		lis := slb.CreateLoadBalancerHTTPListenerArgs(
			slb.HTTPListenerType{
				LoadBalancerId:    lb.LoadBalancerId,
				ListenerPort:      pp.Port,
				BackendServerPort: pp.NodePort,
				//Health Check
				HealthCheck: pp.HealthCheck,

				Bandwidth:     pp.Bandwidth,
				StickySession: pp.StickySession,

				HealthCheckURI:         pp.HealthCheckURI,
				HealthCheckConnectPort: pp.HealthCheckConnectPort,
				HealthyThreshold:       pp.HealthyThreshold,
				UnhealthyThreshold:     pp.UnhealthyThreshold,
				HealthCheckTimeout:     pp.HealthCheckTimeout,
				HealthCheckInterval:    pp.HealthCheckInterval,
			})
		if err := s.c.CreateLoadBalancerHTTPListener(&lis); err != nil {

			return err
		}
	}
	if protocol == strings.ToLower(string(api.ProtocolTCP)) {

		if pp.HealthCheckConnectPort == -520{
			pp.HealthCheckConnectPort = 0
		}
		if err := s.c.CreateLoadBalancerTCPListener(
			&slb.CreateLoadBalancerTCPListenerArgs{
				LoadBalancerId:    lb.LoadBalancerId,
				ListenerPort:      pp.Port,
				BackendServerPort: pp.NodePort,
				//Health Check
				Bandwidth: pp.Bandwidth,

				HealthCheckType:        pp.HealthCheckType,
				HealthCheckURI:         pp.HealthCheckURI,
				HealthCheckConnectPort: pp.HealthCheckConnectPort,
				HealthyThreshold:       pp.HealthyThreshold,
				UnhealthyThreshold:     pp.UnhealthyThreshold,
				HealthCheckTimeout:     pp.HealthCheckConnectTimeout,
				HealthCheckInterval:    pp.HealthCheckInterval,
			}); err != nil {
			return err
		}
	}
	if protocol == strings.ToLower(string(api.ProtocolUDP)) {
		if err := s.c.CreateLoadBalancerUDPListener(
			&slb.CreateLoadBalancerUDPListenerArgs{
				LoadBalancerId:    lb.LoadBalancerId,
				ListenerPort:      pp.Port,
				BackendServerPort: pp.NodePort,
				//Health Check
				Bandwidth: pp.Bandwidth,

				HealthCheckConnectPort: pp.HealthCheckConnectPort,
				HealthyThreshold:       pp.HealthyThreshold,
				UnhealthyThreshold:     pp.UnhealthyThreshold,
				HealthCheckTimeout:     pp.HealthCheckTimeout,
				HealthCheckInterval:    pp.HealthCheckInterval,
			}); err != nil {
			return err
		}
	}

	return nil
}

func (s *SDKClientSLB) getLoadBalancerOpts(service *v1.Service) *slb.CreateLoadBalancerArgs {

	ar := ExtractAnnotationRequest(service)
	return &slb.CreateLoadBalancerArgs{
		AddressType:        ar.AddressType,
		InternetChargeType: ar.ChargeType,
		Bandwidth:          ar.Bandwidth,
		RegionId:           ar.Region,
	}
}

const DEFAULT_SERVER_WEIGHT = 100

func (s *SDKClientSLB) diffServers(nodes []*v1.Node, lb *slb.LoadBalancerType) ([]slb.BackendServerType, []slb.BackendServerType) {
	additions, deletions := []slb.BackendServerType{}, []slb.BackendServerType{}
	for _, n1 := range nodes {
		found := false
		for _, n2 := range lb.BackendServers.BackendServer {
			if n1.Spec.ExternalID == n2.ServerId {
				found = true
				break
			}
		}
		if !found {
			additions = append(additions, slb.BackendServerType{ServerId: n1.Spec.ExternalID, Weight: DEFAULT_SERVER_WEIGHT})
		}
	}
	for _, n1 := range lb.BackendServers.BackendServer {
		found := false
		for _, n2 := range nodes {
			if n1.ServerId == n2.Spec.ExternalID {
				found = true
				break
			}
		}
		if !found {
			deletions = append(deletions, n1)
		}
	}
	return additions, deletions
}

func transProtocol(annotation string, port *v1.ServicePort) (string, error) {
	if annotation != "" {
		for _, v := range strings.Split(annotation, ",") {
			pp := strings.Split(v, ":")
			if len(pp) < 2 {
				return "", errors.New(fmt.Sprintf("Port Protocol format must be like 'https:443' colon separated. pp=[%+v]", pp))
			}

			if pp[0] != "http" &&
				pp[0] != "tcp" &&
				pp[0] != "https" &&
				pp[0] != "udp" {
				return "", errors.New(fmt.Sprintf("Port Protocol format must be either [http|https|tcp|udp], protocol not supported[%s]\n", pp[0]))
			}

			if pp[1] == fmt.Sprintf("%d", port.Port) {
				return pp[0], nil
			}
		}
		return strings.ToLower(string(port.Protocol)), nil
	}

	return strings.ToLower(string(port.Protocol)), nil
}
