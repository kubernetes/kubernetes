package dns

import (
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/golang/glog"

	skymsg "github.com/skynetservices/skydns/msg"
	skyserver "github.com/skynetservices/skydns/server"
)

// TODO: return a real TTL

const domainSuffix = ".cluster.local."
const srvPathFormat = "%s.%s.svc" + domainSuffix
const srvPortPathFormat = "_%s._%s." + srvPathFormat
const headlessSrvPathFormat = "%s." + srvPathFormat
const headlessSrvPortPathFormat = "%s." + srvPortPathFormat

type DNSHandler struct {
	accessor DNSAccessor
}

func (handler *DNSHandler) Records(name string, exact bool) ([]skymsg.Service, error) {
	prefix := strings.Trim(strings.TrimSuffix(name, domainSuffix), ".")
	segments := strings.Split(prefix, ".")

	segLen := len(segments)
	if segLen == 0 {
		return nil, nil
	}

	// Note: etcd doesn't return results for entries starting with an _.
	// This means that we can skip listing certain queries

	// TODO: support the following wildcard types:
	// - [x] *.cluster.local, ns.*.cluster.local, etc --> proceed as normal to other options
	// - [x] *.svc.cluster.local --> svc.cluster.local
	// - [x] *.ns.svc.cluster.local --> ns.svc.cluster.local
	// - [x] *.name.ns.svc.cluster.local --> name.ns.svc.cluster.local
	// - [x] *.*.svc.cluster.local --> svc.cluster.local
	// - [ ] podid.*.ns.svc.cluster.local --> ??
	// - [ ] podid.name.*.svc.cluster.local --> ??
	// - [ ] podid.*.*.svc.cluster.local --> ??
	// - [ ] name.*.svc.cluster.local --> ??
	// - [ ] _port._proto.*.ns.svc.cluster.local --> ??
	// - [ ] _port._proto.name.*.svc.cluster.local
	// - [ ] _port._proto.*.*.svc.cluster.local

	// Remove any leading wildcard segments, since those can be effectively ignored
	for segLen > 0 && (segments[0] == "*" || segments[0] == "any") {
		segments = segments[1:segLen]
		segLen--
	}

	// we may have removed the leading * in *.cluster.local, so check for that
	// [anything].[wildcard].cluster.local maps to [anything].svc.cluster.local
	if segLen == 0 || segments[segLen-1] == "svc" || segments[segLen-1] == "*" || segments[segLen-1] == "any" {
		if len(segments) > 6 {
			return nil, nil
		}

		if exact && len(segments) < 3 {
			// for exact we need either name.ns.svc.cluster.local,
			// hash.name.ns.svc.cluster.local, or _port._proto.name.ns.cluster.local
			return nil, nil
		}

		var entries []skymsg.Service = nil
		var err error
		switch len(segments) {
		case 1:
			// return all services
			entries, err = handler.accessor.GetAll()
		case 2:
			// return all services in Namespace
			entries, err = handler.accessor.GetByNamespace(segments[0])
		case 3:
			// return specified service
			entries, err = handler.accessor.GetByService(segments[0], segments[1])
		case 4:
			// Don't return anything for _proto.name.ns.svc.cluster.local, but do return for
			// podhash.name.ns.svc.cluster.local
			if !strings.HasPrefix(segments[0], "_") {
				entries, err = handler.accessor.GetByPath(skymsg.Path(name))
			}
		case 5:
			// return specific port-proto entry
			entries, err = handler.accessor.GetByPath(skymsg.Path(name))
		}

		if err != nil {
			return nil, err
		}

		if entries == nil || len(entries) == 0 {
			return nil, fmt.Errorf("no records for '%s'", name)
		}

		return entries, nil
	}

	// ignore the legacy case for the moment
	return nil, fmt.Errorf("no record(s) for '%s'", name)
}

func (handler *DNSHandler) ReverseRecord(name string) (*skymsg.Service, error) {
	return handler.accessor.GetByIP(name)
}

func ServeDNS(client *client.Client) error {
	config := &skyserver.Config{
		Domain: domainSuffix,
		Local:  "dns.default.svc" + domainSuffix,
	}
	err := skyserver.SetDefaults(config)
	config.DnsAddr = ""
	config.NoRec = true
	if err != nil {
		glog.Fatalf("could not start DNS: %v", err)
	}

	stopChan := make(chan struct{})
	accessor := NewDNSAccessor(client, stopChan)
	defer close(stopChan)
	handler := &DNSHandler{accessor}

	dnsServer := skyserver.New(handler, config)
	skyserver.Metrics()
	return dnsServer.Run()
}
