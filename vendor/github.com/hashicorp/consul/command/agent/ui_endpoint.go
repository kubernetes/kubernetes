package agent

import (
	"net/http"
	"sort"
	"strings"

	"github.com/hashicorp/consul/consul/structs"
)

// ServiceSummary is used to summarize a service
type ServiceSummary struct {
	Name           string
	Nodes          []string
	ChecksPassing  int
	ChecksWarning  int
	ChecksCritical int
}

// UINodes is used to list the nodes in a given datacenter. We return a
// NodeDump which provides overview information for all the nodes
func (s *HTTPServer) UINodes(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Parse arguments
	args := structs.DCSpecificRequest{}
	if done := s.parse(resp, req, &args.Datacenter, &args.QueryOptions); done {
		return nil, nil
	}

	// Make the RPC request
	var out structs.IndexedNodeDump
	defer setMeta(resp, &out.QueryMeta)
RPC:
	if err := s.agent.RPC("Internal.NodeDump", &args, &out); err != nil {
		// Retry the request allowing stale data if no leader
		if strings.Contains(err.Error(), structs.ErrNoLeader.Error()) && !args.AllowStale {
			args.AllowStale = true
			goto RPC
		}
		return nil, err
	}

	// Use empty list instead of nil
	for _, info := range out.Dump {
		if info.Services == nil {
			info.Services = make([]*structs.NodeService, 0)
		}
		if info.Checks == nil {
			info.Checks = make([]*structs.HealthCheck, 0)
		}
	}
	if out.Dump == nil {
		out.Dump = make(structs.NodeDump, 0)
	}
	return out.Dump, nil
}

// UINodeInfo is used to get info on a single node in a given datacenter. We return a
// NodeInfo which provides overview information for the node
func (s *HTTPServer) UINodeInfo(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Parse arguments
	args := structs.NodeSpecificRequest{}
	if done := s.parse(resp, req, &args.Datacenter, &args.QueryOptions); done {
		return nil, nil
	}

	// Verify we have some DC, or use the default
	args.Node = strings.TrimPrefix(req.URL.Path, "/v1/internal/ui/node/")
	if args.Node == "" {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing node name"))
		return nil, nil
	}

	// Make the RPC request
	var out structs.IndexedNodeDump
	defer setMeta(resp, &out.QueryMeta)
RPC:
	if err := s.agent.RPC("Internal.NodeInfo", &args, &out); err != nil {
		// Retry the request allowing stale data if no leader
		if strings.Contains(err.Error(), structs.ErrNoLeader.Error()) && !args.AllowStale {
			args.AllowStale = true
			goto RPC
		}
		return nil, err
	}

	// Return only the first entry
	if len(out.Dump) > 0 {
		info := out.Dump[0]
		if info.Services == nil {
			info.Services = make([]*structs.NodeService, 0)
		}
		if info.Checks == nil {
			info.Checks = make([]*structs.HealthCheck, 0)
		}
		return info, nil
	}
	return nil, nil
}

// UIServices is used to list the services in a given datacenter. We return a
// ServiceSummary which provides overview information for the service
func (s *HTTPServer) UIServices(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Parse arguments
	args := structs.DCSpecificRequest{}
	if done := s.parse(resp, req, &args.Datacenter, &args.QueryOptions); done {
		return nil, nil
	}

	// Make the RPC request
	var out structs.IndexedNodeDump
	defer setMeta(resp, &out.QueryMeta)
RPC:
	if err := s.agent.RPC("Internal.NodeDump", &args, &out); err != nil {
		// Retry the request allowing stale data if no leader
		if strings.Contains(err.Error(), structs.ErrNoLeader.Error()) && !args.AllowStale {
			args.AllowStale = true
			goto RPC
		}
		return nil, err
	}

	// Generate the summary
	return summarizeServices(out.Dump), nil
}

func summarizeServices(dump structs.NodeDump) []*ServiceSummary {
	// Collect the summary information
	var services []string
	summary := make(map[string]*ServiceSummary)
	getService := func(service string) *ServiceSummary {
		serv, ok := summary[service]
		if !ok {
			serv = &ServiceSummary{Name: service}
			summary[service] = serv
			services = append(services, service)
		}
		return serv
	}

	// Aggregate all the node information
	for _, node := range dump {
		nodeServices := make([]*ServiceSummary, len(node.Services))
		for idx, service := range node.Services {
			sum := getService(service.Service)
			sum.Nodes = append(sum.Nodes, node.Node)
			nodeServices[idx] = sum
		}
		for _, check := range node.Checks {
			var services []*ServiceSummary
			if check.ServiceName == "" {
				services = nodeServices
			} else {
				services = []*ServiceSummary{getService(check.ServiceName)}
			}
			for _, sum := range services {
				switch check.Status {
				case structs.HealthPassing:
					sum.ChecksPassing++
				case structs.HealthWarning:
					sum.ChecksWarning++
				case structs.HealthCritical:
					sum.ChecksCritical++
				}
			}
		}
	}

	// Return the services in sorted order
	sort.Strings(services)
	output := make([]*ServiceSummary, len(summary))
	for idx, service := range services {
		// Sort the nodes
		sum := summary[service]
		sort.Strings(sum.Nodes)
		output[idx] = sum
	}
	return output
}
