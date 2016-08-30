package agent

import (
	"fmt"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/serf/coordinate"
	"github.com/hashicorp/serf/serf"
	"net/http"
	"strconv"
	"strings"
)

type AgentSelf struct {
	Config *Config
	Coord  *coordinate.Coordinate
	Member serf.Member
}

func (s *HTTPServer) AgentSelf(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	var c *coordinate.Coordinate
	if !s.agent.config.DisableCoordinates {
		var err error
		if c, err = s.agent.GetCoordinate(); err != nil {
			return nil, err
		}
	}

	return AgentSelf{
		Config: s.agent.config,
		Coord:  c,
		Member: s.agent.LocalMember(),
	}, nil
}

func (s *HTTPServer) AgentServices(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	services := s.agent.state.Services()
	return services, nil
}

func (s *HTTPServer) AgentChecks(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	checks := s.agent.state.Checks()
	return checks, nil
}

func (s *HTTPServer) AgentMembers(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Check if the WAN is being queried
	wan := false
	if other := req.URL.Query().Get("wan"); other != "" {
		wan = true
	}
	if wan {
		return s.agent.WANMembers(), nil
	} else {
		return s.agent.LANMembers(), nil
	}
}

func (s *HTTPServer) AgentJoin(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Check if the WAN is being queried
	wan := false
	if other := req.URL.Query().Get("wan"); other != "" {
		wan = true
	}

	// Get the address
	addr := strings.TrimPrefix(req.URL.Path, "/v1/agent/join/")
	if wan {
		_, err := s.agent.JoinWAN([]string{addr})
		return nil, err
	} else {
		_, err := s.agent.JoinLAN([]string{addr})
		return nil, err
	}
}

func (s *HTTPServer) AgentForceLeave(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	addr := strings.TrimPrefix(req.URL.Path, "/v1/agent/force-leave/")
	return nil, s.agent.ForceLeave(addr)
}

const invalidCheckMessage = "Must provide TTL or Script/DockerContainerID/HTTP/TCP and Interval"

func (s *HTTPServer) AgentRegisterCheck(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	var args CheckDefinition
	// Fixup the type decode of TTL or Interval
	decodeCB := func(raw interface{}) error {
		return FixupCheckType(raw)
	}
	if err := decodeBody(req, &args, decodeCB); err != nil {
		resp.WriteHeader(400)
		resp.Write([]byte(fmt.Sprintf("Request decode failed: %v", err)))
		return nil, nil
	}

	// Verify the check has a name
	if args.Name == "" {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing check name"))
		return nil, nil
	}

	if args.Status != "" && !structs.ValidStatus(args.Status) {
		resp.WriteHeader(400)
		resp.Write([]byte("Bad check status"))
		return nil, nil
	}

	// Construct the health check
	health := args.HealthCheck(s.agent.config.NodeName)

	// Verify the check type
	chkType := &args.CheckType
	if !chkType.Valid() {
		resp.WriteHeader(400)
		resp.Write([]byte(invalidCheckMessage))
		return nil, nil
	}

	// Get the provided token, if any
	var token string
	s.parseToken(req, &token)

	// Add the check
	if err := s.agent.AddCheck(health, chkType, true, token); err != nil {
		return nil, err
	}
	s.syncChanges()
	return nil, nil
}

func (s *HTTPServer) AgentDeregisterCheck(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	checkID := strings.TrimPrefix(req.URL.Path, "/v1/agent/check/deregister/")
	if err := s.agent.RemoveCheck(checkID, true); err != nil {
		return nil, err
	}
	s.syncChanges()
	return nil, nil
}

func (s *HTTPServer) AgentCheckPass(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	checkID := strings.TrimPrefix(req.URL.Path, "/v1/agent/check/pass/")
	note := req.URL.Query().Get("note")
	if err := s.agent.UpdateCheck(checkID, structs.HealthPassing, note); err != nil {
		return nil, err
	}
	s.syncChanges()
	return nil, nil
}

func (s *HTTPServer) AgentCheckWarn(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	checkID := strings.TrimPrefix(req.URL.Path, "/v1/agent/check/warn/")
	note := req.URL.Query().Get("note")
	if err := s.agent.UpdateCheck(checkID, structs.HealthWarning, note); err != nil {
		return nil, err
	}
	s.syncChanges()
	return nil, nil
}

func (s *HTTPServer) AgentCheckFail(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	checkID := strings.TrimPrefix(req.URL.Path, "/v1/agent/check/fail/")
	note := req.URL.Query().Get("note")
	if err := s.agent.UpdateCheck(checkID, structs.HealthCritical, note); err != nil {
		return nil, err
	}
	s.syncChanges()
	return nil, nil
}

// checkUpdate is the payload for a PUT to AgentCheckUpdate.
type checkUpdate struct {
	// Status us one of the structs.Health* states, "passing", "warning", or
	// "critical".
	Status string

	// Output is the information to post to the UI for operators as the
	// output of the process that decided to hit the TTL check. This is
	// different from the note field that's associated with the check
	// itself.
	Output string
}

// AgentCheckUpdate is a PUT-based alternative to the GET-based Pass/Warn/Fail
// APIs.
func (s *HTTPServer) AgentCheckUpdate(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	if req.Method != "PUT" {
		resp.WriteHeader(405)
		return nil, nil
	}

	var update checkUpdate
	if err := decodeBody(req, &update, nil); err != nil {
		resp.WriteHeader(400)
		resp.Write([]byte(fmt.Sprintf("Request decode failed: %v", err)))
		return nil, nil
	}

	switch update.Status {
	case structs.HealthPassing:
	case structs.HealthWarning:
	case structs.HealthCritical:
	default:
		resp.WriteHeader(400)
		resp.Write([]byte(fmt.Sprintf("Invalid check status: '%s'", update.Status)))
		return nil, nil
	}

	total := len(update.Output)
	if total > CheckBufSize {
		update.Output = fmt.Sprintf("%s ... (captured %d of %d bytes)",
			update.Output[:CheckBufSize], CheckBufSize, total)
	}

	checkID := strings.TrimPrefix(req.URL.Path, "/v1/agent/check/update/")
	if err := s.agent.UpdateCheck(checkID, update.Status, update.Output); err != nil {
		return nil, err
	}
	s.syncChanges()
	return nil, nil
}

func (s *HTTPServer) AgentRegisterService(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	var args ServiceDefinition
	// Fixup the type decode of TTL or Interval if a check if provided
	decodeCB := func(raw interface{}) error {
		rawMap, ok := raw.(map[string]interface{})
		if !ok {
			return nil
		}

		for k, v := range rawMap {
			switch strings.ToLower(k) {
			case "check":
				if err := FixupCheckType(v); err != nil {
					return err
				}
			case "checks":
				chkTypes, ok := v.([]interface{})
				if !ok {
					continue
				}
				for _, chkType := range chkTypes {
					if err := FixupCheckType(chkType); err != nil {
						return err
					}
				}
			}
		}
		return nil
	}
	if err := decodeBody(req, &args, decodeCB); err != nil {
		resp.WriteHeader(400)
		resp.Write([]byte(fmt.Sprintf("Request decode failed: %v", err)))
		return nil, nil
	}

	// Verify the service has a name
	if args.Name == "" {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing service name"))
		return nil, nil
	}

	// Get the node service
	ns := args.NodeService()

	// Verify the check type
	chkTypes := args.CheckTypes()
	for _, check := range chkTypes {
		if check.Status != "" && !structs.ValidStatus(check.Status) {
			resp.WriteHeader(400)
			resp.Write([]byte("Status for checks must 'passing', 'warning', 'critical', 'unknown'"))
			return nil, nil
		}
		if !check.Valid() {
			resp.WriteHeader(400)
			resp.Write([]byte(invalidCheckMessage))
			return nil, nil
		}
	}

	// Get the provided token, if any
	var token string
	s.parseToken(req, &token)

	// Add the check
	if err := s.agent.AddService(ns, chkTypes, true, token); err != nil {
		return nil, err
	}
	s.syncChanges()
	return nil, nil
}

func (s *HTTPServer) AgentDeregisterService(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	serviceID := strings.TrimPrefix(req.URL.Path, "/v1/agent/service/deregister/")
	if err := s.agent.RemoveService(serviceID, true); err != nil {
		return nil, err
	}
	s.syncChanges()
	return nil, nil
}

func (s *HTTPServer) AgentServiceMaintenance(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Only PUT supported
	if req.Method != "PUT" {
		resp.WriteHeader(405)
		return nil, nil
	}

	// Ensure we have a service ID
	serviceID := strings.TrimPrefix(req.URL.Path, "/v1/agent/service/maintenance/")
	if serviceID == "" {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing service ID"))
		return nil, nil
	}

	// Ensure we have some action
	params := req.URL.Query()
	if _, ok := params["enable"]; !ok {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing value for enable"))
		return nil, nil
	}

	raw := params.Get("enable")
	enable, err := strconv.ParseBool(raw)
	if err != nil {
		resp.WriteHeader(400)
		resp.Write([]byte(fmt.Sprintf("Invalid value for enable: %q", raw)))
		return nil, nil
	}

	// Get the provided token, if any
	var token string
	s.parseToken(req, &token)

	if enable {
		reason := params.Get("reason")
		if err = s.agent.EnableServiceMaintenance(serviceID, reason, token); err != nil {
			resp.WriteHeader(404)
			resp.Write([]byte(err.Error()))
			return nil, nil
		}
	} else {
		if err = s.agent.DisableServiceMaintenance(serviceID); err != nil {
			resp.WriteHeader(404)
			resp.Write([]byte(err.Error()))
			return nil, nil
		}
	}
	s.syncChanges()
	return nil, nil
}

func (s *HTTPServer) AgentNodeMaintenance(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Only PUT supported
	if req.Method != "PUT" {
		resp.WriteHeader(405)
		return nil, nil
	}

	// Ensure we have some action
	params := req.URL.Query()
	if _, ok := params["enable"]; !ok {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing value for enable"))
		return nil, nil
	}

	raw := params.Get("enable")
	enable, err := strconv.ParseBool(raw)
	if err != nil {
		resp.WriteHeader(400)
		resp.Write([]byte(fmt.Sprintf("Invalid value for enable: %q", raw)))
		return nil, nil
	}

	// Get the provided token, if any
	var token string
	s.parseToken(req, &token)

	if enable {
		s.agent.EnableNodeMaintenance(params.Get("reason"), token)
	} else {
		s.agent.DisableNodeMaintenance()
	}
	s.syncChanges()
	return nil, nil
}

// syncChanges is a helper function which wraps a blocking call to sync
// services and checks to the server. If the operation fails, we only
// only warn because the write did succeed and anti-entropy will sync later.
func (s *HTTPServer) syncChanges() {
	if err := s.agent.state.syncChanges(); err != nil {
		s.logger.Printf("[ERR] agent: failed to sync changes: %v", err)
	}
}
