package agent

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/hashicorp/consul/consul/structs"
)

// aclCreateResponse is used to wrap the ACL ID
type aclCreateResponse struct {
	ID string
}

// aclDisabled handles if ACL datacenter is not configured
func aclDisabled(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	resp.WriteHeader(401)
	resp.Write([]byte("ACL support disabled"))
	return nil, nil
}

func (s *HTTPServer) ACLDestroy(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Mandate a PUT request
	if req.Method != "PUT" {
		resp.WriteHeader(405)
		return nil, nil
	}

	args := structs.ACLRequest{
		Datacenter: s.agent.config.ACLDatacenter,
		Op:         structs.ACLDelete,
	}
	s.parseToken(req, &args.Token)

	// Pull out the acl id
	args.ACL.ID = strings.TrimPrefix(req.URL.Path, "/v1/acl/destroy/")
	if args.ACL.ID == "" {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing ACL"))
		return nil, nil
	}

	var out string
	if err := s.agent.RPC("ACL.Apply", &args, &out); err != nil {
		return nil, err
	}
	return true, nil
}

func (s *HTTPServer) ACLCreate(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	return s.aclSet(resp, req, false)
}

func (s *HTTPServer) ACLUpdate(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	return s.aclSet(resp, req, true)
}

func (s *HTTPServer) aclSet(resp http.ResponseWriter, req *http.Request, update bool) (interface{}, error) {
	// Mandate a PUT request
	if req.Method != "PUT" {
		resp.WriteHeader(405)
		return nil, nil
	}

	args := structs.ACLRequest{
		Datacenter: s.agent.config.ACLDatacenter,
		Op:         structs.ACLSet,
		ACL: structs.ACL{
			Type: structs.ACLTypeClient,
		},
	}
	s.parseToken(req, &args.Token)

	// Handle optional request body
	if req.ContentLength > 0 {
		if err := decodeBody(req, &args.ACL, nil); err != nil {
			resp.WriteHeader(400)
			resp.Write([]byte(fmt.Sprintf("Request decode failed: %v", err)))
			return nil, nil
		}
	}

	// Ensure there is an ID set for update. ID is optional for
	// create, as one will be generated if not provided.
	if update && args.ACL.ID == "" {
		resp.WriteHeader(400)
		resp.Write([]byte(fmt.Sprintf("ACL ID must be set")))
		return nil, nil
	}

	// Create the acl, get the ID
	var out string
	if err := s.agent.RPC("ACL.Apply", &args, &out); err != nil {
		return nil, err
	}

	// Format the response as a JSON object
	return aclCreateResponse{out}, nil
}

func (s *HTTPServer) ACLClone(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Mandate a PUT request
	if req.Method != "PUT" {
		resp.WriteHeader(405)
		return nil, nil
	}

	args := structs.ACLSpecificRequest{
		Datacenter: s.agent.config.ACLDatacenter,
	}
	var dc string
	if done := s.parse(resp, req, &dc, &args.QueryOptions); done {
		return nil, nil
	}

	// Pull out the acl id
	args.ACL = strings.TrimPrefix(req.URL.Path, "/v1/acl/clone/")
	if args.ACL == "" {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing ACL"))
		return nil, nil
	}

	var out structs.IndexedACLs
	defer setMeta(resp, &out.QueryMeta)
	if err := s.agent.RPC("ACL.Get", &args, &out); err != nil {
		return nil, err
	}

	// Bail if the ACL is not found
	if len(out.ACLs) == 0 {
		resp.WriteHeader(404)
		resp.Write([]byte(fmt.Sprintf("Target ACL not found")))
		return nil, nil
	}

	// Create a new ACL
	createArgs := structs.ACLRequest{
		Datacenter: args.Datacenter,
		Op:         structs.ACLSet,
		ACL:        *out.ACLs[0],
	}
	createArgs.ACL.ID = ""
	createArgs.Token = args.Token

	// Create the acl, get the ID
	var outID string
	if err := s.agent.RPC("ACL.Apply", &createArgs, &outID); err != nil {
		return nil, err
	}

	// Format the response as a JSON object
	return aclCreateResponse{outID}, nil
}

func (s *HTTPServer) ACLGet(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	args := structs.ACLSpecificRequest{
		Datacenter: s.agent.config.ACLDatacenter,
	}
	var dc string
	if done := s.parse(resp, req, &dc, &args.QueryOptions); done {
		return nil, nil
	}

	// Pull out the acl id
	args.ACL = strings.TrimPrefix(req.URL.Path, "/v1/acl/info/")
	if args.ACL == "" {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing ACL"))
		return nil, nil
	}

	var out structs.IndexedACLs
	defer setMeta(resp, &out.QueryMeta)
	if err := s.agent.RPC("ACL.Get", &args, &out); err != nil {
		return nil, err
	}

	// Use empty list instead of nil
	if out.ACLs == nil {
		out.ACLs = make(structs.ACLs, 0)
	}
	return out.ACLs, nil
}

func (s *HTTPServer) ACLList(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	args := structs.DCSpecificRequest{
		Datacenter: s.agent.config.ACLDatacenter,
	}
	var dc string
	if done := s.parse(resp, req, &dc, &args.QueryOptions); done {
		return nil, nil
	}

	var out structs.IndexedACLs
	defer setMeta(resp, &out.QueryMeta)
	if err := s.agent.RPC("ACL.List", &args, &out); err != nil {
		return nil, err
	}

	// Use empty list instead of nil
	if out.ACLs == nil {
		out.ACLs = make(structs.ACLs, 0)
	}
	return out.ACLs, nil
}
