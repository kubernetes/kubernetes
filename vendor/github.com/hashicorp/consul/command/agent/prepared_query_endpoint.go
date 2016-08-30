package agent

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/hashicorp/consul/consul"
	"github.com/hashicorp/consul/consul/structs"
)

const (
	preparedQueryEndpoint      = "PreparedQuery"
	preparedQueryExecuteSuffix = "/execute"
	preparedQueryExplainSuffix = "/explain"
)

// preparedQueryCreateResponse is used to wrap the query ID.
type preparedQueryCreateResponse struct {
	ID string
}

// preparedQueryCreate makes a new prepared query.
func (s *HTTPServer) preparedQueryCreate(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	args := structs.PreparedQueryRequest{
		Op: structs.PreparedQueryCreate,
	}
	s.parseDC(req, &args.Datacenter)
	s.parseToken(req, &args.Token)
	if req.ContentLength > 0 {
		if err := decodeBody(req, &args.Query, nil); err != nil {
			resp.WriteHeader(400)
			resp.Write([]byte(fmt.Sprintf("Request decode failed: %v", err)))
			return nil, nil
		}
	}

	var reply string
	endpoint := s.agent.getEndpoint(preparedQueryEndpoint)
	if err := s.agent.RPC(endpoint+".Apply", &args, &reply); err != nil {
		return nil, err
	}
	return preparedQueryCreateResponse{reply}, nil
}

// preparedQueryList returns all the prepared queries.
func (s *HTTPServer) preparedQueryList(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	var args structs.DCSpecificRequest
	if done := s.parse(resp, req, &args.Datacenter, &args.QueryOptions); done {
		return nil, nil
	}

	var reply structs.IndexedPreparedQueries
	endpoint := s.agent.getEndpoint(preparedQueryEndpoint)
	if err := s.agent.RPC(endpoint+".List", &args, &reply); err != nil {
		return nil, err
	}

	// Use empty list instead of nil.
	if reply.Queries == nil {
		reply.Queries = make(structs.PreparedQueries, 0)
	}
	return reply.Queries, nil
}

// PreparedQueryGeneral handles all the general prepared query requests.
func (s *HTTPServer) PreparedQueryGeneral(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	switch req.Method {
	case "POST":
		return s.preparedQueryCreate(resp, req)

	case "GET":
		return s.preparedQueryList(resp, req)

	default:
		resp.WriteHeader(405)
		return nil, nil
	}
}

// parseLimit parses the optional limit parameter for a prepared query execution.
func parseLimit(req *http.Request, limit *int) error {
	*limit = 0
	if arg := req.URL.Query().Get("limit"); arg != "" {
		if i, err := strconv.Atoi(arg); err != nil {
			return err
		} else {
			*limit = i
		}
	}
	return nil
}

// preparedQueryExecute executes a prepared query.
func (s *HTTPServer) preparedQueryExecute(id string, resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	args := structs.PreparedQueryExecuteRequest{
		QueryIDOrName: id,
	}
	s.parseSource(req, &args.Source)
	if done := s.parse(resp, req, &args.Datacenter, &args.QueryOptions); done {
		return nil, nil
	}
	if err := parseLimit(req, &args.Limit); err != nil {
		return nil, fmt.Errorf("Bad limit: %s", err)
	}

	var reply structs.PreparedQueryExecuteResponse
	endpoint := s.agent.getEndpoint(preparedQueryEndpoint)
	if err := s.agent.RPC(endpoint+".Execute", &args, &reply); err != nil {
		// We have to check the string since the RPC sheds
		// the specific error type.
		if err.Error() == consul.ErrQueryNotFound.Error() {
			resp.WriteHeader(404)
			resp.Write([]byte(err.Error()))
			return nil, nil
		}
		return nil, err
	}

	// Use empty list instead of nil.
	if reply.Nodes == nil {
		reply.Nodes = make(structs.CheckServiceNodes, 0)
	}
	return reply, nil
}

// preparedQueryExplain shows which query a name resolves to, the fully
// interpolated template (if it's a template), as well as additional info
// about the execution of a query.
func (s *HTTPServer) preparedQueryExplain(id string, resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	args := structs.PreparedQueryExecuteRequest{
		QueryIDOrName: id,
	}
	s.parseSource(req, &args.Source)
	if done := s.parse(resp, req, &args.Datacenter, &args.QueryOptions); done {
		return nil, nil
	}
	if err := parseLimit(req, &args.Limit); err != nil {
		return nil, fmt.Errorf("Bad limit: %s", err)
	}

	var reply structs.PreparedQueryExplainResponse
	endpoint := s.agent.getEndpoint(preparedQueryEndpoint)
	if err := s.agent.RPC(endpoint+".Explain", &args, &reply); err != nil {
		// We have to check the string since the RPC sheds
		// the specific error type.
		if err.Error() == consul.ErrQueryNotFound.Error() {
			resp.WriteHeader(404)
			resp.Write([]byte(err.Error()))
			return nil, nil
		}
		return nil, err
	}
	return reply, nil
}

// preparedQueryGet returns a single prepared query.
func (s *HTTPServer) preparedQueryGet(id string, resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	args := structs.PreparedQuerySpecificRequest{
		QueryID: id,
	}
	if done := s.parse(resp, req, &args.Datacenter, &args.QueryOptions); done {
		return nil, nil
	}

	var reply structs.IndexedPreparedQueries
	endpoint := s.agent.getEndpoint(preparedQueryEndpoint)
	if err := s.agent.RPC(endpoint+".Get", &args, &reply); err != nil {
		// We have to check the string since the RPC sheds
		// the specific error type.
		if err.Error() == consul.ErrQueryNotFound.Error() {
			resp.WriteHeader(404)
			resp.Write([]byte(err.Error()))
			return nil, nil
		}
		return nil, err
	}
	return reply.Queries, nil
}

// preparedQueryUpdate updates a prepared query.
func (s *HTTPServer) preparedQueryUpdate(id string, resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	args := structs.PreparedQueryRequest{
		Op: structs.PreparedQueryUpdate,
	}
	s.parseDC(req, &args.Datacenter)
	s.parseToken(req, &args.Token)
	if req.ContentLength > 0 {
		if err := decodeBody(req, &args.Query, nil); err != nil {
			resp.WriteHeader(400)
			resp.Write([]byte(fmt.Sprintf("Request decode failed: %v", err)))
			return nil, nil
		}
	}

	// Take the ID from the URL, not the embedded one.
	args.Query.ID = id

	var reply string
	endpoint := s.agent.getEndpoint(preparedQueryEndpoint)
	if err := s.agent.RPC(endpoint+".Apply", &args, &reply); err != nil {
		return nil, err
	}
	return nil, nil
}

// preparedQueryDelete deletes prepared query.
func (s *HTTPServer) preparedQueryDelete(id string, resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	args := structs.PreparedQueryRequest{
		Op: structs.PreparedQueryDelete,
		Query: &structs.PreparedQuery{
			ID: id,
		},
	}
	s.parseDC(req, &args.Datacenter)
	s.parseToken(req, &args.Token)

	var reply string
	endpoint := s.agent.getEndpoint(preparedQueryEndpoint)
	if err := s.agent.RPC(endpoint+".Apply", &args, &reply); err != nil {
		return nil, err
	}
	return nil, nil
}

// PreparedQuerySpecific handles all the prepared query requests specific to a
// particular query.
func (s *HTTPServer) PreparedQuerySpecific(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	id := strings.TrimPrefix(req.URL.Path, "/v1/query/")

	execute, explain := false, false
	if strings.HasSuffix(id, preparedQueryExecuteSuffix) {
		execute = true
		id = strings.TrimSuffix(id, preparedQueryExecuteSuffix)
	} else if strings.HasSuffix(id, preparedQueryExplainSuffix) {
		explain = true
		id = strings.TrimSuffix(id, preparedQueryExplainSuffix)
	}

	switch req.Method {
	case "GET":
		if execute {
			return s.preparedQueryExecute(id, resp, req)
		} else if explain {
			return s.preparedQueryExplain(id, resp, req)
		} else {
			return s.preparedQueryGet(id, resp, req)
		}

	case "PUT":
		return s.preparedQueryUpdate(id, resp, req)

	case "DELETE":
		return s.preparedQueryDelete(id, resp, req)

	default:
		resp.WriteHeader(405)
		return nil, nil
	}
}
