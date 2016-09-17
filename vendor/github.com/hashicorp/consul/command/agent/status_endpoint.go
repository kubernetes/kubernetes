package agent

import (
	"net/http"
)

func (s *HTTPServer) StatusLeader(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	var out string
	if err := s.agent.RPC("Status.Leader", struct{}{}, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *HTTPServer) StatusPeers(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	var out []string
	if err := s.agent.RPC("Status.Peers", struct{}{}, &out); err != nil {
		return nil, err
	}
	return out, nil
}
