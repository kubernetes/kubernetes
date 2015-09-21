package etcd

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"
)

const (
	rawResponse = iota
	normalResponse
)

type responseType int

type RawResponse struct {
	StatusCode int
	Body       []byte
	Header     http.Header
}

var (
	validHttpStatusCode = map[int]bool{
		http.StatusCreated:            true,
		http.StatusOK:                 true,
		http.StatusBadRequest:         true,
		http.StatusNotFound:           true,
		http.StatusPreconditionFailed: true,
		http.StatusForbidden:          true,
	}
)

// Unmarshal parses RawResponse and stores the result in Response
func (rr *RawResponse) Unmarshal() (*Response, error) {
	if rr.StatusCode != http.StatusOK && rr.StatusCode != http.StatusCreated {
		return nil, handleError(rr.Body)
	}

	resp := new(Response)

	err := json.Unmarshal(rr.Body, resp)

	if err != nil {
		return nil, err
	}

	// attach index and term to response
	resp.EtcdIndex, _ = strconv.ParseUint(rr.Header.Get("X-Etcd-Index"), 10, 64)
	resp.RaftIndex, _ = strconv.ParseUint(rr.Header.Get("X-Raft-Index"), 10, 64)
	resp.RaftTerm, _ = strconv.ParseUint(rr.Header.Get("X-Raft-Term"), 10, 64)

	return resp, nil
}

type Response struct {
	Action    string `json:"action"`
	Node      *Node  `json:"node"`
	PrevNode  *Node  `json:"prevNode,omitempty"`
	EtcdIndex uint64 `json:"etcdIndex"`
	RaftIndex uint64 `json:"raftIndex"`
	RaftTerm  uint64 `json:"raftTerm"`
}

type Node struct {
	Key           string     `json:"key, omitempty"`
	Value         string     `json:"value,omitempty"`
	Dir           bool       `json:"dir,omitempty"`
	Expiration    *time.Time `json:"expiration,omitempty"`
	TTL           int64      `json:"ttl,omitempty"`
	Nodes         Nodes      `json:"nodes,omitempty"`
	ModifiedIndex uint64     `json:"modifiedIndex,omitempty"`
	CreatedIndex  uint64     `json:"createdIndex,omitempty"`
}

type Nodes []*Node

// interfaces for sorting
func (ns Nodes) Len() int {
	return len(ns)
}

func (ns Nodes) Less(i, j int) bool {
	return ns[i].Key < ns[j].Key
}

func (ns Nodes) Swap(i, j int) {
	ns[i], ns[j] = ns[j], ns[i]
}
