package agent

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/hashicorp/consul/consul/structs"
)

const (
	// maxKVSize is used to limit the maximum payload length
	// of a KV entry. If it exceeds this amount, the client is
	// likely abusing the KV store.
	maxKVSize = 512 * 1024
)

func (s *HTTPServer) KVSEndpoint(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Set default DC
	args := structs.KeyRequest{}
	if done := s.parse(resp, req, &args.Datacenter, &args.QueryOptions); done {
		return nil, nil
	}

	// Pull out the key name, validation left to each sub-handler
	args.Key = strings.TrimPrefix(req.URL.Path, "/v1/kv/")

	// Check for a key list
	keyList := false
	params := req.URL.Query()
	if _, ok := params["keys"]; ok {
		keyList = true
	}

	// Switch on the method
	switch req.Method {
	case "GET":
		if keyList {
			return s.KVSGetKeys(resp, req, &args)
		} else {
			return s.KVSGet(resp, req, &args)
		}
	case "PUT":
		return s.KVSPut(resp, req, &args)
	case "DELETE":
		return s.KVSDelete(resp, req, &args)
	default:
		resp.WriteHeader(405)
		return nil, nil
	}
}

// KVSGet handles a GET request
func (s *HTTPServer) KVSGet(resp http.ResponseWriter, req *http.Request, args *structs.KeyRequest) (interface{}, error) {
	// Check for recurse
	method := "KVS.Get"
	params := req.URL.Query()
	if _, ok := params["recurse"]; ok {
		method = "KVS.List"
	} else if missingKey(resp, args) {
		return nil, nil
	}

	// Make the RPC
	var out structs.IndexedDirEntries
	if err := s.agent.RPC(method, &args, &out); err != nil {
		return nil, err
	}
	setMeta(resp, &out.QueryMeta)

	// Check if we get a not found
	if len(out.Entries) == 0 {
		resp.WriteHeader(404)
		return nil, nil
	}

	// Check if we are in raw mode with a normal get, write out
	// the raw body
	if _, ok := params["raw"]; ok && method == "KVS.Get" {
		body := out.Entries[0].Value
		resp.Header().Set("Content-Length", strconv.FormatInt(int64(len(body)), 10))
		resp.Write(body)
		return nil, nil
	}

	return out.Entries, nil
}

// KVSGetKeys handles a GET request for keys
func (s *HTTPServer) KVSGetKeys(resp http.ResponseWriter, req *http.Request, args *structs.KeyRequest) (interface{}, error) {
	// Check for a separator, due to historic spelling error,
	// we now are forced to check for both spellings
	var sep string
	params := req.URL.Query()
	if _, ok := params["seperator"]; ok {
		sep = params.Get("seperator")
	}
	if _, ok := params["separator"]; ok {
		sep = params.Get("separator")
	}

	// Construct the args
	listArgs := structs.KeyListRequest{
		Datacenter:   args.Datacenter,
		Prefix:       args.Key,
		Seperator:    sep,
		QueryOptions: args.QueryOptions,
	}

	// Make the RPC
	var out structs.IndexedKeyList
	if err := s.agent.RPC("KVS.ListKeys", &listArgs, &out); err != nil {
		return nil, err
	}
	setMeta(resp, &out.QueryMeta)

	// Check if we get a not found. We do not generate
	// not found for the root, but just provide the empty list
	if len(out.Keys) == 0 && listArgs.Prefix != "" {
		resp.WriteHeader(404)
		return nil, nil
	}

	// Use empty list instead of null
	if out.Keys == nil {
		out.Keys = []string{}
	}
	return out.Keys, nil
}

// KVSPut handles a PUT request
func (s *HTTPServer) KVSPut(resp http.ResponseWriter, req *http.Request, args *structs.KeyRequest) (interface{}, error) {
	if missingKey(resp, args) {
		return nil, nil
	}
	if conflictingFlags(resp, req, "cas", "acquire", "release") {
		return nil, nil
	}
	applyReq := structs.KVSRequest{
		Datacenter: args.Datacenter,
		Op:         structs.KVSSet,
		DirEnt: structs.DirEntry{
			Key:   args.Key,
			Flags: 0,
			Value: nil,
		},
	}
	applyReq.Token = args.Token

	// Check for flags
	params := req.URL.Query()
	if _, ok := params["flags"]; ok {
		flagVal, err := strconv.ParseUint(params.Get("flags"), 10, 64)
		if err != nil {
			return nil, err
		}
		applyReq.DirEnt.Flags = flagVal
	}

	// Check for cas value
	if _, ok := params["cas"]; ok {
		casVal, err := strconv.ParseUint(params.Get("cas"), 10, 64)
		if err != nil {
			return nil, err
		}
		applyReq.DirEnt.ModifyIndex = casVal
		applyReq.Op = structs.KVSCAS
	}

	// Check for lock acquisition
	if _, ok := params["acquire"]; ok {
		applyReq.DirEnt.Session = params.Get("acquire")
		applyReq.Op = structs.KVSLock
	}

	// Check for lock release
	if _, ok := params["release"]; ok {
		applyReq.DirEnt.Session = params.Get("release")
		applyReq.Op = structs.KVSUnlock
	}

	// Check the content-length
	if req.ContentLength > maxKVSize {
		resp.WriteHeader(413)
		resp.Write([]byte(fmt.Sprintf("Value exceeds %d byte limit", maxKVSize)))
		return nil, nil
	}

	// Copy the value
	buf := bytes.NewBuffer(nil)
	if _, err := io.Copy(buf, req.Body); err != nil {
		return nil, err
	}
	applyReq.DirEnt.Value = buf.Bytes()

	// Make the RPC
	var out bool
	if err := s.agent.RPC("KVS.Apply", &applyReq, &out); err != nil {
		return nil, err
	}

	// Only use the out value if this was a CAS
	if applyReq.Op == structs.KVSSet {
		return true, nil
	} else {
		return out, nil
	}
}

// KVSPut handles a DELETE request
func (s *HTTPServer) KVSDelete(resp http.ResponseWriter, req *http.Request, args *structs.KeyRequest) (interface{}, error) {
	if conflictingFlags(resp, req, "recurse", "cas") {
		return nil, nil
	}
	applyReq := structs.KVSRequest{
		Datacenter: args.Datacenter,
		Op:         structs.KVSDelete,
		DirEnt: structs.DirEntry{
			Key: args.Key,
		},
	}
	applyReq.Token = args.Token

	// Check for recurse
	params := req.URL.Query()
	if _, ok := params["recurse"]; ok {
		applyReq.Op = structs.KVSDeleteTree
	} else if missingKey(resp, args) {
		return nil, nil
	}

	// Check for cas value
	if _, ok := params["cas"]; ok {
		casVal, err := strconv.ParseUint(params.Get("cas"), 10, 64)
		if err != nil {
			return nil, err
		}
		applyReq.DirEnt.ModifyIndex = casVal
		applyReq.Op = structs.KVSDeleteCAS
	}

	// Make the RPC
	var out bool
	if err := s.agent.RPC("KVS.Apply", &applyReq, &out); err != nil {
		return nil, err
	}

	// Only use the out value if this was a CAS
	if applyReq.Op == structs.KVSDeleteCAS {
		return out, nil
	} else {
		return true, nil
	}
}

// missingKey checks if the key is missing
func missingKey(resp http.ResponseWriter, args *structs.KeyRequest) bool {
	if args.Key == "" {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing key name"))
		return true
	}
	return false
}

// conflictingFlags determines if non-composable flags were passed in a request.
func conflictingFlags(resp http.ResponseWriter, req *http.Request, flags ...string) bool {
	params := req.URL.Query()

	found := false
	for _, conflict := range flags {
		if _, ok := params[conflict]; ok {
			if found {
				resp.WriteHeader(400)
				resp.Write([]byte("Conflicting flags: " + params.Encode()))
				return true
			}
			found = true
		}
	}

	return false
}
