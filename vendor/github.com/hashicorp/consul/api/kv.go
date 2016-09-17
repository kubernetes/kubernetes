package api

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
)

// KVPair is used to represent a single K/V entry
type KVPair struct {
	Key         string
	CreateIndex uint64
	ModifyIndex uint64
	LockIndex   uint64
	Flags       uint64
	Value       []byte
	Session     string
}

// KVPairs is a list of KVPair objects
type KVPairs []*KVPair

// KVOp constants give possible operations available in a KVTxn.
type KVOp string

const (
	KVSet          KVOp = "set"
	KVDelete            = "delete"
	KVDeleteCAS         = "delete-cas"
	KVDeleteTree        = "delete-tree"
	KVCAS               = "cas"
	KVLock              = "lock"
	KVUnlock            = "unlock"
	KVGet               = "get"
	KVGetTree           = "get-tree"
	KVCheckSession      = "check-session"
	KVCheckIndex        = "check-index"
)

// KVTxnOp defines a single operation inside a transaction.
type KVTxnOp struct {
	Verb    string
	Key     string
	Value   []byte
	Flags   uint64
	Index   uint64
	Session string
}

// KVTxnOps defines a set of operations to be performed inside a single
// transaction.
type KVTxnOps []*KVTxnOp

// KVTxnResponse has the outcome of a transaction.
type KVTxnResponse struct {
	Results []*KVPair
	Errors  TxnErrors
}

// KV is used to manipulate the K/V API
type KV struct {
	c *Client
}

// KV is used to return a handle to the K/V apis
func (c *Client) KV() *KV {
	return &KV{c}
}

// Get is used to lookup a single key
func (k *KV) Get(key string, q *QueryOptions) (*KVPair, *QueryMeta, error) {
	resp, qm, err := k.getInternal(key, nil, q)
	if err != nil {
		return nil, nil, err
	}
	if resp == nil {
		return nil, qm, nil
	}
	defer resp.Body.Close()

	var entries []*KVPair
	if err := decodeBody(resp, &entries); err != nil {
		return nil, nil, err
	}
	if len(entries) > 0 {
		return entries[0], qm, nil
	}
	return nil, qm, nil
}

// List is used to lookup all keys under a prefix
func (k *KV) List(prefix string, q *QueryOptions) (KVPairs, *QueryMeta, error) {
	resp, qm, err := k.getInternal(prefix, map[string]string{"recurse": ""}, q)
	if err != nil {
		return nil, nil, err
	}
	if resp == nil {
		return nil, qm, nil
	}
	defer resp.Body.Close()

	var entries []*KVPair
	if err := decodeBody(resp, &entries); err != nil {
		return nil, nil, err
	}
	return entries, qm, nil
}

// Keys is used to list all the keys under a prefix. Optionally,
// a separator can be used to limit the responses.
func (k *KV) Keys(prefix, separator string, q *QueryOptions) ([]string, *QueryMeta, error) {
	params := map[string]string{"keys": ""}
	if separator != "" {
		params["separator"] = separator
	}
	resp, qm, err := k.getInternal(prefix, params, q)
	if err != nil {
		return nil, nil, err
	}
	if resp == nil {
		return nil, qm, nil
	}
	defer resp.Body.Close()

	var entries []string
	if err := decodeBody(resp, &entries); err != nil {
		return nil, nil, err
	}
	return entries, qm, nil
}

func (k *KV) getInternal(key string, params map[string]string, q *QueryOptions) (*http.Response, *QueryMeta, error) {
	r := k.c.newRequest("GET", "/v1/kv/"+key)
	r.setQueryOptions(q)
	for param, val := range params {
		r.params.Set(param, val)
	}
	rtt, resp, err := k.c.doRequest(r)
	if err != nil {
		return nil, nil, err
	}

	qm := &QueryMeta{}
	parseQueryMeta(resp, qm)
	qm.RequestTime = rtt

	if resp.StatusCode == 404 {
		resp.Body.Close()
		return nil, qm, nil
	} else if resp.StatusCode != 200 {
		resp.Body.Close()
		return nil, nil, fmt.Errorf("Unexpected response code: %d", resp.StatusCode)
	}
	return resp, qm, nil
}

// Put is used to write a new value. Only the
// Key, Flags and Value is respected.
func (k *KV) Put(p *KVPair, q *WriteOptions) (*WriteMeta, error) {
	params := make(map[string]string, 1)
	if p.Flags != 0 {
		params["flags"] = strconv.FormatUint(p.Flags, 10)
	}
	_, wm, err := k.put(p.Key, params, p.Value, q)
	return wm, err
}

// CAS is used for a Check-And-Set operation. The Key,
// ModifyIndex, Flags and Value are respected. Returns true
// on success or false on failures.
func (k *KV) CAS(p *KVPair, q *WriteOptions) (bool, *WriteMeta, error) {
	params := make(map[string]string, 2)
	if p.Flags != 0 {
		params["flags"] = strconv.FormatUint(p.Flags, 10)
	}
	params["cas"] = strconv.FormatUint(p.ModifyIndex, 10)
	return k.put(p.Key, params, p.Value, q)
}

// Acquire is used for a lock acquisition operation. The Key,
// Flags, Value and Session are respected. Returns true
// on success or false on failures.
func (k *KV) Acquire(p *KVPair, q *WriteOptions) (bool, *WriteMeta, error) {
	params := make(map[string]string, 2)
	if p.Flags != 0 {
		params["flags"] = strconv.FormatUint(p.Flags, 10)
	}
	params["acquire"] = p.Session
	return k.put(p.Key, params, p.Value, q)
}

// Release is used for a lock release operation. The Key,
// Flags, Value and Session are respected. Returns true
// on success or false on failures.
func (k *KV) Release(p *KVPair, q *WriteOptions) (bool, *WriteMeta, error) {
	params := make(map[string]string, 2)
	if p.Flags != 0 {
		params["flags"] = strconv.FormatUint(p.Flags, 10)
	}
	params["release"] = p.Session
	return k.put(p.Key, params, p.Value, q)
}

func (k *KV) put(key string, params map[string]string, body []byte, q *WriteOptions) (bool, *WriteMeta, error) {
	if len(key) > 0 && key[0] == '/' {
		return false, nil, fmt.Errorf("Invalid key. Key must not begin with a '/': %s", key)
	}

	r := k.c.newRequest("PUT", "/v1/kv/"+key)
	r.setWriteOptions(q)
	for param, val := range params {
		r.params.Set(param, val)
	}
	r.body = bytes.NewReader(body)
	rtt, resp, err := requireOK(k.c.doRequest(r))
	if err != nil {
		return false, nil, err
	}
	defer resp.Body.Close()

	qm := &WriteMeta{}
	qm.RequestTime = rtt

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, resp.Body); err != nil {
		return false, nil, fmt.Errorf("Failed to read response: %v", err)
	}
	res := strings.Contains(string(buf.Bytes()), "true")
	return res, qm, nil
}

// Delete is used to delete a single key
func (k *KV) Delete(key string, w *WriteOptions) (*WriteMeta, error) {
	_, qm, err := k.deleteInternal(key, nil, w)
	return qm, err
}

// DeleteCAS is used for a Delete Check-And-Set operation. The Key
// and ModifyIndex are respected. Returns true on success or false on failures.
func (k *KV) DeleteCAS(p *KVPair, q *WriteOptions) (bool, *WriteMeta, error) {
	params := map[string]string{
		"cas": strconv.FormatUint(p.ModifyIndex, 10),
	}
	return k.deleteInternal(p.Key, params, q)
}

// DeleteTree is used to delete all keys under a prefix
func (k *KV) DeleteTree(prefix string, w *WriteOptions) (*WriteMeta, error) {
	_, qm, err := k.deleteInternal(prefix, map[string]string{"recurse": ""}, w)
	return qm, err
}

func (k *KV) deleteInternal(key string, params map[string]string, q *WriteOptions) (bool, *WriteMeta, error) {
	r := k.c.newRequest("DELETE", "/v1/kv/"+key)
	r.setWriteOptions(q)
	for param, val := range params {
		r.params.Set(param, val)
	}
	rtt, resp, err := requireOK(k.c.doRequest(r))
	if err != nil {
		return false, nil, err
	}
	defer resp.Body.Close()

	qm := &WriteMeta{}
	qm.RequestTime = rtt

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, resp.Body); err != nil {
		return false, nil, fmt.Errorf("Failed to read response: %v", err)
	}
	res := strings.Contains(string(buf.Bytes()), "true")
	return res, qm, nil
}

// TxnOp is the internal format we send to Consul. It's not specific to KV,
// though currently only KV operations are supported.
type TxnOp struct {
	KV *KVTxnOp
}

// TxnOps is a list of transaction operations.
type TxnOps []*TxnOp

// TxnResult is the internal format we receive from Consul.
type TxnResult struct {
	KV *KVPair
}

// TxnResults is a list of TxnResult objects.
type TxnResults []*TxnResult

// TxnError is used to return information about an operation in a transaction.
type TxnError struct {
	OpIndex int
	What    string
}

// TxnErrors is a list of TxnError objects.
type TxnErrors []*TxnError

// TxnResponse is the internal format we receive from Consul.
type TxnResponse struct {
	Results TxnResults
	Errors  TxnErrors
}

// Txn is used to apply multiple KV operations in a single, atomic transaction.
//
// Note that Go will perform the required base64 encoding on the values
// automatically because the type is a byte slice. Transactions are defined as a
// list of operations to perform, using the KVOp constants and KVTxnOp structure
// to define operations. If any operation fails, none of the changes are applied
// to the state store. Note that this hides the internal raw transaction interface
// and munges the input and output types into KV-specific ones for ease of use.
// If there are more non-KV operations in the future we may break out a new
// transaction API client, but it will be easy to keep this KV-specific variant
// supported.
//
// Even though this is generally a write operation, we take a QueryOptions input
// and return a QueryMeta output. If the transaction contains only read ops, then
// Consul will fast-path it to a different endpoint internally which supports
// consistency controls, but not blocking. If there are write operations then
// the request will always be routed through raft and any consistency settings
// will be ignored.
//
// Here's an example:
//
// ops := KVTxnOps{
//     &KVTxnOp{
//         Verb:    KVLock,
//         Key:     "test/lock",
//         Session: "adf4238a-882b-9ddc-4a9d-5b6758e4159e",
//         Value:   []byte("hello"),
//     },
//     &KVTxnOp{
//         Verb:    KVGet,
//         Key:     "another/key",
//     },
// }
// ok, response, _, err := kv.Txn(&ops, nil)
//
// If there is a problem making the transaction request then an error will be
// returned. Otherwise, the ok value will be true if the transaction succeeded
// or false if it was rolled back. The response is a structured return value which
// will have the outcome of the transaction. Its Results member will have entries
// for each operation. Deleted keys will have a nil entry in the, and to save
// space, the Value of each key in the Results will be nil unless the operation
// is a KVGet. If the transaction was rolled back, the Errors member will have
// entries referencing the index of the operation that failed along with an error
// message.
func (k *KV) Txn(txn KVTxnOps, q *QueryOptions) (bool, *KVTxnResponse, *QueryMeta, error) {
	r := k.c.newRequest("PUT", "/v1/txn")
	r.setQueryOptions(q)

	// Convert into the internal format since this is an all-KV txn.
	ops := make(TxnOps, 0, len(txn))
	for _, kvOp := range txn {
		ops = append(ops, &TxnOp{KV: kvOp})
	}
	r.obj = ops
	rtt, resp, err := k.c.doRequest(r)
	if err != nil {
		return false, nil, nil, err
	}
	defer resp.Body.Close()

	qm := &QueryMeta{}
	parseQueryMeta(resp, qm)
	qm.RequestTime = rtt

	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusConflict {
		var txnResp TxnResponse
		if err := decodeBody(resp, &txnResp); err != nil {
			return false, nil, nil, err
		}

		// Convert from the internal format.
		kvResp := KVTxnResponse{
			Errors: txnResp.Errors,
		}
		for _, result := range txnResp.Results {
			kvResp.Results = append(kvResp.Results, result.KV)
		}
		return resp.StatusCode == http.StatusOK, &kvResp, qm, nil
	}

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, resp.Body); err != nil {
		return false, nil, nil, fmt.Errorf("Failed to read response: %v", err)
	}
	return false, nil, nil, fmt.Errorf("Failed request: %s", buf.String())
}
