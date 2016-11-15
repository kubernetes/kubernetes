// Package client is a CT log client implementation and contains types and code
// for interacting with RFC6962-compliant CT Log instances.
// See http://tools.ietf.org/html/rfc6962 for details
package client

import (
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/google/certificate-transparency/go"
	"github.com/mreiferson/go-httpclient"
	"golang.org/x/net/context"
)

// URI paths for CT Log endpoints
const (
	AddChainPath    = "/ct/v1/add-chain"
	AddPreChainPath = "/ct/v1/add-pre-chain"
	GetSTHPath      = "/ct/v1/get-sth"
	GetEntriesPath  = "/ct/v1/get-entries"
)

// LogClient represents a client for a given CT Log instance
type LogClient struct {
	uri        string       // the base URI of the log. e.g. http://ct.googleapis/pilot
	httpClient *http.Client // used to interact with the log via HTTP
}

//////////////////////////////////////////////////////////////////////////////////
// JSON structures follow.
// These represent the structures returned by the CT Log server.
//////////////////////////////////////////////////////////////////////////////////

// addChainRequest represents the JSON request body sent to the add-chain CT
// method.
type addChainRequest struct {
	Chain []string `json:"chain"`
}

// addChainResponse represents the JSON response to the add-chain CT method.
// An SCT represents a Log's promise to integrate a [pre-]certificate into the
// log within a defined period of time.
type addChainResponse struct {
	SCTVersion ct.Version `json:"sct_version"` // SCT structure version
	ID         string     `json:"id"`          // Log ID
	Timestamp  uint64     `json:"timestamp"`   // Timestamp of issuance
	Extensions string     `json:"extensions"`  // Holder for any CT extensions
	Signature  string     `json:"signature"`   // Log signature for this SCT
}

// getSTHResponse respresents the JSON response to the get-sth CT method
type getSTHResponse struct {
	TreeSize          uint64 `json:"tree_size"`           // Number of certs in the current tree
	Timestamp         uint64 `json:"timestamp"`           // Time that the tree was created
	SHA256RootHash    string `json:"sha256_root_hash"`    // Root hash of the tree
	TreeHeadSignature string `json:"tree_head_signature"` // Log signature for this STH
}

// base64LeafEntry respresents a Base64 encoded leaf entry
type base64LeafEntry struct {
	LeafInput string `json:"leaf_input"`
	ExtraData string `json:"extra_data"`
}

// getEntriesReponse respresents the JSON response to the CT get-entries method
type getEntriesResponse struct {
	Entries []base64LeafEntry `json:"entries"` // the list of returned entries
}

// getConsistencyProofResponse represents the JSON response to the CT get-consistency-proof method
type getConsistencyProofResponse struct {
	Consistency []string `json:"consistency"`
}

// getAuditProofResponse represents the JSON response to the CT get-audit-proof method
type getAuditProofResponse struct {
	Hash     []string `json:"hash"`      // the hashes which make up the proof
	TreeSize uint64   `json:"tree_size"` // the tree size against which this proof is constructed
}

// getAcceptedRootsResponse represents the JSON response to the CT get-roots method.
type getAcceptedRootsResponse struct {
	Certificates []string `json:"certificates"`
}

// getEntryAndProodReponse represents the JSON response to the CT get-entry-and-proof method
type getEntryAndProofResponse struct {
	LeafInput string   `json:"leaf_input"` // the entry itself
	ExtraData string   `json:"extra_data"` // any chain provided when the entry was added to the log
	AuditPath []string `json:"audit_path"` // the corresponding proof
}

// New constructs a new LogClient instance.
// |uri| is the base URI of the CT log instance to interact with, e.g.
// http://ct.googleapis.com/pilot
func New(uri string) *LogClient {
	var c LogClient
	c.uri = uri
	transport := &httpclient.Transport{
		ConnectTimeout:        10 * time.Second,
		RequestTimeout:        30 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
		MaxIdleConnsPerHost:   10,
		DisableKeepAlives:     false,
	}
	c.httpClient = &http.Client{Transport: transport}
	return &c
}

// Makes a HTTP call to |uri|, and attempts to parse the response as a JSON
// representation of the structure in |res|.
// Returns a non-nil |error| if there was a problem.
func (c *LogClient) fetchAndParse(uri string, res interface{}) error {
	req, err := http.NewRequest("GET", uri, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Keep-Alive", "timeout=15, max=100")
	resp, err := c.httpClient.Do(req)
	var body []byte
	if resp != nil {
		body, err = ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return err
		}
	}
	if err != nil {
		return err
	}
	if err = json.Unmarshal(body, &res); err != nil {
		return err
	}
	return nil
}

// Makes a HTTP POST call to |uri|, and attempts to parse the response as a JSON
// representation of the structure in |res|.
// Returns a non-nil |error| if there was a problem.
func (c *LogClient) postAndParse(uri string, req interface{}, res interface{}) (*http.Response, string, error) {
	postBody, err := json.Marshal(req)
	if err != nil {
		return nil, "", err
	}
	httpReq, err := http.NewRequest("POST", uri, bytes.NewReader(postBody))
	if err != nil {
		return nil, "", err
	}
	httpReq.Header.Set("Keep-Alive", "timeout=15, max=100")
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient.Do(httpReq)
	// Read all of the body, if there is one, so that the http.Client can do
	// Keep-Alive:
	var body []byte
	if resp != nil {
		body, err = ioutil.ReadAll(resp.Body)
		resp.Body.Close()
	}
	if err != nil {
		return resp, string(body), err
	}
	if resp.StatusCode == 200 {
		if err != nil {
			return resp, string(body), err
		}
		if err = json.Unmarshal(body, &res); err != nil {
			return resp, string(body), err
		}
	}
	return resp, string(body), nil
}

func backoffForRetry(ctx context.Context, d time.Duration) error {
	backoffTimer := time.NewTimer(d)
	if ctx != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-backoffTimer.C:
		}
	} else {
		<-backoffTimer.C
	}
	return nil
}

// Attempts to add |chain| to the log, using the api end-point specified by
// |path|. If provided context expires before submission is complete an
// error will be returned.
func (c *LogClient) addChainWithRetry(ctx context.Context, path string, chain []ct.ASN1Cert) (*ct.SignedCertificateTimestamp, error) {
	var resp addChainResponse
	var req addChainRequest
	for _, link := range chain {
		req.Chain = append(req.Chain, base64.StdEncoding.EncodeToString(link))
	}
	httpStatus := "Unknown"
	backoffSeconds := 0
	done := false
	for !done {
		if backoffSeconds > 0 {
			log.Printf("Got %s, backing-off %d seconds", httpStatus, backoffSeconds)
		}
		err := backoffForRetry(ctx, time.Second*time.Duration(backoffSeconds))
		if err != nil {
			return nil, err
		}
		if backoffSeconds > 0 {
			backoffSeconds = 0
		}
		httpResp, errorBody, err := c.postAndParse(c.uri+path, &req, &resp)
		if err != nil {
			backoffSeconds = 10
			continue
		}
		switch {
		case httpResp.StatusCode == 200:
			done = true
		case httpResp.StatusCode == 408:
			// request timeout, retry immediately
		case httpResp.StatusCode == 503:
			// Retry
			backoffSeconds = 10
			if retryAfter := httpResp.Header.Get("Retry-After"); retryAfter != "" {
				if seconds, err := strconv.Atoi(retryAfter); err == nil {
					backoffSeconds = seconds
				}
			}
		default:
			return nil, fmt.Errorf("got HTTP Status %s: %s", httpResp.Status, errorBody)
		}
		httpStatus = httpResp.Status
	}

	rawLogID, err := base64.StdEncoding.DecodeString(resp.ID)
	if err != nil {
		return nil, err
	}
	rawSignature, err := base64.StdEncoding.DecodeString(resp.Signature)
	if err != nil {
		return nil, err
	}
	ds, err := ct.UnmarshalDigitallySigned(bytes.NewReader(rawSignature))
	if err != nil {
		return nil, err
	}
	var logID ct.SHA256Hash
	copy(logID[:], rawLogID)
	return &ct.SignedCertificateTimestamp{
		SCTVersion: resp.SCTVersion,
		LogID:      logID,
		Timestamp:  resp.Timestamp,
		Extensions: ct.CTExtensions(resp.Extensions),
		Signature:  *ds}, nil
}

// AddChain adds the (DER represented) X509 |chain| to the log.
func (c *LogClient) AddChain(chain []ct.ASN1Cert) (*ct.SignedCertificateTimestamp, error) {
	return c.addChainWithRetry(nil, AddChainPath, chain)
}

// AddPreChain adds the (DER represented) Precertificate |chain| to the log.
func (c *LogClient) AddPreChain(chain []ct.ASN1Cert) (*ct.SignedCertificateTimestamp, error) {
	return c.addChainWithRetry(nil, AddPreChainPath, chain)
}

// AddChainWithContext adds the (DER represented) X509 |chain| to the log and
// fails if the provided context expires before the chain is submitted.
func (c *LogClient) AddChainWithContext(ctx context.Context, chain []ct.ASN1Cert) (*ct.SignedCertificateTimestamp, error) {
	return c.addChainWithRetry(ctx, AddChainPath, chain)
}

// GetSTH retrieves the current STH from the log.
// Returns a populated SignedTreeHead, or a non-nil error.
func (c *LogClient) GetSTH() (sth *ct.SignedTreeHead, err error) {
	var resp getSTHResponse
	if err = c.fetchAndParse(c.uri+GetSTHPath, &resp); err != nil {
		return
	}
	sth = &ct.SignedTreeHead{
		TreeSize:  resp.TreeSize,
		Timestamp: resp.Timestamp,
	}

	rawRootHash, err := base64.StdEncoding.DecodeString(resp.SHA256RootHash)
	if err != nil {
		return nil, fmt.Errorf("invalid base64 encoding in sha256_root_hash: %v", err)
	}
	if len(rawRootHash) != sha256.Size {
		return nil, fmt.Errorf("sha256_root_hash is invalid length, expected %d got %d", sha256.Size, len(rawRootHash))
	}
	copy(sth.SHA256RootHash[:], rawRootHash)

	rawSignature, err := base64.StdEncoding.DecodeString(resp.TreeHeadSignature)
	if err != nil {
		return nil, errors.New("invalid base64 encoding in tree_head_signature")
	}
	ds, err := ct.UnmarshalDigitallySigned(bytes.NewReader(rawSignature))
	if err != nil {
		return nil, err
	}
	// TODO(alcutter): Verify signature
	sth.TreeHeadSignature = *ds
	return
}

// GetEntries attempts to retrieve the entries in the sequence [|start|, |end|] from the CT
// log server. (see section 4.6.)
// Returns a slice of LeafInputs or a non-nil error.
func (c *LogClient) GetEntries(start, end int64) ([]ct.LogEntry, error) {
	if end < 0 {
		return nil, errors.New("end should be >= 0")
	}
	if end < start {
		return nil, errors.New("start should be <= end")
	}
	var resp getEntriesResponse
	err := c.fetchAndParse(fmt.Sprintf("%s%s?start=%d&end=%d", c.uri, GetEntriesPath, start, end), &resp)
	if err != nil {
		return nil, err
	}
	entries := make([]ct.LogEntry, len(resp.Entries))
	for index, entry := range resp.Entries {
		leafBytes, err := base64.StdEncoding.DecodeString(entry.LeafInput)
		leaf, err := ct.ReadMerkleTreeLeaf(bytes.NewBuffer(leafBytes))
		if err != nil {
			return nil, err
		}
		entries[index].Leaf = *leaf
		chainBytes, err := base64.StdEncoding.DecodeString(entry.ExtraData)

		var chain []ct.ASN1Cert
		switch leaf.TimestampedEntry.EntryType {
		case ct.X509LogEntryType:
			chain, err = ct.UnmarshalX509ChainArray(chainBytes)

		case ct.PrecertLogEntryType:
			chain, err = ct.UnmarshalPrecertChainArray(chainBytes)

		default:
			return nil, fmt.Errorf("saw unknown entry type: %v", leaf.TimestampedEntry.EntryType)
		}
		if err != nil {
			return nil, err
		}
		entries[index].Chain = chain
		entries[index].Index = start + int64(index)
	}
	return entries, nil
}
