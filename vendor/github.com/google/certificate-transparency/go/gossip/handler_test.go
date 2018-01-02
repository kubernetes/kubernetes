package gossip

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/certificate-transparency/go"
	"github.com/stretchr/testify/assert"
)

const (
	logIDB64 = `aPaY+B9kgr46jO65KB1M/HFRXWeT1ETRCmesu09P+8Q=`

	pubKey = "-----BEGIN PUBLIC KEY-----\n" +
		"MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE1/TMabLkDpCjiupacAlP7xNi0I1J\n" +
		"YP8bQFAHDG1xhtolSY1l4QgNRzRrvSe8liE+NPWHdjGxfx3JhTsN9x8/6Q==\n" +
		"-----END PUBLIC KEY-----\n"

	addSCTFeedbackJSON = `
      {
        "sct_feedback": [
          { "x509_chain": [
            "CHAIN00",
            "CHAIN01"
            ],
            "sct_data": [
            "SCT00",
            "SCT01",
            "SCT02"
            ]
          }, {
            "x509_chain": [
            "CHAIN10",
            "CHAIN11"
            ],
            "sct_data": [
            "SCT10",
            "SCT11",
            "SCT12"
            ]
          }
        ]
      }`

	stuckClockTimeMillis       = 1441360035224 // Fri Sep  4 10:47:15 BST 2015
	stuckClockTimeFutureMillis = 1450000000000 // Sun Dec 13 09:46:40 GMT 2015

	addSTHPollinationJSON = `
      {
        "sths": [
          {
            "sth_version": 0,
            "tree_size": 8285192,
            "timestamp": 1441360035224,
            "sha256_root_hash": "5g2CdT06dF6YcEDPYO50jQWqRvnGwi5BcgGYY10e3+I=",
            "tree_head_signature": "BAMASDBGAiEAnGFvHwZJsSMkj7nd+Hshd9lOcWQvi1HIA2t1D47I1W4CIQCGu7+aVm0y/hxWGk+HcFIqoA9DptQkdxUdgIrdq5LRQw==",
            "log_id": "aPaY+B9kgr46jO65KB1M/HFRXWeT1ETRCmesu09P+8Q="
          }, {
            "sth_version": 0,
            "tree_size": 8285157,
            "timestamp": 1441356438793,
            "sha256_root_hash": "A9YRqKNRutdXq3ADPeRxJrqAZv24w4bACrM9IBKK/io=",
            "tree_head_signature": "BAMARzBFAiEAxFSIDea57BRpB+RQxwd2/gzEieOXZx2Hvu7/0L0Oo7wCICH4hte0sPI6G5IGYJbL0lDTMjnGC7NmUOIQRBrm07vM",
            "log_id": "aPaY+B9kgr46jO65KB1M/HFRXWeT1ETRCmesu09P+8Q="
          }, {
            "sth_version": 0,
            "tree_size": 8285124,
            "timestamp": 1441352904860,
            "sha256_root_hash": "gIvD8vwCqzvI/cCM3vT5l5VBXbyeGXOgU1eymOHy2S0=",
            "tree_head_signature": "BAMARjBEAiAwV+gwZQVJxPDrVLc32QiWnu44Mw4wT3oK8AkyuihU/AIgb/qc3UPgZDSx+C9nadKCCsgrQ285Hm5HnngtfY7ph18=",
            "log_id": "aPaY+B9kgr46jO65KB1M/HFRXWeT1ETRCmesu09P+8Q="
          }
        ]
      }`

	addSTHPollinationUnknownLogIDJSON = `
      {
        "sths": [
          {
            "sth_version": 0,
            "tree_size": 8285192,
            "timestamp": 1441360035224,
            "sha256_root_hash": "5g2CdT06dF6YcEDPYO50jQWqRvnGwi5BcgGYY10e3+I=",
            "tree_head_signature": "BAMASDBGAiEAnGFvHwZJsSMkj7nd+Hshd9lOcWQvi1HIA2t1D47I1W4CIQCGu7+aVm0y/hxWGk+HcFIqoA9DptQkdxUdgIrdq5LRQw==",
            "log_id": "BLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLA="
          }
        ]
      }`

	addSTHPollinationInvalidSignatureJSON = `
      {
        "sths": [
          {
            "sth_version": 0,
            "tree_size": 8285192,
            "timestamp": 1441360035224,
            "sha256_root_hash": "5g2CdT06dF6YcEDPYO50jQWqRvnGwi5BcgGYY10e3+I=",
            "tree_head_signature": "BAMARLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAHBLAH",
            "log_id": "aPaY+B9kgr46jO65KB1M/HFRXWeT1ETRCmesu09P+8Q="
          }
        ]
      }`
)

type stuckClock struct {
	at time.Time
}

func (s stuckClock) Now() time.Time {
	return s.at
}

func createAndOpenStorage() *Storage {
	// Jump through some hoops to get a temp file name.
	// ioutil.TempFile(...) actually creates an empty file for us; we just want the name though, so we'll delete the created file.
	// (SQLite *may* be fine with opening a zero-byte file and assuming that's ok, but let's not chance it.)
	dbFile, err := ioutil.TempFile("", "handler_test")
	if err != nil {
		log.Fatalf("Failed to get a temporary file: %v", err)
	}
	if err := dbFile.Close(); err != nil {
		log.Fatalf("Failed to Close() temporary file: %v", err)
	}
	if err := os.Remove(dbFile.Name()); err != nil {
		log.Fatalf("Failed to Remove() temporary file: %v", err)
	}

	s := &Storage{}
	if err := s.Open(dbFile.Name()); err != nil {
		log.Fatalf("Failed to Open() storage: %v", err)
	}
	return s
}

func closeAndDeleteStorage(s *Storage) {
	s.Close()
	if err := os.Remove(s.dbPath); err != nil {
		log.Printf("Failed to remove test DB (%v): %v", s.dbPath, err)
	}
}

func mustCreateSignatureVerifiers(t *testing.T) SignatureVerifierMap {
	m := make(SignatureVerifierMap)
	key, id, _, err := ct.PublicKeyFromPEM([]byte(pubKey))
	if err != nil {
		t.Fatalf("Failed to parse pubkey: %v", err)
	}
	sv, err := ct.NewSignatureVerifier(key)
	if err != nil {
		t.Fatalf("Failed to create new SignatureVerifier: %v", err)
	}
	m[id] = *sv
	return m
}

func sctFeedbackFromString(t *testing.T, s string) SCTFeedback {
	json := json.NewDecoder(strings.NewReader(s))
	var f SCTFeedback
	if err := json.Decode(&f); err != nil {
		t.Fatalf("Failed to parse JSON: %v", err)
	}
	return f
}

func sthPollinationFromString(t *testing.T, s string) STHPollination {
	json := json.NewDecoder(strings.NewReader(s))
	var f STHPollination
	if err := json.Decode(&f); err != nil {
		t.Fatalf("Failed to parse JSON: %v", err)
	}
	return f
}

func expectStorageHasFeedback(t *testing.T, s *Storage, chain []string, sct string) {
	sctID, err := s.getSCTID(sct)
	if err != nil {
		t.Fatalf("Failed to look up ID for SCT %v: %v", sct, err)
	}
	chainID, err := s.getChainID(chain)
	if err != nil {
		t.Fatalf("Failed to look up ID for Chain %v: %v", chain, err)
	}
	assert.True(t, s.hasFeedback(sctID, chainID))
}

func mustGet(t *testing.T, f func() (int64, error)) int64 {
	v, err := f()
	if err != nil {
		t.Fatalf("Got error while calling %v: %v", f, err)
	}
	return v
}

func testStuckClock(m int64) stuckClock {
	return stuckClock{
		at: time.Unix(m/1000, 0),
	}
}

func TestHandlesValidSCTFeedback(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeMillis))

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sct-feedback", strings.NewReader(addSCTFeedbackJSON))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	h.HandleSCTFeedback(rr, req)
	assert.Equal(t, http.StatusOK, rr.Code)

	f := sctFeedbackFromString(t, addSCTFeedbackJSON)
	for _, entry := range f.Feedback {
		for _, sct := range entry.SCTData {
			expectStorageHasFeedback(t, s, entry.X509Chain, sct)
		}
	}
}

func TestHandlesDuplicatedSCTFeedback(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeMillis))

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sct-feedback", strings.NewReader(addSCTFeedbackJSON))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	for i := 0; i < 10; i++ {
		h.HandleSCTFeedback(rr, req)
		assert.Equal(t, http.StatusOK, rr.Code)
	}

	numExpectedChains := 0
	numExpectedSCTs := 0
	f := sctFeedbackFromString(t, addSCTFeedbackJSON)
	for _, entry := range f.Feedback {
		numExpectedChains++
		for _, sct := range entry.SCTData {
			numExpectedSCTs++
			expectStorageHasFeedback(t, s, entry.X509Chain, sct)
		}
	}

	assert.EqualValues(t, numExpectedChains, mustGet(t, s.getNumChains))
	assert.EqualValues(t, numExpectedSCTs, mustGet(t, s.getNumSCTs))
	assert.EqualValues(t, numExpectedSCTs, mustGet(t, s.getNumFeedback)) // one feedback entry per SCT/Chain pair
}

func TestRejectsInvalidSCTFeedback(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeMillis))

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sct-feedback", strings.NewReader("BlahBlah},"))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	h.HandleSCTFeedback(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestHandlesValidSTHPollination(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeMillis))

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sth-pollination", strings.NewReader(addSTHPollinationJSON))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	h.HandleSTHPollination(rr, req)
	if !assert.Equal(t, http.StatusOK, rr.Code) {
		t.Fatal(rr.Body.String())
	}

	f := sthPollinationFromString(t, addSTHPollinationJSON)

	assert.EqualValues(t, len(f.STHs), mustGet(t, s.getNumSTHs))
	for _, sth := range f.STHs {
		assert.True(t, s.hasSTH(sth))
	}
}

func TestHandlesDuplicateSTHPollination(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeMillis))

	pollen := sthPollinationFromString(t, addSTHPollinationJSON)
	pollenJSON, err := json.Marshal(pollen)
	if err != nil {
		t.Fatalf("Failed to marshal pollen JSON: %v", err)
	}

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sth-pollination", bytes.NewReader(pollenJSON))

	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	for i := 0; i < 10; i++ {
		h.HandleSTHPollination(rr, req)
		assert.Equal(t, http.StatusOK, rr.Code)
	}

	assert.EqualValues(t, len(pollen.STHs), mustGet(t, s.getNumSTHs))
	for _, sth := range pollen.STHs {
		assert.True(t, s.hasSTH(sth))
	}
}

func TestHandlesInvalidSTHPollination(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeMillis))

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sth-pollination", strings.NewReader("blahblah,,}{"))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	h.HandleSTHPollination(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestRejectsSTHFromUnknownLog(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeMillis))

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sth-pollination", strings.NewReader(addSTHPollinationUnknownLogIDJSON))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	h.HandleSTHPollination(rr, req)
	if !assert.Equal(t, http.StatusOK, rr.Code) {
		t.Fatal(rr.Body.String())
	}

	assert.EqualValues(t, 0, mustGet(t, s.getNumSTHs))
}

func TestRejectsSTHWithInvalidSignature(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeMillis))

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sth-pollination", strings.NewReader(addSTHPollinationInvalidSignatureJSON))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	h.HandleSTHPollination(rr, req)
	if !assert.Equal(t, http.StatusOK, rr.Code) {
		t.Fatal(rr.Body.String())
	}

	assert.EqualValues(t, 0, mustGet(t, s.getNumSTHs))
}

func TestReturnsSTHPollination(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeMillis))

	sentPollen := sthPollinationFromString(t, addSTHPollinationJSON)
	sentPollenJSON, err := json.Marshal(sentPollen)
	if err != nil {
		t.Fatalf("Failed to marshal pollen JSON: %v", err)
	}

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sth-pollination", bytes.NewReader(sentPollenJSON))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	h.HandleSTHPollination(rr, req)
	assert.Equal(t, http.StatusOK, rr.Code)

	// Make the request again because it seems there's a race inside (go-)sqlite3
	// somewhere; occasionally the storage handler doesn't see any pollen
	// despite the fact that the transaction which wrote it committed before
	// the select was executed.
	h.HandleSTHPollination(rr, req)
	assert.Equal(t, http.StatusOK, rr.Code)

	// since this is an empty DB, we should get back all of the pollination we sent
	// TODO(alcutter): We probably shouldn't blindly return stuff we were just given really, that's kinda silly, but it'll do for now.
	recvPollen := sthPollinationFromString(t, rr.Body.String())

	for _, sth := range sentPollen.STHs {
		assert.Contains(t, recvPollen.STHs, sth)
	}

	assert.Equal(t, len(sentPollen.STHs), len(recvPollen.STHs))
}

func TestDoesNotReturnStalePollen(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeFutureMillis))

	sentPollen := sthPollinationFromString(t, addSTHPollinationJSON)
	sentPollenJSON, err := json.Marshal(sentPollen)
	if err != nil {
		t.Fatalf("Failed to marshal pollen JSON: %v", err)
	}

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sth-pollination", bytes.NewReader(sentPollenJSON))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	h.HandleSTHPollination(rr, req)
	assert.Equal(t, http.StatusOK, rr.Code)

	// since this is an empty DB, the only pollen available is the non-fresh stuff we just sent, so expect to get nothing back.
	recvPollen := sthPollinationFromString(t, rr.Body.String())
	assert.Equal(t, 0, len(recvPollen.STHs))
}

func TestLimitsSTHPollinationReturned(t *testing.T) {
	s := createAndOpenStorage()
	defer closeAndDeleteStorage(s)

	*defaultNumPollinationsToReturn = 1
	v := mustCreateSignatureVerifiers(t)
	h := newHandlerWithClock(s, v, testStuckClock(stuckClockTimeMillis))

	sentPollen := sthPollinationFromString(t, addSTHPollinationJSON)
	sentPollenJSON, err := json.Marshal(sentPollen)
	if err != nil {
		t.Fatalf("Failed to marshal pollen JSON: %v", err)
	}

	rr := httptest.NewRecorder()
	req, err := http.NewRequest("POST", "/.well-known/ct/v1/sth-pollination", bytes.NewReader(sentPollenJSON))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	h.HandleSTHPollination(rr, req)
	assert.Equal(t, http.StatusOK, rr.Code)

	// Make the request again because it seems there's a race inside (go-)sqlite3
	// somewhere; occasionally the storage handler doesn't see any pollen
	// despite the fact that the transaction which wrote it committed before
	// the select was executed.
	h.HandleSTHPollination(rr, req)
	assert.Equal(t, http.StatusOK, rr.Code)

	recvPollen := sthPollinationFromString(t, rr.Body.String())

	assert.Equal(t, 1, len(recvPollen.STHs))
	assert.Contains(t, sentPollen.STHs, recvPollen.STHs[0])
}
