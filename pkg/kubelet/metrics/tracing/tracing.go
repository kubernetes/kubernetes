package tracing

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	types "k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/clock"
)

var (
	enabled        = false
	tracingManager = NewManager()
)

// GetManager() exposes the tracing manager
func GetManager() Manager {
	return tracingManager
}

// SetEnableTracing enables tracing by activating tracing probes.
func SetEnableTracing(en bool) {
	enabled = en
}

// SetProbeWithTs places a probe in code to record the timestamp for an operation
// TODO(coufon): currently we do not use pod UID yet. The timestamps for an operation are
// stored in an array. In future we may store the timestamps for per pod per operation, so we
// can track operation timestamps for a specific pod.
func SetProbeWithTs(podUID types.UID, operation string, ts time.Time) {
	if enabled {
		tracingManager.NewSampleWithTs(podUID, operation, ts)
	}
}

// SetProbe places a probe in code and use current time as the timestamp.
func SetProbe(podUID types.UID, operation string) {
	if enabled {
		tracingManager.NewSample(podUID, operation)
	}
}

// Manager is the interface of tracing managers.
type Manager interface {
	NewSampleWithTs(types.UID, string, time.Time)
	NewSample(types.UID, string)
	GetSamplesJSON() ([]byte, error)
	ResetSamples()
	ServeHTTP(http.ResponseWriter, *http.Request)
}

// NewManager creates and returns a tracing manager.
func NewManager() Manager {
	return &lightweightManager{
		tsPerOperation: make(tsArrayMap),
		clock:          clock.RealClock{},
	}
}

// tsArrayMap is a map with operation name as key, and an array of timestamps as value.
type tsArrayMap map[string][]int64

// lightweightManager is a  lightweight implementation of tracing manager.
// It only records the sequence of timestamp when an operation starts. It does not support
// tracing of related operations for a specific pod.
// Currently this is enough for performance analysis and visualization.
type lightweightManager struct {
	tsPerOperation tsArrayMap
	// clock is an interface that provides time related functionality in a way that makes it
	// easy to test the code.
	clock clock.Clock
	// TODO(coufon): we use a read-write mutex to protect concurrent accesses to 'tsPerOperation'
	// It is OK when operation/probe number is small. If the number is large, we may have a mutex/spin lock
	// for each operation.
	rwmutex sync.RWMutex
}

func (lm *lightweightManager) NewSampleWithTs(podUID types.UID, operation string, ts time.Time) {
	lm.rwmutex.Lock()
	_, ok := lm.tsPerOperation[operation]
	if !ok {
		lm.tsPerOperation[operation] = []int64{}
	}
	lm.tsPerOperation[operation] = append(lm.tsPerOperation[operation], ts.UnixNano())
	lm.rwmutex.Unlock()
}

func (lm *lightweightManager) NewSample(podUID types.UID, operation string) {
	lm.NewSampleWithTs(podUID, operation, lm.clock.Now())
}

// ResetSamples clears all samples
func (lm *lightweightManager) ResetSamples() {
	lm.rwmutex.Lock()
	lm.tsPerOperation = tsArrayMap{}
	lm.rwmutex.Unlock()
}

// GetSamplesJSON returns all samples in JSON format
func (lm *lightweightManager) GetSamplesJSON() ([]byte, error) {
	lm.rwmutex.RLock()
	data, err := json.Marshal(lm.tsPerOperation)
	lm.rwmutex.RUnlock()
	if err != nil {
		return nil, err
	}
	return data, nil
}

// ServeHTTP is the http handler of tracing manager. It returns all samples in JSON for GET method,
// and clear all samples for DELETE method.
func (lm *lightweightManager) ServeHTTP(res http.ResponseWriter, req *http.Request) {
	if req.Method == "DELETE" {
		lm.ResetSamples()
		io.WriteString(res, "traces reset")
		return
	}

	data, err := lm.GetSamplesJSON()

	if err != nil {
		res.Header().Set("Content-type", "text/html")
		res.WriteHeader(http.StatusInternalServerError)
		res.Write([]byte(fmt.Sprintf("<h3>Internal Error</h3><p>%v", err)))
		return
	}
	res.Header().Set("Content-type", "application/json")
	res.WriteHeader(http.StatusOK)
	res.Write(data)
}
