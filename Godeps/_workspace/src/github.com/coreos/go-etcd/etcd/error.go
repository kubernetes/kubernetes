package etcd

import (
	"encoding/json"
	"fmt"
)

const (
	ErrCodeEtcdNotReachable    = 501
	ErrCodeUnhandledHTTPStatus = 502
)

var (
	errorMap = map[int]string{
		ErrCodeEtcdNotReachable: "All the given peers are not reachable",
	}
)

type EtcdError struct {
	ErrorCode int    `json:"errorCode"`
	Message   string `json:"message"`
	Cause     string `json:"cause,omitempty"`
	Index     uint64 `json:"index"`
}

func (e EtcdError) Error() string {
	return fmt.Sprintf("%v: %v (%v) [%v]", e.ErrorCode, e.Message, e.Cause, e.Index)
}

func newError(errorCode int, cause string, index uint64) *EtcdError {
	return &EtcdError{
		ErrorCode: errorCode,
		Message:   errorMap[errorCode],
		Cause:     cause,
		Index:     index,
	}
}

func handleError(b []byte) error {
	etcdErr := new(EtcdError)

	err := json.Unmarshal(b, etcdErr)
	if err != nil {
		logger.Warningf("cannot unmarshal etcd error: %v", err)
		return err
	}

	return etcdErr
}
