package types

import (
	"encoding/json"
)

type RemoteBeforeSuiteState int

const (
	RemoteBeforeSuiteStateInvalid RemoteBeforeSuiteState = iota

	RemoteBeforeSuiteStatePending
	RemoteBeforeSuiteStatePassed
	RemoteBeforeSuiteStateFailed
	RemoteBeforeSuiteStateDisappeared
)

type RemoteBeforeSuiteData struct {
	Data  []byte
	State RemoteBeforeSuiteState
}

func (r RemoteBeforeSuiteData) ToJSON() []byte {
	data, _ := json.Marshal(r)
	return data
}

type RemoteAfterSuiteData struct {
	CanRun bool
}
