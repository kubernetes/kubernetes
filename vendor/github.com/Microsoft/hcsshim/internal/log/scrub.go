package log

import (
	"bytes"
	"encoding/json"
	"errors"
	"sync/atomic"

	hcsschema "github.com/Microsoft/hcsshim/internal/hcs/schema2"
)

// This package scrubs objects of potentially sensitive information to pass to logging

type genMap = map[string]interface{}
type scrubberFunc func(genMap) error

const _scrubbedReplacement = "<scrubbed>"

var (
	ErrUnknownType = errors.New("encoded object is of unknown type")

	// case sensitive keywords, so "env" is not a substring on "Environment"
	_scrubKeywords = [][]byte{[]byte("env"), []byte("Environment")}

	_scrub int32
)

// SetScrubbing enables scrubbing
func SetScrubbing(enable bool) {
	v := int32(0) // cant convert from bool to int32 directly
	if enable {
		v = 1
	}
	atomic.StoreInt32(&_scrub, v)
}

// IsScrubbingEnabled checks if scrubbing is enabled
func IsScrubbingEnabled() bool {
	v := atomic.LoadInt32(&_scrub)
	return v != 0
}

// ScrubProcessParameters scrubs HCS Create Process requests with config parameters of
// type internal/hcs/schema2.ScrubProcessParameters (aka hcsshema.ScrubProcessParameters)
func ScrubProcessParameters(s string) (string, error) {
	// todo: deal with v1 ProcessConfig
	b := []byte(s)
	if !IsScrubbingEnabled() || !hasKeywords(b) || !json.Valid(b) {
		return s, nil
	}

	pp := hcsschema.ProcessParameters{}
	if err := json.Unmarshal(b, &pp); err != nil {
		return "", err
	}
	pp.Environment = map[string]string{_scrubbedReplacement: _scrubbedReplacement}

	b, err := encode(pp)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ScrubBridgeCreate scrubs requests sent over the bridge of type
// internal/gcs/protocol.containerCreate wrapping an internal/hcsoci.linuxHostedSystem
func ScrubBridgeCreate(b []byte) ([]byte, error) {
	return scrubBytes(b, scrubBridgeCreate)
}

func scrubBridgeCreate(m genMap) error {
	if !isRequestBase(m) {
		return ErrUnknownType
	}
	if ss, ok := m["ContainerConfig"]; ok {
		// ContainerConfig is a json encoded struct passed as a regular string field
		s, ok := ss.(string)
		if !ok {
			return ErrUnknownType
		}
		b, err := scrubBytes([]byte(s), scrubLinuxHostedSystem)
		if err != nil {
			return err
		}
		m["ContainerConfig"] = string(b)
		return nil
	}
	return ErrUnknownType
}

func scrubLinuxHostedSystem(m genMap) error {
	if m, ok := index(m, "OciSpecification"); ok { //nolint:govet // shadow
		if _, ok := m["annotations"]; ok {
			m["annotations"] = map[string]string{_scrubbedReplacement: _scrubbedReplacement}
		}
		if m, ok := index(m, "process"); ok { //nolint:govet // shadow
			if _, ok := m["env"]; ok {
				m["env"] = []string{_scrubbedReplacement}
				return nil
			}
		}
	}
	return ErrUnknownType
}

// ScrubBridgeExecProcess scrubs requests sent over the bridge of type
// internal/gcs/protocol.containerExecuteProcess
func ScrubBridgeExecProcess(b []byte) ([]byte, error) {
	return scrubBytes(b, scrubExecuteProcess)
}

func scrubExecuteProcess(m genMap) error {
	if !isRequestBase(m) {
		return ErrUnknownType
	}
	if m, ok := index(m, "Settings"); ok { //nolint:govet // shadow
		if ss, ok := m["ProcessParameters"]; ok {
			// ProcessParameters is a json encoded struct passed as a regular sting field
			s, ok := ss.(string)
			if !ok {
				return ErrUnknownType
			}

			s, err := ScrubProcessParameters(s)
			if err != nil {
				return err
			}

			m["ProcessParameters"] = s
			return nil
		}
	}
	return ErrUnknownType
}

func scrubBytes(b []byte, scrub scrubberFunc) ([]byte, error) {
	if !IsScrubbingEnabled() || !hasKeywords(b) || !json.Valid(b) {
		return b, nil
	}

	m := make(genMap)
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}

	// could use regexp, but if the env strings contain braces, the regexp fails
	// parsing into individual structs would require access to private structs
	if err := scrub(m); err != nil {
		return nil, err
	}

	b, err := encode(m)
	if err != nil {
		return nil, err
	}

	return b, nil
}

func isRequestBase(m genMap) bool {
	// neither of these are (currently) `omitempty`
	_, a := m["ActivityId"]
	_, c := m["ContainerId"]
	return a && c
}

// combination `m, ok := m[s]` and `m, ok := m.(genMap)`
func index(m genMap, s string) (genMap, bool) {
	if m, ok := m[s]; ok {
		mm, ok := m.(genMap)
		return mm, ok
	}

	return m, false
}

func hasKeywords(b []byte) bool {
	for _, bb := range _scrubKeywords {
		if bytes.Contains(b, bb) {
			return true
		}
	}
	return false
}
