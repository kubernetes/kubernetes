/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package sharingaffinity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/google/cel-go/interpreter"

	sigsjson "sigs.k8s.io/json"
)

var (
	ErrClaimSelfInconsistent = errors.New("claim self-inconsistent")
	ErrInvalidDefinition     = errors.New("invalid sharing affinity definition")
	ErrParameters            = errors.New("invalid opaque parameters")
	ErrCELEval               = errors.New("cel evaluation failed")
	ErrNonStringReturn       = errors.New("cel expression returned non-string")
	ErrCostExceeded          = errors.New("cel evaluation cost exceeded")
	ErrMissingField          = errors.New("missing field")

	errNonStringResult = errors.New("non-string result")
)

// Error attributes a definition or extraction failure without rendering inputs.
type Error struct {
	Kind      error
	Pool      PoolKey
	Request   string
	Extractor string
	// ConfigIndex is the original config index, or -1 when not applicable.
	ConfigIndex int
	Err         error
}

func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}
	var b strings.Builder
	if e.Kind != nil {
		b.WriteString(e.Kind.Error())
	} else {
		b.WriteString("sharing affinity extraction failed")
	}
	if e.Pool.Driver != "" || e.Pool.Name != "" {
		fmt.Fprintf(&b, " for pool %q/%q generation %d", e.Pool.Driver, e.Pool.Name, e.Pool.Generation)
	}
	if e.Request != "" {
		fmt.Fprintf(&b, ", request %q", e.Request)
	}
	if e.Extractor != "" {
		fmt.Fprintf(&b, ", extractor %q", e.Extractor)
	}
	if e.ConfigIndex >= 0 {
		fmt.Fprintf(&b, ", config %d", e.ConfigIndex)
	}
	if e.Err != nil {
		fmt.Fprintf(&b, ": %v", e.Err)
	}
	return b.String()
}

func (e *Error) Unwrap() error {
	return e.Err
}

func (e *Error) Is(target error) bool {
	return e != nil && errors.Is(e.Kind, target)
}

// Outcome classifies errors for callers that handle pool and claim faults
// separately from fatal extraction failures.
type Outcome int

const (
	OutcomeClaimInconsistent Outcome = iota
	OutcomeInvalidDefinition
	OutcomeRuntime
)

// Classify classifies a non-nil extraction error. Definition errors take
// precedence over their underlying cost errors.
func Classify(err error) Outcome {
	switch {
	case errors.Is(err, ErrInvalidDefinition):
		return OutcomeInvalidDefinition
	case errors.Is(err, ErrClaimSelfInconsistent):
		return OutcomeClaimInconsistent
	default:
		return OutcomeRuntime
	}
}

func ctxError(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		if cause := context.Cause(ctx); cause != nil && cause != err {
			return fmt.Errorf("sharing affinity extraction: %w: %w", err, cause)
		}
		return fmt.Errorf("sharing affinity extraction: %w", err)
	}
	return nil
}

// Decoder and CEL errors can contain private values, even for missing keys.
// Preserve their identity without printing the underlying message.
type redactedError struct {
	summary string
	cause   error
}

func (e *redactedError) Error() string { return e.summary }
func (e *redactedError) Unwrap() error { return e.cause }

func redactParameters(err error) error {
	summary := "opaque parameters must decode to a JSON object"
	var typeError *json.UnmarshalTypeError
	if syntax, offset := sigsjson.SyntaxErrorOffset(err); syntax {
		summary = fmt.Sprintf("invalid JSON syntax at offset %d", offset)
	} else if errors.As(err, &typeError) {
		summary = fmt.Sprintf("cannot decode %s at offset %d", typeError.Type, typeError.Offset)
	}
	return &redactedError{summary: summary, cause: err}
}

func classifyEvalError(ctx context.Context, base *Error, err error) error {
	if cancelled := ctxError(ctx); cancelled != nil {
		base.Kind, base.Err = ErrCELEval, cancelled
		return base
	}
	var cancelled interpreter.EvalCancelledError
	switch {
	case errors.As(err, &cancelled) && cancelled.Cause == interpreter.CostLimitExceeded:
		base.Kind = ErrCostExceeded
	case errors.Is(err, errNonStringResult):
		base.Kind = ErrNonStringReturn
	case strings.Contains(err.Error(), "no such key:"):
		base.Kind = ErrMissingField
	default:
		base.Kind = ErrCELEval
	}
	base.Err = &redactedError{summary: base.Kind.Error(), cause: err}
	return base
}
