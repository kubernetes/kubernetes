//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package pollers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/exported"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/log"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/shared"
)

// the well-known set of LRO status/provisioning state values.
const (
	StatusSucceeded  = "Succeeded"
	StatusCanceled   = "Canceled"
	StatusFailed     = "Failed"
	StatusInProgress = "InProgress"
)

// IsTerminalState returns true if the LRO's state is terminal.
func IsTerminalState(s string) bool {
	return strings.EqualFold(s, StatusSucceeded) || strings.EqualFold(s, StatusFailed) || strings.EqualFold(s, StatusCanceled)
}

// Failed returns true if the LRO's state is terminal failure.
func Failed(s string) bool {
	return strings.EqualFold(s, StatusFailed) || strings.EqualFold(s, StatusCanceled)
}

// Succeeded returns true if the LRO's state is terminal success.
func Succeeded(s string) bool {
	return strings.EqualFold(s, StatusSucceeded)
}

// returns true if the LRO response contains a valid HTTP status code
func StatusCodeValid(resp *http.Response) bool {
	return exported.HasStatusCode(resp, http.StatusOK, http.StatusAccepted, http.StatusCreated, http.StatusNoContent)
}

// IsValidURL verifies that the URL is valid and absolute.
func IsValidURL(s string) bool {
	u, err := url.Parse(s)
	return err == nil && u.IsAbs()
}

// getTokenTypeName creates a type name from the type parameter T.
func getTokenTypeName[T any]() (string, error) {
	tt := shared.TypeOfT[T]()
	var n string
	if tt.Kind() == reflect.Pointer {
		n = "*"
		tt = tt.Elem()
	}
	n += tt.Name()
	if n == "" {
		return "", errors.New("nameless types are not allowed")
	}
	return n, nil
}

type resumeTokenWrapper[T any] struct {
	Type  string `json:"type"`
	Token T      `json:"token"`
}

// NewResumeToken creates a resume token from the specified type.
// An error is returned if the generic type has no name (e.g. struct{}).
func NewResumeToken[TResult, TSource any](from TSource) (string, error) {
	n, err := getTokenTypeName[TResult]()
	if err != nil {
		return "", err
	}
	b, err := json.Marshal(resumeTokenWrapper[TSource]{
		Type:  n,
		Token: from,
	})
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ExtractToken returns the poller-specific token information from the provided token value.
func ExtractToken(token string) ([]byte, error) {
	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal([]byte(token), &raw); err != nil {
		return nil, err
	}
	// this is dependent on the type resumeTokenWrapper[T]
	tk, ok := raw["token"]
	if !ok {
		return nil, errors.New("missing token value")
	}
	return tk, nil
}

// IsTokenValid returns an error if the specified token isn't applicable for generic type T.
func IsTokenValid[T any](token string) error {
	raw := map[string]interface{}{}
	if err := json.Unmarshal([]byte(token), &raw); err != nil {
		return err
	}
	t, ok := raw["type"]
	if !ok {
		return errors.New("missing type value")
	}
	tt, ok := t.(string)
	if !ok {
		return fmt.Errorf("invalid type format %T", t)
	}
	n, err := getTokenTypeName[T]()
	if err != nil {
		return err
	}
	if tt != n {
		return fmt.Errorf("cannot resume from this poller token. token is for type %s, not %s", tt, n)
	}
	return nil
}

// ErrNoBody is returned if the response didn't contain a body.
var ErrNoBody = errors.New("the response did not contain a body")

// GetJSON reads the response body into a raw JSON object.
// It returns ErrNoBody if there was no content.
func GetJSON(resp *http.Response) (map[string]interface{}, error) {
	body, err := exported.Payload(resp)
	if err != nil {
		return nil, err
	}
	if len(body) == 0 {
		return nil, ErrNoBody
	}
	// unmarshall the body to get the value
	var jsonBody map[string]interface{}
	if err = json.Unmarshal(body, &jsonBody); err != nil {
		return nil, err
	}
	return jsonBody, nil
}

// provisioningState returns the provisioning state from the response or the empty string.
func provisioningState(jsonBody map[string]interface{}) string {
	jsonProps, ok := jsonBody["properties"]
	if !ok {
		return ""
	}
	props, ok := jsonProps.(map[string]interface{})
	if !ok {
		return ""
	}
	rawPs, ok := props["provisioningState"]
	if !ok {
		return ""
	}
	ps, ok := rawPs.(string)
	if !ok {
		return ""
	}
	return ps
}

// status returns the status from the response or the empty string.
func status(jsonBody map[string]interface{}) string {
	rawStatus, ok := jsonBody["status"]
	if !ok {
		return ""
	}
	status, ok := rawStatus.(string)
	if !ok {
		return ""
	}
	return status
}

// GetStatus returns the LRO's status from the response body.
// Typically used for Azure-AsyncOperation flows.
// If there is no status in the response body the empty string is returned.
func GetStatus(resp *http.Response) (string, error) {
	jsonBody, err := GetJSON(resp)
	if err != nil {
		return "", err
	}
	return status(jsonBody), nil
}

// GetProvisioningState returns the LRO's state from the response body.
// If there is no state in the response body the empty string is returned.
func GetProvisioningState(resp *http.Response) (string, error) {
	jsonBody, err := GetJSON(resp)
	if err != nil {
		return "", err
	}
	return provisioningState(jsonBody), nil
}

// GetResourceLocation returns the LRO's resourceLocation value from the response body.
// Typically used for Operation-Location flows.
// If there is no resourceLocation in the response body the empty string is returned.
func GetResourceLocation(resp *http.Response) (string, error) {
	jsonBody, err := GetJSON(resp)
	if err != nil {
		return "", err
	}
	v, ok := jsonBody["resourceLocation"]
	if !ok {
		// it might be ok if the field doesn't exist, the caller must make that determination
		return "", nil
	}
	vv, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("the resourceLocation value %v was not in string format", v)
	}
	return vv, nil
}

// used if the operation synchronously completed
type NopPoller[T any] struct {
	resp   *http.Response
	result T
}

// NewNopPoller creates a NopPoller from the provided response.
// It unmarshals the response body into an instance of T.
func NewNopPoller[T any](resp *http.Response) (*NopPoller[T], error) {
	np := &NopPoller[T]{resp: resp}
	if resp.StatusCode == http.StatusNoContent {
		return np, nil
	}
	payload, err := exported.Payload(resp)
	if err != nil {
		return nil, err
	}
	if len(payload) == 0 {
		return np, nil
	}
	if err = json.Unmarshal(payload, &np.result); err != nil {
		return nil, err
	}
	return np, nil
}

func (*NopPoller[T]) Done() bool {
	return true
}

func (p *NopPoller[T]) Poll(context.Context) (*http.Response, error) {
	return p.resp, nil
}

func (p *NopPoller[T]) Result(ctx context.Context, out *T) error {
	*out = p.result
	return nil
}

// PollHelper creates and executes the request, calling update() with the response.
// If the request fails, the update func is not called.
// The update func returns the state of the operation for logging purposes or an error
// if it fails to extract the required state from the response.
func PollHelper(ctx context.Context, endpoint string, pl exported.Pipeline, update func(resp *http.Response) (string, error)) error {
	req, err := exported.NewRequest(ctx, http.MethodGet, endpoint)
	if err != nil {
		return err
	}
	resp, err := pl.Do(req)
	if err != nil {
		return err
	}
	state, err := update(resp)
	if err != nil {
		return err
	}
	log.Writef(log.EventLRO, "State %s", state)
	return nil
}

// ResultHelper processes the response as success or failure.
// In the success case, it unmarshals the payload into either a new instance of T or out.
// In the failure case, it creates an *azcore.Response error from the response.
func ResultHelper[T any](resp *http.Response, failed bool, out *T) error {
	// short-circuit the simple success case with no response body to unmarshal
	if resp.StatusCode == http.StatusNoContent {
		return nil
	}

	defer resp.Body.Close()
	if !StatusCodeValid(resp) || failed {
		// the LRO failed.  unmarshall the error and update state
		return exported.NewResponseError(resp)
	}

	// success case
	payload, err := exported.Payload(resp)
	if err != nil {
		return err
	}
	if len(payload) == 0 {
		return nil
	}

	if err = json.Unmarshal(payload, out); err != nil {
		return err
	}
	return nil
}
