/*
Copyright 2017 The Kubernetes Authors.

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

package schedulingpolicy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
)

// policyUndefinedError represents an undefined response from the policy
// engine. This typically means the relevant policy has not been loaded into
// the engine.
type policyUndefinedError struct{}

func (policyUndefinedError) Error() string {
	return "policy decision is undefined"
}

// policyEngineQuery represents a single query against the policy engine.
type policyEngineQuery struct {
	client       *rest.RESTClient
	retryBackoff time.Duration
	obj          runtime.Object
	gvk          schema.GroupVersionKind
}

// newPolicyEngineQuery returns a policyEngineQuery that can be executed.
func newPolicyEngineQuery(client *rest.RESTClient, retryBackoff time.Duration, obj runtime.Object, gvk schema.GroupVersionKind) *policyEngineQuery {
	return &policyEngineQuery{
		client:       client,
		retryBackoff: retryBackoff,
		obj:          obj,
		gvk:          gvk,
	}
}

// Do returns the result of the policy engine query. If the policy decision is
// undefined or an unknown error occurs, err is non-nil. Otherwise, result is
// non-nil and contains the result of policy evaluation.
func (query *policyEngineQuery) Do() (decision *policyDecision, err error) {

	bs, err := query.encode()
	if err != nil {
		return nil, err
	}

	var result rest.Result

	err = webhook.WithExponentialBackoff(query.retryBackoff, func() error {
		result = query.client.Post().
			Body(bs).
			Do()
		return result.Error()
	})

	if err != nil {
		if errors.IsNotFound(err) {
			return nil, policyUndefinedError{}
		}
		return nil, err
	}

	return decodeResult(result)
}

// encode returns the encoded version of the query's runtime.Object.
func (query *policyEngineQuery) encode() ([]byte, error) {

	var info runtime.SerializerInfo
	infos := api.Codecs.SupportedMediaTypes()

	for i := range infos {
		if infos[i].MediaType == "application/json" {
			info = infos[i]
		}
	}

	if info.Serializer == nil {
		return nil, fmt.Errorf("serialization not supported")
	}

	codec := api.Codecs.EncoderForVersion(info.Serializer, query.gvk.GroupVersion())

	var buf bytes.Buffer
	if err := codec.Encode(query.obj, &buf); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

// policyDecision represents a response from the policy engine.
type policyDecision struct {
	Errors      []string          `json:"errors,omitempty"`
	Annotations map[string]string `json:"annotations,omitempty"`
}

// Error returns an error if the policy raised an error.
func (d *policyDecision) Error() error {
	if len(d.Errors) == 0 {
		return nil
	}
	return fmt.Errorf("reason(s): %v", strings.Join(d.Errors, "; "))
}

func decodeResult(result rest.Result) (*policyDecision, error) {

	bs, err := result.Raw()
	if err != nil {
		return nil, err
	}

	buf := bytes.NewBuffer(bs)
	var decision policyDecision

	if err := json.NewDecoder(buf).Decode(&decision); err != nil {
		return nil, err
	}

	return &decision, nil
}
