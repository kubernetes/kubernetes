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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/user"
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
	user         user.Info
	obj          runtime.Object
	gvk          schema.GroupVersionKind
}

// newPolicyEngineQuery returns a policyEngineQuery that can be executed.
func newPolicyEngineQuery(client *rest.RESTClient, retryBackoff time.Duration, user user.Info, obj runtime.Object, gvk schema.GroupVersionKind) *policyEngineQuery {
	return &policyEngineQuery{
		client:       client,
		retryBackoff: retryBackoff,
		user:         user,
		obj:          obj,
		gvk:          gvk,
	}
}

// Do returns the result of the policy engine query. If the policy decision is
// undefined or an unknown error occurs, err is non-nil. Othewrise, result is
// non-nil and contains the result of policy evaluation.
func (query *policyEngineQuery) Do() (decision *policyDecision, err error) {

	raw, err := convertObject(query.obj, query.gvk.GroupVersion())
	if err != nil {
		return nil, err
	}

	body := policyEngineInput{}
	body.Input.Resource = raw
	body.Input.User.Name = query.user.GetName()
	body.Input.User.Groups = query.user.GetGroups()
	body.Input.User.UID = query.user.GetUID()
	body.Input.User.Extra = query.user.GetExtra()

	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(&body); err != nil {
		return nil, err
	}

	var result rest.Result

	err = webhook.WithExponentialBackoff(query.retryBackoff, func() error {
		result = query.client.Post().
			SetHeader("Content-Type", "application/json").
			Body(&buf).
			Do()
		return result.Error()
	})

	if err != nil {
		return nil, err
	}

	return decodeResult(result)
}

// TODO(tsandall): review the request type as well as SerDes

// policyEngineInput represents the input data provided to the policy engine when
// executing a policy query.
type policyEngineInput struct {
	Input struct {
		Resource interface{} `json:"resource"`
		User     struct {
			Name   string              `json:"name"`
			UID    string              `json:"uid"`
			Groups []string            `json:"groups"`
			Extra  map[string][]string `json:"extra"`
		} `json:"user"`
	} `json:"input"`
}

// policyEngineResponse represents the policy engine's response to a policy
// query.
type policyEngineResponse struct {
	Result *policyDecision `json:"result,omitempty"`
}

// policyDecision represents a response from the policy engine.
type policyDecision struct {
	Errors   []string `json:"errors,omitempty"`
	Resource struct {
		Metadata struct {
			Annotations map[string]string `json:"annotations,omitempty"`
		} `json:"metadata,omitempty"`
	} `json:"resource,omitempty"`
}

// Error returns an error if the policy raised an error.
func (d *policyDecision) Error() error {
	if len(d.Errors) == 0 {
		return nil
	}
	return fmt.Errorf("reason(s): %v", strings.Join(d.Errors, "; "))
}

func convertObject(obj runtime.Object, gv schema.GroupVersion) (interface{}, error) {

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

	codec := api.Codecs.EncoderForVersion(info.Serializer, gv)
	var buf bytes.Buffer
	if err := codec.Encode(obj, &buf); err != nil {
		return nil, err
	}

	var result interface{}
	if err := json.NewDecoder(&buf).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

func decodeResult(result rest.Result) (*policyDecision, error) {

	bs, err := result.Raw()
	if err != nil {
		return nil, err
	}

	buf := bytes.NewBuffer(bs)
	var response policyEngineResponse

	if err := json.NewDecoder(buf).Decode(&response); err != nil {
		return nil, err
	}

	if response.Result == nil {
		return nil, policyUndefinedError{}
	}

	return response.Result, nil
}
