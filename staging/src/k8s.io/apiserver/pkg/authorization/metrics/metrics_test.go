/*
Copyright 2024 The Kubernetes Authors.

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

package metrics

import (
	"context"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestRecordAuthorizationDecisionsTotal(t *testing.T) {
	prefix := `
    # HELP apiserver_authorization_decisions_total [ALPHA] Total number of terminal decisions made by an authorizer split by authorizer type, name, and decision.
    # TYPE apiserver_authorization_decisions_total counter`
	metrics := []string{
		namespace + "_" + subsystem + "_decisions_total",
	}

	authorizationDecisionsTotal.Reset()
	RegisterMetrics()

	dummyAuthorizer := &dummyAuthorizer{}
	a := InstrumentedAuthorizer("mytype", "myname", dummyAuthorizer)

	// allow
	{
		dummyAuthorizer.decision = authorizer.DecisionAllow
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="allowed",name="myname",type="mytype"} 1
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}

	// deny
	{
		dummyAuthorizer.decision = authorizer.DecisionDeny
		_, _, _ = a.Authorize(context.Background(), nil)
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="denied",name="myname",type="mytype"} 2
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}

	// no-opinion emits no metric
	{
		dummyAuthorizer.decision = authorizer.DecisionNoOpinion
		_, _, _ = a.Authorize(context.Background(), nil)
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}

	// unknown decision emits a metric
	{
		dummyAuthorizer.decision = authorizer.DecisionDeny + 10
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="unknown",name="myname",type="mytype"} 1
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}

}

type dummyAuthorizer struct {
	decision authorizer.Decision
	err      error
}

func (d *dummyAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	return d.decision, "", d.err
}
