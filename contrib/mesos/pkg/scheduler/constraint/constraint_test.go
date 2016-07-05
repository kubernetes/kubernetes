/*
Copyright 2015 The Kubernetes Authors.

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

package constraint

import (
	"encoding/json"
	"testing"
)

func TestDeserialize(t *testing.T) {
	shouldMatch := func(js string, field string, operator OperatorType, value string) (err error) {
		constraint := Constraint{}
		if err = json.Unmarshal(([]byte)(js), &constraint); err != nil {
			return
		}
		if field != constraint.Field {
			t.Fatalf("expected field %q instead of %q", field, constraint.Field)
		}
		if operator != constraint.Operator {
			t.Fatalf("expected operator %v instead of %v", operator, constraint.Operator)
		}
		if value != constraint.Value {
			t.Fatalf("expected value %q instead of %q", value, constraint.Value)
		}
		return
	}
	failOnError := func(err error) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	failOnError(shouldMatch(`["hostname","UNIQUE"]`, "hostname", UniqueOperator, ""))
	failOnError(shouldMatch(`["rackid","GROUP_BY","1"]`, "rackid", GroupByOperator, "1"))
	failOnError(shouldMatch(`["jdk","LIKE","7"]`, "jdk", LikeOperator, "7"))
	failOnError(shouldMatch(`["jdk","UNLIKE","7"]`, "jdk", UnlikeOperator, "7"))
	failOnError(shouldMatch(`["bob","CLUSTER","foo"]`, "bob", ClusterOperator, "foo"))
	err := shouldMatch(`["bill","NOT_REALLY_AN_OPERATOR","pete"]`, "bill", ClusterOperator, "pete")
	if err == nil {
		t.Fatalf("expected unmarshalling error for invalid operator")
	}
}

func TestSerialize(t *testing.T) {
	shouldMatch := func(expected string, constraint *Constraint) error {
		data, err := json.Marshal(constraint)
		if err != nil {
			return err
		}
		js := string(data)
		if js != expected {
			t.Fatalf("expected json %q instead of %q", expected, js)
		}
		return nil
	}
	failOnError := func(err error) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	failOnError(shouldMatch(`["hostname","UNIQUE"]`, &Constraint{"hostname", UniqueOperator, ""}))
	failOnError(shouldMatch(`["rackid","GROUP_BY","1"]`, &Constraint{"rackid", GroupByOperator, "1"}))
	failOnError(shouldMatch(`["jdk","LIKE","7"]`, &Constraint{"jdk", LikeOperator, "7"}))
	failOnError(shouldMatch(`["jdk","UNLIKE","7"]`, &Constraint{"jdk", UnlikeOperator, "7"}))
	failOnError(shouldMatch(`["bob","CLUSTER","foo"]`, &Constraint{"bob", ClusterOperator, "foo"}))
}
