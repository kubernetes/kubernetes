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
	"fmt"
)

type OperatorType int

const (
	UniqueOperator OperatorType = iota
	LikeOperator
	ClusterOperator
	GroupByOperator
	UnlikeOperator
)

var (
	labels = []string{
		"UNIQUE",
		"LIKE",
		"CLUSTER",
		"GROUP_BY",
		"UNLIKE",
	}

	labelToType map[string]OperatorType
)

func init() {
	labelToType = make(map[string]OperatorType)
	for i, s := range labels {
		labelToType[s] = OperatorType(i)
	}
}

func (t OperatorType) String() string {
	switch t {
	case UniqueOperator, LikeOperator, ClusterOperator, GroupByOperator, UnlikeOperator:
		return labels[int(t)]
	default:
		panic(fmt.Sprintf("unrecognized operator type: %d", int(t)))
	}
}

func parseOperatorType(s string) (OperatorType, error) {
	t, found := labelToType[s]
	if !found {
		return UniqueOperator, fmt.Errorf("unrecognized operator %q", s)
	}
	return t, nil
}

type Constraint struct {
	Field    string       // required
	Operator OperatorType // required
	Value    string       // optional
}

func (c *Constraint) MarshalJSON() ([]byte, error) {
	var a []string
	if c != nil {
		if c.Value != "" {
			a = append(a, c.Field, c.Operator.String(), c.Value)
		} else {
			a = append(a, c.Field, c.Operator.String())
		}
	}
	return json.Marshal(a)
}

func (c *Constraint) UnmarshalJSON(buf []byte) (err error) {
	var a []string
	if err = json.Unmarshal(buf, &a); err != nil {
		return err
	}
	switch x := len(a); {
	case x < 2:
		err = fmt.Errorf("not enough arguments to form constraint")
	case x > 3:
		err = fmt.Errorf("too many arguments to form constraint")
	case x == 3:
		c.Value = a[2]
		fallthrough
	case x == 2:
		c.Field = a[0]
		c.Operator, err = parseOperatorType(a[1])
	}
	return err
}
