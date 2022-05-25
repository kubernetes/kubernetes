/*
Copyright 2022 The Kubernetes Authors.

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

package semanticrv

// Comparison is the result of comparing two values in a partial order.
type Comparison struct {
	LE bool // less than or equal
	GE bool // greater than or equal
}

func Equal() Comparison { return Comparison{LE: true, GE: true} }

func Less() Comparison { return Comparison{LE: true, GE: false} }

func Greater() Comparison { return Comparison{LE: false, GE: true} }

func Incomparable() Comparison { return Comparison{LE: false, GE: false} }

func (c Comparison) IsLess() bool { return c.LE && !c.GE }

func (c Comparison) IsLessOrEqual() bool { return c.LE }

func (c Comparison) IsGreater() bool { return c.GE && !c.LE }

func (c Comparison) IsGreaterOrEqual() bool { return c.GE }

func (c Comparison) IsEqual() bool { return c.LE && c.GE }

func (c Comparison) IsComparable() bool { return c.LE || c.GE }

func (c Comparison) IsStrictlyOrdered() bool { return c.LE != c.GE }

func (c Comparison) And(d Comparison) Comparison {
	return Comparison{LE: c.LE && d.LE, GE: c.GE && d.GE}
}

func (c Comparison) Implies(d Comparison) bool {
	return (!c.LE || d.LE) && (!c.GE || d.GE)
}

func (c Comparison) Reverse() Comparison {
	return Comparison{LE: c.GE, GE: c.LE}
}
