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

package storage

import (
	"math/rand"
)

type ChoiceWeight[T any] struct {
	Choice T
	Weight int
}

func PickRandom[T any](choices []ChoiceWeight[T]) T {
	sum := 0
	for _, op := range choices {
		sum += op.Weight
	}
	roll := rand.Int() % sum
	for _, op := range choices {
		if roll < op.Weight {
			return op.Choice
		}
		roll -= op.Weight
	}
	panic("unexpected")
}
