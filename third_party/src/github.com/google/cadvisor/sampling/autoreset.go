// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package sampling

import "time"

type autoResetSampler struct {
	shouldReset func(d interface{}) bool
	sampler     Sampler
}

func (self *autoResetSampler) Len() int {
	return self.sampler.Len()
}

func (self *autoResetSampler) Reset() {
	self.sampler.Reset()
}

func (self *autoResetSampler) Map(f func(d interface{})) {
	self.sampler.Map(f)
}

func (self *autoResetSampler) Filter(filter func(d interface{}) bool) {
	self.sampler.Filter(filter)
}

func (self *autoResetSampler) Update(d interface{}) {
	if self.shouldReset(d) {
		self.sampler.Reset()
	}
	self.sampler.Update(d)
}

func NewPeriodicallyResetSampler(period time.Duration, sampler Sampler) Sampler {
	lastRest := time.Now()
	shouldReset := func(d interface{}) bool {
		if time.Now().Sub(lastRest) > period {
			lastRest = time.Now()
			return true
		}
		return false
	}
	return &autoResetSampler{
		shouldReset: shouldReset,
		sampler:     sampler,
	}
}
