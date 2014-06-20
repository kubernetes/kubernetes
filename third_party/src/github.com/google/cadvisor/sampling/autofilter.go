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

type autoFilterSampler struct {
	// filter will run to remove elements before adding every observation
	filter  func(d interface{}) bool
	sampler Sampler
}

func (self *autoFilterSampler) Len() int {
	return self.sampler.Len()
}

func (self *autoFilterSampler) Reset() {
	self.sampler.Reset()
}

func (self *autoFilterSampler) Map(f func(d interface{})) {
	self.sampler.Map(f)
}

func (self *autoFilterSampler) Filter(filter func(d interface{}) bool) {
	self.sampler.Filter(filter)
}

func (self *autoFilterSampler) Update(d interface{}) {
	self.Filter(self.filter)
	self.sampler.Update(d)
}

// Add a decorator for sampler. Whenever an Update() is called, the sampler will
// call filter() first to remove elements in the decorated sampler.
func NewAutoFilterSampler(sampler Sampler, filter func(d interface{}) bool) Sampler {
	return &autoFilterSampler{
		filter:  filter,
		sampler: sampler,
	}
}
