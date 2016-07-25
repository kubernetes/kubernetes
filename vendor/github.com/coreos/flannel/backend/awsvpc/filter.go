// Copyright 2015 flannel authors
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

package awsvpc

import "github.com/aws/aws-sdk-go/service/ec2"

type ecFilter []*ec2.Filter

func (f *ecFilter) Add(key, value string) {
	for _, fltr := range *f {
		if fltr.Name != nil && *fltr.Name == key {
			fltr.Values = append(fltr.Values, &value)
			return
		}
	}

	newFilter := &ec2.Filter{
		Name:   &key,
		Values: []*string{&value},
	}

	*f = append(*f, newFilter)
}

func newFilter() ecFilter {
	return make(ecFilter, 0)
}
