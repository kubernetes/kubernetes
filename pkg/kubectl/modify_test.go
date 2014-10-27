/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubectl

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

type FakeRESTClient struct{}

func (c *FakeRESTClient) Get() *client.Request {
	return &client.Request{}
}
func (c *FakeRESTClient) Put() *client.Request {
	return &client.Request{}
}
func (c *FakeRESTClient) Post() *client.Request {
	return &client.Request{}
}
func (c *FakeRESTClient) Delete() *client.Request {
	return &client.Request{}
}

func TestRESTModifierDelete(t *testing.T) {
	tests := []struct {
		Err bool
	}{
	/*{
		Err: true,
	},*/
	}
	for _, test := range tests {
		client := &FakeRESTClient{}
		modifier := &RESTModifier{
			RESTClient: client,
		}
		err := modifier.Delete("bar", "foo")
		switch {
		case err == nil && test.Err:
			t.Errorf("Unexpected non-error")
			continue
		case err != nil && !test.Err:
			t.Errorf("Unexpected error: %v", err)
			continue
		}
	}
}
