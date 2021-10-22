// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kms_test

import (
	"context"

	kms "cloud.google.com/go/kms/apiv1"
)

func ExampleKeyManagementClient_ResourceIAM() {
	ctx := context.Background()
	c, err := kms.NewKeyManagementClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	// TODO: fill in key ring resource path
	keyRing := "projects/[PROJECT_ID]/locations/[LOCATION]/keyRings/[KEY_RING]"
	handle := c.ResourceIAM(keyRing)

	policy, err := handle.Policy(ctx)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use policy.
	_ = policy
}
