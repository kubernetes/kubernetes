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

package secretmanager_test

import (
	"context"

	secretmanager "cloud.google.com/go/secretmanager/apiv1beta1"
)

func ExampleClient_IAM() {
	ctx := context.Background()
	c, err := secretmanager.NewClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}

	// TODO: fill in secret resource path
	secret := "projects/[PROJECT_ID]/secrets/[SECRET]"
	handle := c.IAM(secret)

	policy, err := handle.Policy(ctx)
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use policy.
	_ = policy
}
