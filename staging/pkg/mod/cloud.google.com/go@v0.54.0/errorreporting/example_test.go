// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package errorreporting_test

import (
	"context"
	"errors"
	"log"

	"cloud.google.com/go/errorreporting"
)

func Example() {
	// Create the client.
	ctx := context.Background()
	ec, err := errorreporting.NewClient(ctx, "my-gcp-project", errorreporting.Config{
		ServiceName:    "myservice",
		ServiceVersion: "v1.0",
	})
	if err != nil {
		// TODO: handle error
	}
	defer func() {
		if err := ec.Close(); err != nil {
			log.Printf("failed to report errors to Stackdriver: %v", err)
		}
	}()

	// Report an error.
	err = doSomething()
	if err != nil {
		ec.Report(errorreporting.Entry{
			Error: err,
		})
	}
}

func doSomething() error {
	return errors.New("something went wrong")
}
