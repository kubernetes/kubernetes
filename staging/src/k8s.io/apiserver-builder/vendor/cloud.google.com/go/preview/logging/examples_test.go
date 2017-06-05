// Copyright 2016 Google Inc. All Rights Reserved.
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

package logging_test

import (
	"fmt"
	"os"

	"cloud.google.com/go/preview/logging"
	"golang.org/x/net/context"
)

func ExampleNewClient() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: handle error.
	}
	// Use client to manage logs, metrics and sinks.
	// Close the client when finished.
	if err := client.Close(); err != nil {
		// TODO: handle error.
	}
}

func ExampleClient_Ping() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: handle error.
	}
	if err := client.Ping(ctx); err != nil {
		// TODO: handle error.
	}
}

func ExampleNewClient_errorFunc() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	// Print all errors to stdout.
	client.OnError = func(e error) {
		fmt.Fprintf(os.Stdout, "logging: %v", e)
	}
	// Use client to manage logs, metrics and sinks.
	// Close the client when finished.
	if err := client.Close(); err != nil {
		// TODO: Handle error.
	}
}

func ExampleClient_DeleteLog() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	err = client.DeleteLog(ctx, "my-log")
	if err != nil {
		// TODO: Handle error.
	}
}

func ExampleClient_Logger() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	lg := client.Logger("my-log")
	_ = lg // TODO: use the Logger.
}

func ExampleLogger_LogSync() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	lg := client.Logger("my-log")
	err = lg.LogSync(ctx, logging.Entry{Payload: "red alert"})
	if err != nil {
		// TODO: Handle error.
	}
}

func ExampleLogger_Log() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	lg := client.Logger("my-log")
	lg.Log(logging.Entry{Payload: "something happened"})
}

func ExampleLogger_Flush() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	lg := client.Logger("my-log")
	lg.Log(logging.Entry{Payload: "something happened"})
	lg.Flush()
}

func ExampleLogger_StandardLogger() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	lg := client.Logger("my-log")
	slg := lg.StandardLogger(logging.Info)
	slg.Println("an informative message")
}

func ExampleClient_CreateMetric() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	err = client.CreateMetric(ctx, &logging.Metric{
		ID:          "severe-errors",
		Description: "entries at ERROR or higher severities",
		Filter:      "severity >= ERROR",
	})
	if err != nil {
		// TODO: Handle error.
	}
}

func ExampleClient_DeleteMetric() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	if err := client.DeleteMetric(ctx, "severe-errors"); err != nil {
		// TODO: Handle error.
	}
}

func ExampleClient_Metric() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	m, err := client.Metric(ctx, "severe-errors")
	if err != nil {
		// TODO: Handle error.
	}
	fmt.Println(m)
}

func ExampleClient_UpdateMetric() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	err = client.UpdateMetric(ctx, &logging.Metric{
		ID:          "severe-errors",
		Description: "entries at high severities",
		Filter:      "severity > ERROR",
	})
	if err != nil {
		// TODO: Handle error.
	}
}

func ExampleClient_CreateSink() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	sink, err := client.CreateSink(ctx, &logging.Sink{
		ID:          "severe-errors-to-gcs",
		Destination: "storage.googleapis.com/my-bucket",
		Filter:      "severity >= ERROR",
	})
	if err != nil {
		// TODO: Handle error.
	}
	fmt.Println(sink)
}

func ExampleClient_DeleteSink() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	if err := client.DeleteSink(ctx, "severe-errors-to-gcs"); err != nil {
		// TODO: Handle error.
	}
}

func ExampleClient_Sink() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	s, err := client.Sink(ctx, "severe-errors-to-gcs")
	if err != nil {
		// TODO: Handle error.
	}
	fmt.Println(s)
}

func ExampleClient_UpdateSink() {
	ctx := context.Background()
	client, err := logging.NewClient(ctx, "my-project")
	if err != nil {
		// TODO: Handle error.
	}
	sink, err := client.UpdateSink(ctx, &logging.Sink{
		ID:          "severe-errors-to-gcs",
		Destination: "storage.googleapis.com/my-other-bucket",
		Filter:      "severity >= ERROR",
	})
	if err != nil {
		// TODO: Handle error.
	}
	fmt.Println(sink)
}

func ExampleParseSeverity() {
	sev := logging.ParseSeverity("ALERT")
	fmt.Println(sev)
	// Output: Alert
}
