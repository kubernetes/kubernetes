// Copyright 2016 Google Inc. All Rights Reserved.
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

package bytestream

import (
	"bytes"
	"fmt"
	"io"
	"log"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

func ExampleNewClient(serverPort int, resourceName string) {
	ctx := context.Background()
	conn, err := grpc.Dial(fmt.Sprintf("localhost:%d", serverPort), grpc.WithInsecure())
	if err != nil {
		log.Printf("grpc.Dial: %v", err)
		return
	}
	client := NewClient(conn)
	reader, err := client.NewReader(ctx, resourceName)
	if err != nil {
		log.Printf("NewReader(%q): %v", resourceName, err)
	}
	var buf bytes.Buffer
	n, err := buf.ReadFrom(reader)
	if err != nil && err != io.EOF {
		log.Printf("Read %d bytes, got err=%v", n, err)
	}
	log.Printf("read %q", string(buf.Bytes()))
}

func ExampleNewReader(serverPort int, resourceName string) {
	ctx := context.Background()
	conn, err := grpc.Dial(fmt.Sprintf("localhost:%d", serverPort), grpc.WithInsecure())
	if err != nil {
		log.Printf("grpc.Dial: %v", err)
		return
	}
	client := NewClient(conn)
	reader, err := client.NewReader(ctx, resourceName)
	if err != nil {
		log.Printf("NewReader(%q): %v", resourceName, err)
	}
	var buf bytes.Buffer
	n, err := buf.ReadFrom(reader)
	if err != nil && err != io.EOF {
		log.Printf("Read %d bytes, got err=%v", n, err)
	}
	log.Printf("read %q", string(buf.Bytes()))
}

func ExampleNewWriter(serverPort int, resourceName string) {
	ctx := context.Background()
	conn, err := grpc.Dial(fmt.Sprintf("localhost:%d", serverPort), grpc.WithInsecure())
	if err != nil {
		log.Printf("grpc.Dial: %v", err)
		return
	}
	client := NewClient(conn)

	w, err := client.NewWriter(ctx, resourceName)
	if err != nil {
		log.Printf("NewWriter: %v", err)
		return
	}
	defer func() {
		err := w.Close()
		if err != nil {
			log.Printf("Close: %v", err)
		}
	}()

	buf := []byte("hello world")
	n, err := w.Write(buf)
	if err != nil {
		log.Printf("Write: %v", err)
	}
	log.Printf("Wrote %d bytes", n)
}
