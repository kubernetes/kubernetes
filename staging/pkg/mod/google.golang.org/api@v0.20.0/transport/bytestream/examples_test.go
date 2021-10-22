// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytestream

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"

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
	log.Printf("read %q", buf.String())
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
	log.Printf("read %q", buf.String())
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
