/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Binary client is an example client.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"time"

	"google.golang.org/grpc"
	pb "google.golang.org/grpc/examples/features/proto/echo"
	"google.golang.org/grpc/metadata"
)

var addr = flag.String("addr", "localhost:50051", "the address to connect to")

const (
	timestampFormat = time.StampNano // "Jan _2 15:04:05.000"
	streamingCount  = 10
)

func unaryCallWithMetadata(c pb.EchoClient, message string) {
	fmt.Printf("--- unary ---\n")
	// Create metadata and context.
	md := metadata.Pairs("timestamp", time.Now().Format(timestampFormat))
	ctx := metadata.NewOutgoingContext(context.Background(), md)

	// Make RPC using the context with the metadata.
	var header, trailer metadata.MD
	r, err := c.UnaryEcho(ctx, &pb.EchoRequest{Message: message}, grpc.Header(&header), grpc.Trailer(&trailer))
	if err != nil {
		log.Fatalf("failed to call UnaryEcho: %v", err)
	}

	if t, ok := header["timestamp"]; ok {
		fmt.Printf("timestamp from header:\n")
		for i, e := range t {
			fmt.Printf(" %d. %s\n", i, e)
		}
	} else {
		log.Fatal("timestamp expected but doesn't exist in header")
	}
	if l, ok := header["location"]; ok {
		fmt.Printf("location from header:\n")
		for i, e := range l {
			fmt.Printf(" %d. %s\n", i, e)
		}
	} else {
		log.Fatal("location expected but doesn't exist in header")
	}
	fmt.Printf("response:\n")
	fmt.Printf(" - %s\n", r.Message)

	if t, ok := trailer["timestamp"]; ok {
		fmt.Printf("timestamp from trailer:\n")
		for i, e := range t {
			fmt.Printf(" %d. %s\n", i, e)
		}
	} else {
		log.Fatal("timestamp expected but doesn't exist in trailer")
	}
}

func serverStreamingWithMetadata(c pb.EchoClient, message string) {
	fmt.Printf("--- server streaming ---\n")
	// Create metadata and context.
	md := metadata.Pairs("timestamp", time.Now().Format(timestampFormat))
	ctx := metadata.NewOutgoingContext(context.Background(), md)

	// Make RPC using the context with the metadata.
	stream, err := c.ServerStreamingEcho(ctx, &pb.EchoRequest{Message: message})
	if err != nil {
		log.Fatalf("failed to call ServerStreamingEcho: %v", err)
	}

	// Read the header when the header arrives.
	header, err := stream.Header()
	if err != nil {
		log.Fatalf("failed to get header from stream: %v", err)
	}
	// Read metadata from server's header.
	if t, ok := header["timestamp"]; ok {
		fmt.Printf("timestamp from header:\n")
		for i, e := range t {
			fmt.Printf(" %d. %s\n", i, e)
		}
	} else {
		log.Fatal("timestamp expected but doesn't exist in header")
	}
	if l, ok := header["location"]; ok {
		fmt.Printf("location from header:\n")
		for i, e := range l {
			fmt.Printf(" %d. %s\n", i, e)
		}
	} else {
		log.Fatal("location expected but doesn't exist in header")
	}

	// Read all the responses.
	var rpcStatus error
	fmt.Printf("response:\n")
	for {
		r, err := stream.Recv()
		if err != nil {
			rpcStatus = err
			break
		}
		fmt.Printf(" - %s\n", r.Message)
	}
	if rpcStatus != io.EOF {
		log.Fatalf("failed to finish server streaming: %v", rpcStatus)
	}

	// Read the trailer after the RPC is finished.
	trailer := stream.Trailer()
	// Read metadata from server's trailer.
	if t, ok := trailer["timestamp"]; ok {
		fmt.Printf("timestamp from trailer:\n")
		for i, e := range t {
			fmt.Printf(" %d. %s\n", i, e)
		}
	} else {
		log.Fatal("timestamp expected but doesn't exist in trailer")
	}
}

func clientStreamWithMetadata(c pb.EchoClient, message string) {
	fmt.Printf("--- client streaming ---\n")
	// Create metadata and context.
	md := metadata.Pairs("timestamp", time.Now().Format(timestampFormat))
	ctx := metadata.NewOutgoingContext(context.Background(), md)

	// Make RPC using the context with the metadata.
	stream, err := c.ClientStreamingEcho(ctx)
	if err != nil {
		log.Fatalf("failed to call ClientStreamingEcho: %v\n", err)
	}

	// Read the header when the header arrives.
	header, err := stream.Header()
	if err != nil {
		log.Fatalf("failed to get header from stream: %v", err)
	}
	// Read metadata from server's header.
	if t, ok := header["timestamp"]; ok {
		fmt.Printf("timestamp from header:\n")
		for i, e := range t {
			fmt.Printf(" %d. %s\n", i, e)
		}
	} else {
		log.Fatal("timestamp expected but doesn't exist in header")
	}
	if l, ok := header["location"]; ok {
		fmt.Printf("location from header:\n")
		for i, e := range l {
			fmt.Printf(" %d. %s\n", i, e)
		}
	} else {
		log.Fatal("location expected but doesn't exist in header")
	}

	// Send all requests to the server.
	for i := 0; i < streamingCount; i++ {
		if err := stream.Send(&pb.EchoRequest{Message: message}); err != nil {
			log.Fatalf("failed to send streaming: %v\n", err)
		}
	}

	// Read the response.
	r, err := stream.CloseAndRecv()
	if err != nil {
		log.Fatalf("failed to CloseAndRecv: %v\n", err)
	}
	fmt.Printf("response:\n")
	fmt.Printf(" - %s\n\n", r.Message)

	// Read the trailer after the RPC is finished.
	trailer := stream.Trailer()
	// Read metadata from server's trailer.
	if t, ok := trailer["timestamp"]; ok {
		fmt.Printf("timestamp from trailer:\n")
		for i, e := range t {
			fmt.Printf(" %d. %s\n", i, e)
		}
	} else {
		log.Fatal("timestamp expected but doesn't exist in trailer")
	}
}

func bidirectionalWithMetadata(c pb.EchoClient, message string) {
	fmt.Printf("--- bidirectional ---\n")
	// Create metadata and context.
	md := metadata.Pairs("timestamp", time.Now().Format(timestampFormat))
	ctx := metadata.NewOutgoingContext(context.Background(), md)

	// Make RPC using the context with the metadata.
	stream, err := c.BidirectionalStreamingEcho(ctx)
	if err != nil {
		log.Fatalf("failed to call BidirectionalStreamingEcho: %v\n", err)
	}

	go func() {
		// Read the header when the header arrives.
		header, err := stream.Header()
		if err != nil {
			log.Fatalf("failed to get header from stream: %v", err)
		}
		// Read metadata from server's header.
		if t, ok := header["timestamp"]; ok {
			fmt.Printf("timestamp from header:\n")
			for i, e := range t {
				fmt.Printf(" %d. %s\n", i, e)
			}
		} else {
			log.Fatal("timestamp expected but doesn't exist in header")
		}
		if l, ok := header["location"]; ok {
			fmt.Printf("location from header:\n")
			for i, e := range l {
				fmt.Printf(" %d. %s\n", i, e)
			}
		} else {
			log.Fatal("location expected but doesn't exist in header")
		}

		// Send all requests to the server.
		for i := 0; i < streamingCount; i++ {
			if err := stream.Send(&pb.EchoRequest{Message: message}); err != nil {
				log.Fatalf("failed to send streaming: %v\n", err)
			}
		}
		stream.CloseSend()
	}()

	// Read all the responses.
	var rpcStatus error
	fmt.Printf("response:\n")
	for {
		r, err := stream.Recv()
		if err != nil {
			rpcStatus = err
			break
		}
		fmt.Printf(" - %s\n", r.Message)
	}
	if rpcStatus != io.EOF {
		log.Fatalf("failed to finish server streaming: %v", rpcStatus)
	}

	// Read the trailer after the RPC is finished.
	trailer := stream.Trailer()
	// Read metadata from server's trailer.
	if t, ok := trailer["timestamp"]; ok {
		fmt.Printf("timestamp from trailer:\n")
		for i, e := range t {
			fmt.Printf(" %d. %s\n", i, e)
		}
	} else {
		log.Fatal("timestamp expected but doesn't exist in trailer")
	}

}

const message = "this is examples/metadata"

func main() {
	flag.Parse()
	// Set up a connection to the server.
	conn, err := grpc.Dial(*addr, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()

	c := pb.NewEchoClient(conn)

	unaryCallWithMetadata(c, message)
	time.Sleep(1 * time.Second)

	serverStreamingWithMetadata(c, message)
	time.Sleep(1 * time.Second)

	clientStreamWithMetadata(c, message)
	time.Sleep(1 * time.Second)

	bidirectionalWithMetadata(c, message)
}
