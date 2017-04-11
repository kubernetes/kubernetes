package main

import (
	"flag"
	"fmt"
	"os"
	"testing"
	"time"

	server "github.com/grpc-ecosystem/grpc-gateway/examples/server"
)

func runServers() <-chan error {
	ch := make(chan error, 2)
	go func() {
		if err := server.Run(); err != nil {
			ch <- fmt.Errorf("cannot run grpc service: %v", err)
		}
	}()
	go func() {
		if err := Run(":8080"); err != nil {
			ch <- fmt.Errorf("cannot run gateway service: %v", err)
		}
	}()
	return ch
}

func TestMain(m *testing.M) {
	flag.Parse()
	errCh := runServers()

	ch := make(chan int, 1)
	go func() {
		time.Sleep(100 * time.Millisecond)
		ch <- m.Run()
	}()

	select {
	case err := <-errCh:
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	case status := <-ch:
		os.Exit(status)
	}
}
