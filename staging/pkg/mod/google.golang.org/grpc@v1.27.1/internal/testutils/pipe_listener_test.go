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

package testutils_test

import (
	"testing"
	"time"

	"google.golang.org/grpc/internal/testutils"
)

func TestPipeListener(t *testing.T) {
	pl := testutils.NewPipeListener()
	recvdBytes := make(chan []byte, 1)
	const want = "hello world"

	go func() {
		c, err := pl.Accept()
		if err != nil {
			t.Error(err)
		}

		read := make([]byte, len(want))
		_, err = c.Read(read)
		if err != nil {
			t.Error(err)
		}
		recvdBytes <- read
	}()

	dl := pl.Dialer()
	conn, err := dl("", time.Duration(0))
	if err != nil {
		t.Fatal(err)
	}

	_, err = conn.Write([]byte(want))
	if err != nil {
		t.Fatal(err)
	}

	select {
	case gotBytes := <-recvdBytes:
		got := string(gotBytes)
		if got != want {
			t.Fatalf("expected to get %s, got %s", got, want)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("timed out waiting for server to receive bytes")
	}
}

func TestUnblocking(t *testing.T) {
	for _, test := range []struct {
		desc                 string
		blockFuncShouldError bool
		blockFunc            func(*testutils.PipeListener, chan struct{}) error
		unblockFunc          func(*testutils.PipeListener) error
	}{
		{
			desc: "Accept unblocks Dial",
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				dl := pl.Dialer()
				_, err := dl("", time.Duration(0))
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				_, err := pl.Accept()
				return err
			},
		},
		{
			desc:                 "Close unblocks Dial",
			blockFuncShouldError: true, // because pl.Close will be called
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				dl := pl.Dialer()
				_, err := dl("", time.Duration(0))
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				return pl.Close()
			},
		},
		{
			desc: "Dial unblocks Accept",
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				_, err := pl.Accept()
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				dl := pl.Dialer()
				_, err := dl("", time.Duration(0))
				return err
			},
		},
		{
			desc:                 "Close unblocks Accept",
			blockFuncShouldError: true, // because pl.Close will be called
			blockFunc: func(pl *testutils.PipeListener, done chan struct{}) error {
				_, err := pl.Accept()
				close(done)
				return err
			},
			unblockFunc: func(pl *testutils.PipeListener) error {
				return pl.Close()
			},
		},
	} {
		t.Log(test.desc)
		testUnblocking(t, test.blockFunc, test.unblockFunc, test.blockFuncShouldError)
	}
}

func testUnblocking(t *testing.T, blockFunc func(*testutils.PipeListener, chan struct{}) error, unblockFunc func(*testutils.PipeListener) error, blockFuncShouldError bool) {
	pl := testutils.NewPipeListener()
	dialFinished := make(chan struct{})

	go func() {
		err := blockFunc(pl, dialFinished)
		if blockFuncShouldError && err == nil {
			t.Error("expected blocking func to return error because pl.Close was called, but got nil")
		}

		if !blockFuncShouldError && err != nil {
			t.Error(err)
		}
	}()

	select {
	case <-dialFinished:
		t.Fatal("expected Dial to block until pl.Close or pl.Accept")
	default:
	}

	if err := unblockFunc(pl); err != nil {
		t.Fatal(err)
	}

	select {
	case <-dialFinished:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("expected Accept to unblock after pl.Accept was called")
	}
}
