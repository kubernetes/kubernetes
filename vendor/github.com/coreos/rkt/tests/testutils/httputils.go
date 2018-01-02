// Copyright 2015 The rkt Authors
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

package testutils

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/coreos/rkt/tests/testutils/logger"
	"github.com/hydrogen18/stoppableListener"
)

func HTTPServe(addr string, timeout int) error {
	hostname, err := os.Hostname()
	if err != nil {
		panic(err)
	}

	originalListener, err := net.Listen("tcp4", addr)
	if err != nil {
		panic(err)
	}
	sl, err := stoppableListener.New(originalListener)
	if err != nil {
		panic(err)
	}

	c := make(chan string)
	go func() {
		logger.Logf("%v: serving on %v\n", hostname, addr)

		// Wait for either timeout or connect from client
		select {
		case <-time.After(time.Duration(timeout) * time.Second):
			{
				logger.Logf("%v: Serve timed out after %v seconds\n", hostname, timeout)
			}
		case client := (<-c):
			{
				logger.Logf("%v: Serve got a connection from %v\n", hostname, client)
			}
		}
		sl.Stop()
	}()

	serveMux := http.NewServeMux()
	serveMux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		logger.Logf("%v: Serve got a connection from %v\n", hostname, r.RemoteAddr)
		fmt.Fprintf(w, "%v", hostname)
		c <- r.RemoteAddr
	})
	err = http.Serve(sl, serveMux)
	if err != nil && err.Error() == "Listener stopped" {
		err = nil
	}
	return err
}

func HTTPGet(addr string) (string, error) {
	logger.Logf("Connecting to %v", addr)
	res, err := http.Get(fmt.Sprintf("%v", addr))
	if err != nil {
		return "", err
	}
	text, err := ioutil.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		return "", err
	}
	return string(text), nil
}
