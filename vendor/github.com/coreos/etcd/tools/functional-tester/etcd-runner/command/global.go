// Copyright 2016 The etcd Authors
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

package command

import (
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/coreos/etcd/clientv3"

	"github.com/spf13/cobra"
)

var (
	rounds                 int           // total number of rounds the operation needs to be performed
	totalClientConnections int           // total number of client connections to be made with server
	noOfPrefixes           int           // total number of prefixes which will be watched upon
	watchPerPrefix         int           // number of watchers per prefix
	reqRate                int           // put request per second
	totalKeys              int           // total number of keys for operation
	runningTime            time.Duration // time for which operation should be performed
)

// GlobalFlags are flags that defined globally
// and are inherited to all sub-commands.
type GlobalFlags struct {
	Endpoints   []string
	DialTimeout time.Duration
}

type roundClient struct {
	c        *clientv3.Client
	progress int
	acquire  func() error
	validate func() error
	release  func() error
}

func newClient(eps []string, timeout time.Duration) *clientv3.Client {
	c, err := clientv3.New(clientv3.Config{
		Endpoints:   eps,
		DialTimeout: time.Duration(timeout) * time.Second,
	})
	if err != nil {
		log.Fatal(err)
	}
	return c
}

func doRounds(rcs []roundClient, rounds int) {
	var mu sync.Mutex
	var wg sync.WaitGroup

	wg.Add(len(rcs))
	finished := make(chan struct{}, 0)
	for i := range rcs {
		go func(rc *roundClient) {
			defer wg.Done()
			for rc.progress < rounds {
				for rc.acquire() != nil { /* spin */
				}

				mu.Lock()
				if err := rc.validate(); err != nil {
					log.Fatal(err)
				}
				mu.Unlock()

				time.Sleep(10 * time.Millisecond)
				rc.progress++
				finished <- struct{}{}

				mu.Lock()
				for rc.release() != nil {
					mu.Unlock()
					mu.Lock()
				}
				mu.Unlock()
			}
		}(&rcs[i])
	}

	start := time.Now()
	for i := 1; i < len(rcs)*rounds+1; i++ {
		select {
		case <-finished:
			if i%100 == 0 {
				fmt.Printf("finished %d, took %v\n", i, time.Since(start))
				start = time.Now()
			}
		case <-time.After(time.Minute):
			log.Panic("no progress after 1 minute!")
		}
	}
	wg.Wait()

	for _, rc := range rcs {
		rc.c.Close()
	}
}

func endpointsFromFlag(cmd *cobra.Command) []string {
	endpoints, err := cmd.Flags().GetStringSlice("endpoints")
	if err != nil {
		ExitWithError(ExitError, err)
	}
	return endpoints
}

func dialTimeoutFromCmd(cmd *cobra.Command) time.Duration {
	dialTimeout, err := cmd.Flags().GetDuration("dial-timeout")
	if err != nil {
		ExitWithError(ExitError, err)
	}
	return dialTimeout
}
