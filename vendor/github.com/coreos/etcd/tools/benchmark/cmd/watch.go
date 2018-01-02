// Copyright 2015 The etcd Authors
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

package cmd

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"sync/atomic"
	"time"

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/report"

	"github.com/spf13/cobra"
	"golang.org/x/net/context"
	"gopkg.in/cheggaaa/pb.v1"
)

// watchCmd represents the watch command
var watchCmd = &cobra.Command{
	Use:   "watch",
	Short: "Benchmark watch",
	Long: `Benchmark watch tests the performance of processing watch requests and 
sending events to watchers. It tests the sending performance by 
changing the value of the watched keys with concurrent put 
requests.

During the test, each watcher watches (--total/--watchers) keys 

(a watcher might watch on the same key multiple times if 
--watched-key-total is small).

Each key is watched by (--total/--watched-key-total) watchers.
`,
	Run: watchFunc,
}

var (
	watchTotalStreams int
	watchTotal        int
	watchedKeyTotal   int

	watchPutRate  int
	watchPutTotal int

	watchKeySize      int
	watchKeySpaceSize int
	watchSeqKeys      bool

	eventsTotal int

	nrWatchCompleted       int32
	nrRecvCompleted        int32
	watchCompletedNotifier chan struct{}
	recvCompletedNotifier  chan struct{}
)

func init() {
	RootCmd.AddCommand(watchCmd)
	watchCmd.Flags().IntVar(&watchTotalStreams, "watchers", 10000, "Total number of watchers")
	watchCmd.Flags().IntVar(&watchTotal, "total", 100000, "Total number of watch requests")
	watchCmd.Flags().IntVar(&watchedKeyTotal, "watched-key-total", 10000, "Total number of keys to be watched")

	watchCmd.Flags().IntVar(&watchPutRate, "put-rate", 100, "Number of keys to put per second")
	watchCmd.Flags().IntVar(&watchPutTotal, "put-total", 10000, "Number of put requests")

	watchCmd.Flags().IntVar(&watchKeySize, "key-size", 32, "Key size of watch request")
	watchCmd.Flags().IntVar(&watchKeySpaceSize, "key-space-size", 1, "Maximum possible keys")
	watchCmd.Flags().BoolVar(&watchSeqKeys, "sequential-keys", false, "Use sequential keys")
}

func watchFunc(cmd *cobra.Command, args []string) {
	if watchKeySpaceSize <= 0 {
		fmt.Fprintf(os.Stderr, "expected positive --key-space-size, got (%v)", watchKeySpaceSize)
		os.Exit(1)
	}

	watched := make([]string, watchedKeyTotal)
	numWatchers := make(map[string]int)
	for i := range watched {
		k := make([]byte, watchKeySize)
		if watchSeqKeys {
			binary.PutVarint(k, int64(i%watchKeySpaceSize))
		} else {
			binary.PutVarint(k, int64(rand.Intn(watchKeySpaceSize)))
		}
		watched[i] = string(k)
	}

	requests := make(chan string, totalClients)

	clients := mustCreateClients(totalClients, totalConns)

	streams := make([]v3.Watcher, watchTotalStreams)
	for i := range streams {
		streams[i] = v3.NewWatcher(clients[i%len(clients)])
	}

	// watching phase
	bar = pb.New(watchTotal)
	bar.Format("Bom !")
	bar.Start()

	atomic.StoreInt32(&nrWatchCompleted, int32(0))
	watchCompletedNotifier = make(chan struct{})

	r := report.NewReportRate("%4.4f")
	for i := range streams {
		go doWatch(streams[i], requests, r.Results())
	}

	go func() {
		for i := 0; i < watchTotal; i++ {
			key := watched[i%len(watched)]
			requests <- key
			numWatchers[key]++
		}
		close(requests)
	}()

	rc := r.Run()
	<-watchCompletedNotifier
	bar.Finish()
	close(r.Results())
	fmt.Printf("Watch creation summary:\n%s", <-rc)

	// put phase
	eventsTotal = 0
	for i := 0; i < watchPutTotal; i++ {
		eventsTotal += numWatchers[watched[i%len(watched)]]
	}

	bar = pb.New(eventsTotal)
	bar.Format("Bom !")
	bar.Start()

	atomic.StoreInt32(&nrRecvCompleted, 0)
	recvCompletedNotifier = make(chan struct{})
	putreqc := make(chan v3.Op)

	r = report.NewReportRate("%4.4f")
	for i := 0; i < watchPutTotal; i++ {
		go func(c *v3.Client) {
			for op := range putreqc {
				if _, err := c.Do(context.TODO(), op); err != nil {
					fmt.Fprintf(os.Stderr, "failed to Put for watch benchmark: %v\n", err)
					os.Exit(1)
				}
			}
		}(clients[i%len(clients)])
	}

	go func() {
		for i := 0; i < watchPutTotal; i++ {
			putreqc <- v3.OpPut(watched[i%(len(watched))], "data")
			// TODO: use a real rate-limiter instead of sleep.
			time.Sleep(time.Second / time.Duration(watchPutRate))
		}
		close(putreqc)
	}()

	rc = r.Run()
	<-recvCompletedNotifier
	bar.Finish()
	close(r.Results())
	fmt.Printf("Watch events received summary:\n%s", <-rc)
}

func doWatch(stream v3.Watcher, requests <-chan string, results chan<- report.Result) {
	for r := range requests {
		st := time.Now()
		wch := stream.Watch(context.TODO(), r)
		results <- report.Result{Start: st, End: time.Now()}
		bar.Increment()
		go recvWatchChan(wch, results)
	}
	atomic.AddInt32(&nrWatchCompleted, 1)
	if atomic.LoadInt32(&nrWatchCompleted) == int32(watchTotalStreams) {
		watchCompletedNotifier <- struct{}{}
	}
}

func recvWatchChan(wch v3.WatchChan, results chan<- report.Result) {
	for r := range wch {
		st := time.Now()
		for range r.Events {
			results <- report.Result{Start: st, End: time.Now()}
			bar.Increment()
			atomic.AddInt32(&nrRecvCompleted, 1)
		}

		if atomic.LoadInt32(&nrRecvCompleted) == int32(eventsTotal) {
			recvCompletedNotifier <- struct{}{}
			break
		}
	}
}
