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

package cmd

import (
	"fmt"
	"sync"
	"time"

	v3 "github.com/coreos/etcd/clientv3"

	"github.com/spf13/cobra"
	"golang.org/x/net/context"
	"gopkg.in/cheggaaa/pb.v1"
)

// watchGetCmd represents the watch command
var watchGetCmd = &cobra.Command{
	Use:   "watch-get",
	Short: "Benchmark watch with get",
	Long:  `Benchmark for serialized key gets with many unsynced watchers`,
	Run:   watchGetFunc,
}

var (
	watchGetTotalWatchers int
	watchGetTotalStreams  int
	watchEvents           int
	firstWatch            sync.Once
)

func init() {
	RootCmd.AddCommand(watchGetCmd)
	watchGetCmd.Flags().IntVar(&watchGetTotalWatchers, "watchers", 10000, "Total number of watchers")
	watchGetCmd.Flags().IntVar(&watchGetTotalStreams, "streams", 1, "Total number of watcher streams")
	watchGetCmd.Flags().IntVar(&watchEvents, "events", 8, "Number of events per watcher")
}

func watchGetFunc(cmd *cobra.Command, args []string) {
	clients := mustCreateClients(totalClients, totalConns)
	getClient := mustCreateClients(1, 1)

	// setup keys for watchers
	watchRev := int64(0)
	for i := 0; i < watchEvents; i++ {
		v := fmt.Sprintf("%d", i)
		resp, err := clients[0].Put(context.TODO(), "watchkey", v)
		if err != nil {
			panic(err)
		}
		if i == 0 {
			watchRev = resp.Header.Revision
		}
	}

	streams := make([]v3.Watcher, watchGetTotalStreams)
	for i := range streams {
		streams[i] = v3.NewWatcher(clients[i%len(clients)])
	}

	// results from trying to do serialized gets with concurrent watchers
	results = make(chan result)

	bar = pb.New(watchGetTotalWatchers * watchEvents)
	bar.Format("Bom !")
	bar.Start()

	pdoneC := printReport(results)
	wg.Add(watchGetTotalWatchers)
	ctx, cancel := context.WithCancel(context.TODO())
	f := func() {
		doSerializedGet(ctx, getClient[0], results)
	}
	for i := 0; i < watchGetTotalWatchers; i++ {
		go doUnsyncWatch(streams[i%len(streams)], watchRev, f)
	}
	wg.Wait()
	cancel()
	bar.Finish()
	fmt.Printf("Get during watch summary:\n")
	<-pdoneC
}

func doSerializedGet(ctx context.Context, client *v3.Client, results chan result) {
	for {
		st := time.Now()
		_, err := client.Get(ctx, "abc", v3.WithSerializable())
		if ctx.Err() != nil {
			break
		}
		var errStr string
		if err != nil {
			errStr = err.Error()
		}
		res := result{errStr: errStr, duration: time.Since(st), happened: time.Now()}
		results <- res
	}
	close(results)
}

func doUnsyncWatch(stream v3.Watcher, rev int64, f func()) {
	wch := stream.Watch(context.TODO(), "watchkey", v3.WithRev(rev))
	if wch == nil {
		panic("could not open watch channel")
	}
	firstWatch.Do(func() { go f() })
	i := 0
	for i < watchEvents {
		wev := <-wch
		i += len(wev.Events)
		bar.Add(len(wev.Events))
	}
	wg.Done()
}
