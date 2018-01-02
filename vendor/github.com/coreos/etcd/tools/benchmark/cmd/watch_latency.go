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
	"fmt"
	"os"
	"time"

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/report"

	"github.com/spf13/cobra"
	"golang.org/x/net/context"
	"golang.org/x/time/rate"
	"gopkg.in/cheggaaa/pb.v1"
)

// watchLatencyCmd represents the watch latency command
var watchLatencyCmd = &cobra.Command{
	Use:   "watch-latency",
	Short: "Benchmark watch latency",
	Long: `Benchmarks the latency for watches by measuring 
	the latency between writing to a key and receiving the 
	associated watch response.`,
	Run: watchLatencyFunc,
}

var (
	watchLTotal     int
	watchLPutRate   int
	watchLKeySize   int
	watchLValueSize int
)

func init() {
	RootCmd.AddCommand(watchLatencyCmd)
	watchLatencyCmd.Flags().IntVar(&watchLTotal, "total", 10000, "Total number of watch responses.")
	watchLatencyCmd.Flags().IntVar(&watchLPutRate, "put-rate", 100, "Number of keys to put per second")
	watchLatencyCmd.Flags().IntVar(&watchLKeySize, "key-size", 32, "Key size of watch request")
	watchLatencyCmd.Flags().IntVar(&watchLValueSize, "val-size", 32, "Val size of watch request")
}

func watchLatencyFunc(cmd *cobra.Command, args []string) {
	key := string(mustRandBytes(watchLKeySize))
	value := string(mustRandBytes(watchLValueSize))

	client := mustCreateConn()
	stream := v3.NewWatcher(client)
	wch := stream.Watch(context.TODO(), key)

	bar = pb.New(watchLTotal)
	bar.Format("Bom !")
	bar.Start()

	limiter := rate.NewLimiter(rate.Limit(watchLPutRate), watchLPutRate)
	r := newReport()
	rc := r.Run()

	for i := 0; i < watchLTotal; i++ {
		// limit key put as per reqRate
		if err := limiter.Wait(context.TODO()); err != nil {
			break
		}
		_, err := client.Put(context.TODO(), string(key), value)

		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to Put for watch latency benchmark: %v\n", err)
			os.Exit(1)
		}
		st := time.Now()
		<-wch
		r.Results() <- report.Result{Err: err, Start: st, End: time.Now()}
		bar.Increment()
	}

	close(r.Results())
	bar.Finish()
	fmt.Printf("%s", <-rc)
}
