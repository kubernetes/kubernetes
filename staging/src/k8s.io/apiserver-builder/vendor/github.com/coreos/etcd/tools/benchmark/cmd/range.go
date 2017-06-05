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
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
	"gopkg.in/cheggaaa/pb.v1"
)

// rangeCmd represents the range command
var rangeCmd = &cobra.Command{
	Use:   "range key [end-range]",
	Short: "Benchmark range",

	Run: rangeFunc,
}

var (
	rangeTotal       int
	rangeConsistency string
)

func init() {
	RootCmd.AddCommand(rangeCmd)
	rangeCmd.Flags().IntVar(&rangeTotal, "total", 10000, "Total number of range requests")
	rangeCmd.Flags().StringVar(&rangeConsistency, "consistency", "l", "Linearizable(l) or Serializable(s)")
}

func rangeFunc(cmd *cobra.Command, args []string) {
	if len(args) == 0 || len(args) > 2 {
		fmt.Fprintln(os.Stderr, cmd.Usage())
		os.Exit(1)
	}

	k := args[0]
	end := ""
	if len(args) == 2 {
		end = args[1]
	}

	if rangeConsistency == "l" {
		fmt.Println("bench with linearizable range")
	} else if rangeConsistency == "s" {
		fmt.Println("bench with serializable range")
	} else {
		fmt.Fprintln(os.Stderr, cmd.Usage())
		os.Exit(1)
	}

	results = make(chan result)
	requests := make(chan v3.Op, totalClients)
	bar = pb.New(rangeTotal)

	clients := mustCreateClients(totalClients, totalConns)

	bar.Format("Bom !")
	bar.Start()

	for i := range clients {
		wg.Add(1)
		go doRange(clients[i].KV, requests)
	}

	pdoneC := printReport(results)

	go func() {
		for i := 0; i < rangeTotal; i++ {
			opts := []v3.OpOption{v3.WithRange(end)}
			if rangeConsistency == "s" {
				opts = append(opts, v3.WithSerializable())
			}
			op := v3.OpGet(k, opts...)
			requests <- op
		}
		close(requests)
	}()

	wg.Wait()

	bar.Finish()

	close(results)
	<-pdoneC
}

func doRange(client v3.KV, requests <-chan v3.Op) {
	defer wg.Done()

	for op := range requests {
		st := time.Now()
		_, err := client.Do(context.Background(), op)

		var errStr string
		if err != nil {
			errStr = err.Error()
		}
		results <- result{errStr: errStr, duration: time.Since(st), happened: time.Now()}
		bar.Increment()
	}
}
