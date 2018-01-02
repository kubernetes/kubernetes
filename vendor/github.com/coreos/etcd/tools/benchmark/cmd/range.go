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
	"math"
	"os"
	"time"

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/report"

	"github.com/spf13/cobra"
	"golang.org/x/net/context"
	"golang.org/x/time/rate"
	"gopkg.in/cheggaaa/pb.v1"
)

// rangeCmd represents the range command
var rangeCmd = &cobra.Command{
	Use:   "range key [end-range]",
	Short: "Benchmark range",

	Run: rangeFunc,
}

var (
	rangeRate        int
	rangeTotal       int
	rangeConsistency string
)

func init() {
	RootCmd.AddCommand(rangeCmd)
	rangeCmd.Flags().IntVar(&rangeRate, "rate", 0, "Maximum range requests per second (0 is no limit)")
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

	if rangeRate == 0 {
		rangeRate = math.MaxInt32
	}
	limit := rate.NewLimiter(rate.Limit(rangeRate), 1)

	requests := make(chan v3.Op, totalClients)
	clients := mustCreateClients(totalClients, totalConns)

	bar = pb.New(rangeTotal)
	bar.Format("Bom !")
	bar.Start()

	r := newReport()
	for i := range clients {
		wg.Add(1)
		go func(c *v3.Client) {
			defer wg.Done()
			for op := range requests {
				limit.Wait(context.Background())

				st := time.Now()
				_, err := c.Do(context.Background(), op)
				r.Results() <- report.Result{Err: err, Start: st, End: time.Now()}
				bar.Increment()
			}
		}(clients[i])
	}

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

	rc := r.Run()
	wg.Wait()
	close(r.Results())
	bar.Finish()
	fmt.Printf("%s", <-rc)
}
