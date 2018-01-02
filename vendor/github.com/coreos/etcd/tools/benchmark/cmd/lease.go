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
	"time"

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/report"

	"github.com/spf13/cobra"
	"golang.org/x/net/context"
	"gopkg.in/cheggaaa/pb.v1"
)

var leaseKeepaliveCmd = &cobra.Command{
	Use:   "lease-keepalive",
	Short: "Benchmark lease keepalive",

	Run: leaseKeepaliveFunc,
}

var (
	leaseKeepaliveTotal int
)

func init() {
	RootCmd.AddCommand(leaseKeepaliveCmd)
	leaseKeepaliveCmd.Flags().IntVar(&leaseKeepaliveTotal, "total", 10000, "Total number of lease keepalive requests")
}

func leaseKeepaliveFunc(cmd *cobra.Command, args []string) {
	requests := make(chan struct{})
	clients := mustCreateClients(totalClients, totalConns)

	bar = pb.New(leaseKeepaliveTotal)
	bar.Format("Bom !")
	bar.Start()

	r := newReport()
	for i := range clients {
		wg.Add(1)
		go func(c v3.Lease) {
			defer wg.Done()
			resp, err := c.Grant(context.Background(), 100)
			if err != nil {
				panic(err)
			}
			for _ = range requests {
				st := time.Now()
				_, err := c.KeepAliveOnce(context.TODO(), resp.ID)
				r.Results() <- report.Result{Err: err, Start: st, End: time.Now()}
				bar.Increment()
			}
		}(clients[i])
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < leaseKeepaliveTotal; i++ {
			requests <- struct{}{}
		}
		close(requests)
	}()

	rc := r.Run()
	wg.Wait()
	close(r.Results())
	bar.Finish()
	fmt.Printf("%s", <-rc)
}
