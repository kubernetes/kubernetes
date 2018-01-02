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
	"os"
	"time"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc"
	"github.com/coreos/etcd/mvcc/backend"
	"github.com/spf13/cobra"
)

var (
	batchInterval int
	batchLimit    int

	s mvcc.KV
)

func initMVCC() {
	be := backend.New("mvcc-bench", time.Duration(batchInterval), batchLimit)
	s = mvcc.NewStore(be, &lease.FakeLessor{}, nil)
	os.Remove("mvcc-bench") // boltDB has an opened fd, so removing the file is ok
}

// mvccCmd represents the MVCC storage benchmarking tools
var mvccCmd = &cobra.Command{
	Use:   "mvcc",
	Short: "Benchmark mvcc",
	Long: `storage subcommand is a set of various benchmark tools for MVCC storage subsystem of etcd.
Actual benchmarks are implemented as its subcommands.`,

	PersistentPreRun: mvccPreRun,
}

func init() {
	RootCmd.AddCommand(mvccCmd)

	mvccCmd.PersistentFlags().IntVar(&batchInterval, "batch-interval", 100, "Interval of batching (milliseconds)")
	mvccCmd.PersistentFlags().IntVar(&batchLimit, "batch-limit", 10000, "A limit of batched transaction")
}

func mvccPreRun(cmd *cobra.Command, args []string) {
	initMVCC()
}
