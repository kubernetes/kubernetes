// Copyright 2015 Nippon Telegraph and Telephone Corporation.
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
	"github.com/coreos/etcd/storage"
	"github.com/coreos/etcd/storage/backend"
	"github.com/spf13/cobra"
)

var (
	batchInterval int
	batchLimit    int

	s storage.KV
)

func initStorage() {
	be := backend.New("storage-bench", time.Duration(batchInterval), batchLimit)
	s = storage.NewStore(be, &lease.FakeLessor{})
	os.Remove("storage-bench") // boltDB has an opened fd, so removing the file is ok
}

// storageCmd represents the storage benchmarking tools
var storageCmd = &cobra.Command{
	Use:   "storage",
	Short: "Benchmark storage",
	Long: `storage subcommand is a set of various benchmark tools for storage subsystem of etcd.
Actual benchmarks are implemented as its subcommands.`,

	PersistentPreRun: storagePreRun,
}

func init() {
	RootCmd.AddCommand(storageCmd)

	storageCmd.PersistentFlags().IntVar(&batchInterval, "batch-interval", 100, "Interval of batching (milliseconds)")
	storageCmd.PersistentFlags().IntVar(&batchLimit, "batch-limit", 10000, "A limit of batched transaction")
}

func storagePreRun(cmd *cobra.Command, args []string) {
	initStorage()
}
