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
	"crypto/rand"
	"fmt"
	"os"
	"runtime/pprof"
	"time"

	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/pkg/report"

	"github.com/spf13/cobra"
)

// mvccPutCmd represents a storage put performance benchmarking tool
var mvccPutCmd = &cobra.Command{
	Use:   "put",
	Short: "Benchmark put performance of storage",

	Run: mvccPutFunc,
}

var (
	totalNrKeys    int
	storageKeySize int
	valueSize      int
	txn            bool
)

func init() {
	mvccCmd.AddCommand(mvccPutCmd)

	mvccPutCmd.Flags().IntVar(&totalNrKeys, "total", 100, "a total number of keys to put")
	mvccPutCmd.Flags().IntVar(&storageKeySize, "key-size", 64, "a size of key (Byte)")
	mvccPutCmd.Flags().IntVar(&valueSize, "value-size", 64, "a size of value (Byte)")
	mvccPutCmd.Flags().BoolVar(&txn, "txn", false, "put a key in transaction or not")

	// TODO: after the PR https://github.com/spf13/cobra/pull/220 is merged, the below pprof related flags should be moved to RootCmd
	mvccPutCmd.Flags().StringVar(&cpuProfPath, "cpuprofile", "", "the path of file for storing cpu profile result")
	mvccPutCmd.Flags().StringVar(&memProfPath, "memprofile", "", "the path of file for storing heap profile result")

}

func createBytesSlice(bytesN, sliceN int) [][]byte {
	rs := make([][]byte, sliceN)
	for i := range rs {
		rs[i] = make([]byte, bytesN)
		if _, err := rand.Read(rs[i]); err != nil {
			panic(err)
		}
	}
	return rs
}

func mvccPutFunc(cmd *cobra.Command, args []string) {
	if cpuProfPath != "" {
		f, err := os.Create(cpuProfPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Failed to create a file for storing cpu profile result: ", err)
			os.Exit(1)
		}

		err = pprof.StartCPUProfile(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Failed to start cpu profile: ", err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}

	if memProfPath != "" {
		f, err := os.Create(memProfPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Failed to create a file for storing heap profile result: ", err)
			os.Exit(1)
		}

		defer func() {
			err := pprof.WriteHeapProfile(f)
			if err != nil {
				fmt.Fprintln(os.Stderr, "Failed to write heap profile result: ", err)
				// can do nothing for handling the error
			}
		}()
	}

	keys := createBytesSlice(storageKeySize, totalNrKeys)
	vals := createBytesSlice(valueSize, totalNrKeys)

	r := newReport()
	rrc := r.Results()

	rc := r.Run()
	for i := 0; i < totalNrKeys; i++ {
		st := time.Now()
		if txn {
			id := s.TxnBegin()
			if _, err := s.TxnPut(id, keys[i], vals[i], lease.NoLease); err != nil {
				fmt.Fprintln(os.Stderr, "txn put error:", err)
				os.Exit(1)
			}
			s.TxnEnd(id)
		} else {
			s.Put(keys[i], vals[i], lease.NoLease)
		}
		rrc <- report.Result{Start: st, End: time.Now()}
	}

	close(r.Results())
	fmt.Printf("%s", <-rc)
}
