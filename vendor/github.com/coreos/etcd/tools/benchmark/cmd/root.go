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
	"sync"
	"time"

	"github.com/coreos/etcd/pkg/transport"

	"github.com/spf13/cobra"
	"gopkg.in/cheggaaa/pb.v1"
)

// This represents the base command when called without any subcommands
var RootCmd = &cobra.Command{
	Use:   "benchmark",
	Short: "A low-level benchmark tool for etcd3",
	Long: `benchmark is a low-level benchmark tool for etcd3.
It uses gRPC client directly and does not depend on 
etcd client library.
	`,
}

var (
	endpoints    []string
	totalConns   uint
	totalClients uint
	precise      bool
	sample       bool

	bar *pb.ProgressBar
	wg  sync.WaitGroup

	tls transport.TLSInfo

	cpuProfPath string
	memProfPath string

	user string

	dialTimeout time.Duration
)

func init() {
	RootCmd.PersistentFlags().StringSliceVar(&endpoints, "endpoints", []string{"127.0.0.1:2379"}, "gRPC endpoints")
	RootCmd.PersistentFlags().UintVar(&totalConns, "conns", 1, "Total number of gRPC connections")
	RootCmd.PersistentFlags().UintVar(&totalClients, "clients", 1, "Total number of gRPC clients")

	RootCmd.PersistentFlags().BoolVar(&precise, "precise", false, "use full floating point precision")
	RootCmd.PersistentFlags().BoolVar(&sample, "sample", false, "'true' to sample requests for every second")
	RootCmd.PersistentFlags().StringVar(&tls.CertFile, "cert", "", "identify HTTPS client using this SSL certificate file")
	RootCmd.PersistentFlags().StringVar(&tls.KeyFile, "key", "", "identify HTTPS client using this SSL key file")
	RootCmd.PersistentFlags().StringVar(&tls.CAFile, "cacert", "", "verify certificates of HTTPS-enabled servers using this CA bundle")

	RootCmd.PersistentFlags().StringVar(&user, "user", "", "specify username and password in username:password format")
	RootCmd.PersistentFlags().DurationVar(&dialTimeout, "dial-timeout", 0, "dial timeout for client connections")
}
