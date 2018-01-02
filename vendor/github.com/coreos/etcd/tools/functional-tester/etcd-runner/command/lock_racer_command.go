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

package command

import (
	"context"
	"errors"
	"fmt"

	"github.com/coreos/etcd/clientv3/concurrency"
	"github.com/spf13/cobra"
)

// NewLockRacerCommand returns the cobra command for "lock-racer runner".
func NewLockRacerCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "lock-racer",
		Short: "Performs lock race operation",
		Run:   runRacerFunc,
	}
	cmd.Flags().IntVar(&rounds, "rounds", 100, "number of rounds to run")
	cmd.Flags().IntVar(&totalClientConnections, "total-client-connections", 10, "total number of client connections")
	return cmd
}

func runRacerFunc(cmd *cobra.Command, args []string) {
	if len(args) > 0 {
		ExitWithError(ExitBadArgs, errors.New("lock-racer does not take any argument"))
	}

	rcs := make([]roundClient, totalClientConnections)
	ctx := context.Background()
	cnt := 0

	eps := endpointsFromFlag(cmd)
	dialTimeout := dialTimeoutFromCmd(cmd)

	for i := range rcs {
		var (
			s   *concurrency.Session
			err error
		)

		rcs[i].c = newClient(eps, dialTimeout)

		for {
			s, err = concurrency.NewSession(rcs[i].c)
			if err == nil {
				break
			}
		}
		m := concurrency.NewMutex(s, "racers")
		rcs[i].acquire = func() error { return m.Lock(ctx) }
		rcs[i].validate = func() error {
			if cnt++; cnt != 1 {
				return fmt.Errorf("bad lock; count: %d", cnt)
			}
			return nil
		}
		rcs[i].release = func() error {
			if err := m.Unlock(ctx); err != nil {
				return err
			}
			cnt = 0
			return nil
		}
	}
	doRounds(rcs, rounds)
}
