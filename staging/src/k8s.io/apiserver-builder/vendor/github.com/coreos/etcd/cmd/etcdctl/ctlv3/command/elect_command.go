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
	"errors"
	"os"
	"os/signal"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

var (
	electListen bool
)

// NewElectCommand returns the cobra command for "elect".
func NewElectCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "elect <election-name> [proposal]",
		Short: "Observes and participates in leader election",
		Run:   electCommandFunc,
	}
	cmd.Flags().BoolVarP(&electListen, "listen", "l", false, "observation mode")
	return cmd
}

func electCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 && len(args) != 2 {
		ExitWithError(ExitBadArgs, errors.New("elect takes one election name argument and an optional proposal argument."))
	}
	c := mustClientFromCmd(cmd)

	var err error
	if len(args) == 1 {
		if !electListen {
			ExitWithError(ExitBadArgs, errors.New("no proposal argument but -l not set"))
		}
		err = observe(c, args[0])
	} else {
		if electListen {
			ExitWithError(ExitBadArgs, errors.New("proposal given but -l is set"))
		}
		err = campaign(c, args[0], args[1])
	}
	if err != nil {
		ExitWithError(ExitError, err)
	}
}

func observe(c *clientv3.Client, election string) error {
	e := concurrency.NewElection(c, election)
	ctx, cancel := context.WithCancel(context.TODO())

	donec := make(chan struct{})
	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, os.Interrupt, os.Kill)
	go func() {
		<-sigc
		cancel()
	}()

	go func() {
		for resp := range e.Observe(ctx) {
			display.Get(resp)
		}
		close(donec)
	}()

	<-donec

	select {
	case <-ctx.Done():
	default:
		return errors.New("elect: observer lost")
	}

	return nil
}

func campaign(c *clientv3.Client, election string, prop string) error {
	e := concurrency.NewElection(c, election)
	ctx, cancel := context.WithCancel(context.TODO())

	donec := make(chan struct{})
	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, os.Interrupt, os.Kill)
	go func() {
		<-sigc
		cancel()
		close(donec)
	}()

	s, serr := concurrency.NewSession(c)
	if serr != nil {
		return serr
	}

	if err := e.Campaign(ctx, prop); err != nil {
		return err
	}

	// print key since elected
	resp, err := c.Get(ctx, e.Key())
	if err != nil {
		return err
	}
	display.Get(*resp)

	select {
	case <-donec:
	case <-s.Done():
		return errors.New("elect: session expired")
	}

	return e.Resign(context.TODO())
}
