/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/component-base/logs"
	"k8s.io/klog/v2"

	_ "k8s.io/component-base/logs/json/register"
)

func main() {
	command := NewLoggerCommand()
	logs.InitLogs()
	defer logs.FlushLogs()

	if err := command.Execute(); err != nil {
		os.Exit(1)
	}
}

func NewLoggerCommand() *cobra.Command {
	c := newConfig()

	cmd := &cobra.Command{
		Run: func(cmd *cobra.Command, args []string) {
			errs := c.Validate()
			if len(errs) != 0 {
				fmt.Fprintf(os.Stderr, "%v\n", errs)
				os.Exit(1)
			}
			c.Apply()
			runLogger(c)
		},
	}
	c.AddFlags(cmd.Flags())
	return cmd
}

type config struct {
	duration time.Duration
	messageLength int
	errorPercentage float64

	logs *logs.Options
}

func newConfig() *config {
	return &config{
		duration: 10 *time.Second,
		messageLength: 300,
		errorPercentage: 1.0,

		logs: logs.NewOptions(),
	}
}

func (c *config) AddFlags(fs *pflag.FlagSet) {
	c.logs.AddFlags(fs)
	fs.DurationVar(&c.duration, "duration", c.duration, "How long loader should run for")
	fs.IntVar(&c.messageLength, "message-length", c.messageLength, "Length of the message written")
	fs.Float64Var(&c.errorPercentage, "error-percentage", c.errorPercentage, "Percentage of errors logs written")
}

func (c *config) Validate() []error {
	errs := c.logs.Validate()
	if c.errorPercentage < 0 || c.errorPercentage > 100 {
		errs = append(errs, fmt.Errorf("error-percentage needs to be between 0 and 100"))
	}
	return errs
}


func (c *config) Apply() {
	c.logs.Apply()
}

func runLogger(c *config) {
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, c.duration)
	defer cancel()

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			work(ctx, c)
		}()
	}
	wg.Wait()
	klog.Flush()
}

func work(ctx context.Context, c *config) {
	msg := strings.Repeat("X", c.messageLength)
	acc := 0.0
	err := errors.New("fail")
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		if acc > 100 {
			klog.ErrorS(err, msg, "key", "value")
			acc -= 100
		} else {
			klog.InfoS(msg, "key", "value")
		}
		acc += c.errorPercentage
	}
}
