/*
Copyright 2019 The Kubernetes Authors.

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

package pause

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"
)

var (
	delayShutdown int
)

// CmdPause is used by agnhost Cobra.
var CmdPause = &cobra.Command{
	Use:   "pause",
	Short: "Pauses the execution",
	Long:  `Pauses the execution. Useful for keeping the containers running, so other commands can be executed.`,
	Args:  cobra.MaximumNArgs(0),
	Run:   pause,
}

func init() {
	CmdPause.Flags().IntVar(&delayShutdown, "delay-shutdown", 0, "Number of seconds to delay shutdown when receiving SIGTERM/SIGINT.")
}

func pause(cmd *cobra.Command, args []string) {
	fmt.Println("Paused")
	sigCh := make(chan os.Signal, 1)
	done := make(chan int, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	signal.Notify(sigCh, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		switch sig {
		case syscall.SIGINT:
			done <- 1
		case syscall.SIGTERM:
			done <- 2
		}
	}()

	result := <-done
	if delayShutdown > 0 {
		fmt.Printf("Sleeping %d seconds before exit ...", delayShutdown)
		time.Sleep(time.Duration(delayShutdown) * time.Second)
	}

	fmt.Printf("exiting %d\n", result)
	os.Exit(result)
}
