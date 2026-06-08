/*
MIT License

Copyright (c) 2024 Vish Kannan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Package vishhstress was forked from <https://github.com/vishh/stress/tree/eab4e3384bcad9899b8b801b4a1917a758e97d96>
// so that it can be consumed from agnhost.
package vishhstress

import (
	"io"
	"io/ioutil"
	"os"
	"time"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
)

var (
	argMemTotal         string
	argMemStepSize      string
	argMemSleepDuration time.Duration
	argCpus             int
	buffer              [][]byte
)

// CmdStress is used by agnhost Cobra.
var CmdStress = &cobra.Command{
	Use:   "stress",
	Short: "Lightweight compute resource stress utlity",
	Args:  cobra.NoArgs,
	Run:   main,
}

func init() {
	flags := CmdStress.Flags()
	flags.StringVar(&argMemTotal, "mem-total", "0", "total memory to be consumed. Memory will be consumed via multiple allocations.")
	flags.StringVar(&argMemStepSize, "mem-alloc-size", "4Ki", "amount of memory to be consumed in each allocation")
	flags.DurationVar(&argMemSleepDuration, "mem-alloc-sleep", time.Millisecond, "duration to sleep between allocations")
	flags.IntVar(&argCpus, "cpus", 0, "total number of CPUs to utilize")
}

func main(cmd *cobra.Command, _ []string) {
	total := resource.MustParse(argMemTotal)
	stepSize := resource.MustParse(argMemStepSize)
	klog.Infof("Allocating %q memory, in %q chunks, with a %v sleep between allocations", total.String(), stepSize.String(), argMemSleepDuration)
	burnCPU()
	allocateMemory(total, stepSize)
	klog.Infof("Allocated %q memory", total.String())
	select {}
}

func burnCPU() {
	src, err := os.Open("/dev/zero")
	if err != nil {
		klog.Fatalf("failed to open /dev/zero")
	}
	for i := 0; i < argCpus; i++ {
		klog.Infof("Spawning a thread to consume CPU")
		go func() {
			_, err := io.Copy(ioutil.Discard, src)
			if err != nil {
				klog.Fatalf("failed to copy from /dev/zero to /dev/null: %v", err)
			}
		}()
	}
}

func allocateMemory(total, stepSize resource.Quantity) {
	for i := int64(1); i*stepSize.Value() <= total.Value(); i++ {
		newBuffer := make([]byte, stepSize.Value())
		for i := range newBuffer {
			newBuffer[i] = 0
		}
		buffer = append(buffer, newBuffer)
		time.Sleep(argMemSleepDuration)
	}
}
