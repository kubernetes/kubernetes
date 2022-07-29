/*
Copyright 2021 The Kubernetes Authors.

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

package benchmark

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	logsapi "k8s.io/component-base/logs/api/v1"
	logsjson "k8s.io/component-base/logs/json"
	"k8s.io/klog/v2"
)

func BenchmarkEncoding(b *testing.B) {
	// Each "data/(v[0-9]/)?*.log" file is expected to contain JSON log
	// messages. We generate one sub-benchmark for each file where logging
	// is tested with the log level from the directory.  Symlinks can be
	// used to test the same file with different levels.
	if err := filepath.Walk("data", func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !strings.HasSuffix(path, ".log") {
			return nil
		}
		messages, stats, err := loadLog(path)
		if err != nil {
			return err
		}
		if info.Mode()&fs.ModeSymlink == 0 {
			b.Log(path + "\n" + stats.String())
		}
		b.Run(strings.TrimSuffix(strings.TrimPrefix(path, "data/"), ".log"), func(b *testing.B) {
			// Take verbosity threshold from directory, if present.
			vMatch := regexp.MustCompile(`/v(\d+)/`).FindStringSubmatch(path)
			v := 0
			if vMatch != nil {
				v, _ = strconv.Atoi(vMatch[1])
			}
			fileSizes := map[string]int{}
			b.Run("stats", func(b *testing.B) {
				// Nothing to do. Use this for "go test -v
				// -bench=BenchmarkLogging/.*/stats" to print
				// just the statistics.
			})
			b.Run("printf", func(b *testing.B) {
				b.ResetTimer()
				output = 0
				for i := 0; i < b.N; i++ {
					for _, item := range messages {
						if item.verbosity <= v {
							printf(item)
						}
					}
				}
				fileSizes["printf"] = int(output) / b.N
			})
			b.Run("structured", func(b *testing.B) {
				b.ResetTimer()
				output = 0
				for i := 0; i < b.N; i++ {
					for _, item := range messages {
						if item.verbosity <= v {
							prints(item)
						}
					}
				}
				fileSizes["structured"] = int(output) / b.N
			})
			b.Run("JSON", func(b *testing.B) {
				klog.SetLogger(jsonLogger)
				defer klog.ClearLogger()
				b.ResetTimer()
				output = 0
				for i := 0; i < b.N; i++ {
					for _, item := range messages {
						if item.verbosity <= v {
							prints(item)
						}
					}
				}
				fileSizes["JSON"] = int(output) / b.N
			})

			b.Log(fmt.Sprintf("file sizes: %v\n", fileSizes))
		})
		return nil
	}); err != nil {
		b.Fatalf("reading 'data' directory: %v", err)
	}
}

type loadGeneratorConfig struct {
	// Length of the message written in each log entry.
	messageLength int

	// Percentage of error log entries written.
	errorPercentage float64

	// Number of concurrent goroutines that generate log entries.
	workers int
}

// BenchmarkWriting simulates writing of a stream which mixes info and error log
// messages at a certain ratio. In contrast to BenchmarkEncoding, this stresses
// the output handling and includes the usual additional information (caller,
// time stamp).
//
// See https://github.com/kubernetes/kubernetes/issues/107029 for the
// motivation.
func BenchmarkWriting(b *testing.B) {
	flag.Set("skip_headers", "false")
	defer flag.Set("skip_headers", "true")

	// This could be made configurable and/or we could benchmark different
	// configurations automatically.
	config := loadGeneratorConfig{
		messageLength:   300,
		errorPercentage: 1.0,
		workers:         100,
	}

	benchmarkWriting(b, config)
}

func benchmarkWriting(b *testing.B, config loadGeneratorConfig) {
	b.Run("discard", func(b *testing.B) {
		benchmarkOutputFormats(b, config, true)
	})
	b.Run("tmp-files", func(b *testing.B) {
		benchmarkOutputFormats(b, config, false)
	})
}

func benchmarkOutputFormats(b *testing.B, config loadGeneratorConfig, discard bool) {
	tmpDir := b.TempDir()
	b.Run("structured", func(b *testing.B) {
		var out *os.File
		if !discard {
			var err error
			out, err = os.Create(path.Join(tmpDir, "all.log"))
			if err != nil {
				b.Fatal(err)
			}
			klog.SetOutput(out)
			defer klog.SetOutput(&output)
		}
		generateOutput(b, config, nil, out)
	})
	b.Run("JSON", func(b *testing.B) {
		c := logsapi.NewLoggingConfiguration()
		var logger logr.Logger
		var flush func()
		var out1, out2 *os.File
		if !discard {
			var err error
			out1, err = os.Create(path.Join(tmpDir, "stream-1.log"))
			if err != nil {
				b.Fatal(err)
			}
			defer out1.Close()
			out2, err = os.Create(path.Join(tmpDir, "stream-2.log"))
			if err != nil {
				b.Fatal(err)
			}
			defer out2.Close()
		}
		b.Run("single-stream", func(b *testing.B) {
			if discard {
				logger, flush = logsjson.NewJSONLogger(c.Verbosity, logsjson.AddNopSync(&output), nil, nil)
			} else {
				stderr := os.Stderr
				os.Stderr = out1
				defer func() {
					os.Stderr = stderr
				}()
				logger, flush = logsjson.Factory{}.Create(*c)
			}
			klog.SetLogger(logger)
			defer klog.ClearLogger()
			generateOutput(b, config, flush, out1)
		})

		b.Run("split-stream", func(b *testing.B) {
			if discard {
				logger, flush = logsjson.NewJSONLogger(c.Verbosity, logsjson.AddNopSync(&output), logsjson.AddNopSync(&output), nil)
			} else {
				stdout, stderr := os.Stdout, os.Stderr
				os.Stdout, os.Stderr = out1, out2
				defer func() {
					os.Stdout, os.Stderr = stdout, stderr
				}()
				c := logsapi.NewLoggingConfiguration()
				c.Options.JSON.SplitStream = true
				logger, flush = logsjson.Factory{}.Create(*c)
			}
			klog.SetLogger(logger)
			defer klog.ClearLogger()
			generateOutput(b, config, flush, out1, out2)
		})
	})
}

func generateOutput(b *testing.B, config loadGeneratorConfig, flush func(), files ...*os.File) {
	msg := strings.Repeat("X", config.messageLength)
	err := errors.New("fail")
	start := time.Now()

	// Scale by 1000 because "go test -bench" starts with b.N == 1, which is very low.
	n := b.N * 1000

	b.ResetTimer()
	var wg sync.WaitGroup
	for i := 0; i < config.workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			acc := 0.0
			for i := 0; i < n; i++ {
				if acc > 100 {
					klog.ErrorS(err, msg, "key", "value")
					acc -= 100
				} else {
					klog.InfoS(msg, "key", "value")
				}
				acc += config.errorPercentage
			}
		}()
	}
	wg.Wait()
	klog.Flush()
	if flush != nil {
		flush()
	}
	b.StopTimer()

	// Print some information about the result.
	end := time.Now()
	duration := end.Sub(start)
	total := n * config.workers
	b.Logf("Wrote %d log entries in %s -> %.1f/s", total, duration, float64(total)/duration.Seconds())
	for i, file := range files {
		if file != nil {
			pos, err := file.Seek(0, os.SEEK_END)
			if err != nil {
				b.Fatal(err)
			}
			if _, err := file.Seek(0, os.SEEK_SET); err != nil {
				b.Fatal(err)
			}
			max := 50
			buffer := make([]byte, max)
			actual, err := file.Read(buffer)
			if err != nil {
				if err != io.EOF {
					b.Fatal(err)
				}
				buffer = nil
			}
			if actual == max {
				buffer[max-3] = '.'
				buffer[max-2] = '.'
				buffer[max-1] = '.'
			}
			b.Logf("      %d bytes to file #%d -> %.1fMiB/s (starts with: %s)", pos, i, float64(pos)/duration.Seconds()/1024/1024, string(buffer))
		}
	}
}
