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
	"fmt"
	"io/fs"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"k8s.io/klog/v2"
)

func BenchmarkLogging(b *testing.B) {
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
