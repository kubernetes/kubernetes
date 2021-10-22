/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/*
To format the benchmark result:
  go run benchmark/benchresult/main.go resultfile

To see the performance change based on a old result:
  go run benchmark/benchresult/main.go resultfile_old resultfile
It will print the comparison result of intersection benchmarks between two files.

*/
package main

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc/benchmark/stats"
)

func createMap(fileName string) map[string]stats.BenchResults {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatalf("Read file %s error: %s\n", fileName, err)
	}
	defer f.Close()
	var data []stats.BenchResults
	decoder := gob.NewDecoder(f)
	if err = decoder.Decode(&data); err != nil {
		log.Fatalf("Decode file %s error: %s\n", fileName, err)
	}
	m := make(map[string]stats.BenchResults)
	for _, d := range data {
		m[d.RunMode+"-"+d.Features.String()] = d
	}
	return m
}

func intChange(title string, val1, val2 uint64) string {
	return fmt.Sprintf("%20s %12d %12d %8.2f%%\n", title, val1, val2, float64(int64(val2)-int64(val1))*100/float64(val1))
}

func floatChange(title string, val1, val2 float64) string {
	return fmt.Sprintf("%20s %12.2f %12.2f %8.2f%%\n", title, val1, val2, float64(int64(val2)-int64(val1))*100/float64(val1))
}
func timeChange(title string, val1, val2 time.Duration) string {
	return fmt.Sprintf("%20s %12s %12s %8.2f%%\n", title, val1.String(),
		val2.String(), float64(val2-val1)*100/float64(val1))
}

func strDiff(title, val1, val2 string) string {
	return fmt.Sprintf("%20s %12s %12s\n", title, val1, val2)
}

func compareTwoMap(m1, m2 map[string]stats.BenchResults) {
	for k2, v2 := range m2 {
		if v1, ok := m1[k2]; ok {
			changes := k2 + "\n"
			changes += fmt.Sprintf("%20s %12s %12s %8s\n", "Title", "Before", "After", "Percentage")
			changes += intChange("TotalOps", v1.Data.TotalOps, v2.Data.TotalOps)
			changes += intChange("SendOps", v1.Data.SendOps, v2.Data.SendOps)
			changes += intChange("RecvOps", v1.Data.RecvOps, v2.Data.RecvOps)
			changes += floatChange("Bytes/op", v1.Data.AllocedBytes, v2.Data.AllocedBytes)
			changes += floatChange("Allocs/op", v1.Data.Allocs, v2.Data.Allocs)
			changes += floatChange("ReqT/op", v1.Data.ReqT, v2.Data.ReqT)
			changes += floatChange("RespT/op", v1.Data.RespT, v2.Data.RespT)
			changes += timeChange("50th-Lat", v1.Data.Fiftieth, v2.Data.Fiftieth)
			changes += timeChange("90th-Lat", v1.Data.Ninetieth, v2.Data.Ninetieth)
			changes += timeChange("99th-Lat", v1.Data.NinetyNinth, v2.Data.NinetyNinth)
			changes += timeChange("Avg-Lat", v1.Data.Average, v2.Data.Average)
			changes += strDiff("GoVersion", v1.GoVersion, v2.GoVersion)
			changes += strDiff("GrpcVersion", v1.GrpcVersion, v2.GrpcVersion)
			fmt.Printf("%s\n", changes)
		}
	}
}

func compareBenchmark(file1, file2 string) {
	compareTwoMap(createMap(file1), createMap(file2))
}

func printHeader() {
	fmt.Printf("%-80s%12s%12s%12s%18s%18s%18s%18s%12s%12s%12s%12s\n",
		"Name", "TotalOps", "SendOps", "RecvOps", "Bytes/op (B)", "Allocs/op (#)",
		"RequestT", "ResponseT", "L-50", "L-90", "L-99", "L-Avg")
}

func printline(benchName string, d stats.RunData) {
	fmt.Printf("%-80s%12d%12d%12d%18.2f%18.2f%18.2f%18.2f%12v%12v%12v%12v\n",
		benchName, d.TotalOps, d.SendOps, d.RecvOps, d.AllocedBytes, d.Allocs,
		d.ReqT, d.RespT, d.Fiftieth, d.Ninetieth, d.NinetyNinth, d.Average)
}

func formatBenchmark(fileName string) {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatalf("Read file %s error: %s\n", fileName, err)
	}
	defer f.Close()
	var results []stats.BenchResults
	decoder := gob.NewDecoder(f)
	if err = decoder.Decode(&results); err != nil {
		log.Fatalf("Decode file %s error: %s\n", fileName, err)
	}
	if len(results) == 0 {
		log.Fatalf("No benchmark results in file %s\n", fileName)
	}

	fmt.Println("\nShared features:\n" + strings.Repeat("-", 20))
	fmt.Print(results[0].Features.SharedFeatures(results[0].SharedFeatures))
	fmt.Println(strings.Repeat("-", 35))

	wantFeatures := results[0].SharedFeatures
	for i := 0; i < len(results[0].SharedFeatures); i++ {
		wantFeatures[i] = !wantFeatures[i]
	}

	printHeader()
	for _, r := range results {
		printline(r.RunMode+r.Features.PrintableName(wantFeatures), r.Data)
	}
}

func main() {
	if len(os.Args) == 2 {
		formatBenchmark(os.Args[1])
	} else {
		compareBenchmark(os.Args[1], os.Args[2])
	}
}
