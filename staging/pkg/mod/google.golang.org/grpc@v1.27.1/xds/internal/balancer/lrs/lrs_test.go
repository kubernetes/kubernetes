/*
 *
 * Copyright 2019 gRPC authors.
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
 */

package lrs

import (
	"context"
	"fmt"
	"io"
	"net"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	endpointpb "github.com/envoyproxy/go-control-plane/envoy/api/v2/endpoint"
	lrsgrpc "github.com/envoyproxy/go-control-plane/envoy/service/load_stats/v2"
	lrspb "github.com/envoyproxy/go-control-plane/envoy/service/load_stats/v2"
	"github.com/golang/protobuf/proto"
	durationpb "github.com/golang/protobuf/ptypes/duration"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/xds/internal"
)

const testService = "grpc.service.test"

var (
	dropCategories = []string{"drop_for_real", "drop_for_fun"}
	localities     = []internal.Locality{{Region: "a"}, {Region: "b"}}
	errTest        = fmt.Errorf("test error")
)

type rpcCountDataForTest struct {
	succeeded   uint64
	errored     uint64
	inProgress  uint64
	serverLoads map[string]float64
}

func newRPCCountDataForTest(succeeded, errored, inprogress uint64, serverLoads map[string]float64) *rpcCountDataForTest {
	return &rpcCountDataForTest{
		succeeded:   succeeded,
		errored:     errored,
		inProgress:  inprogress,
		serverLoads: serverLoads,
	}
}

// Equal() is needed to compare unexported fields.
func (rcd *rpcCountDataForTest) Equal(b *rpcCountDataForTest) bool {
	return rcd.inProgress == b.inProgress &&
		rcd.errored == b.errored &&
		rcd.succeeded == b.succeeded &&
		reflect.DeepEqual(rcd.serverLoads, b.serverLoads)
}

// equalClusterStats sorts requests and clear report internal before comparing.
func equalClusterStats(a, b []*endpointpb.ClusterStats) bool {
	for _, t := range [][]*endpointpb.ClusterStats{a, b} {
		for _, s := range t {
			sort.Slice(s.DroppedRequests, func(i, j int) bool {
				return s.DroppedRequests[i].Category < s.DroppedRequests[j].Category
			})
			sort.Slice(s.UpstreamLocalityStats, func(i, j int) bool {
				return s.UpstreamLocalityStats[i].Locality.String() < s.UpstreamLocalityStats[j].Locality.String()
			})
			for _, us := range s.UpstreamLocalityStats {
				sort.Slice(us.LoadMetricStats, func(i, j int) bool {
					return us.LoadMetricStats[i].MetricName < us.LoadMetricStats[j].MetricName
				})
			}
			s.LoadReportInterval = nil
		}
	}
	return cmp.Equal(a, b, cmp.Comparer(proto.Equal))
}

func Test_lrsStore_buildStats_drops(t *testing.T) {
	tests := []struct {
		name  string
		drops []map[string]uint64
	}{
		{
			name: "one drop report",
			drops: []map[string]uint64{{
				dropCategories[0]: 31,
				dropCategories[1]: 41,
			}},
		},
		{
			name: "two drop reports",
			drops: []map[string]uint64{{
				dropCategories[0]: 31,
				dropCategories[1]: 41,
			}, {
				dropCategories[0]: 59,
				dropCategories[1]: 26,
			}},
		},
		{
			name: "no empty report",
			drops: []map[string]uint64{{
				dropCategories[0]: 31,
				dropCategories[1]: 41,
			}, {
				dropCategories[0]: 0, // This shouldn't cause an empty report for category[0].
				dropCategories[1]: 26,
			}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ls := NewStore().(*lrsStore)

			for _, ds := range tt.drops {
				var (
					totalDropped uint64
					droppedReqs  []*endpointpb.ClusterStats_DroppedRequests
				)
				for cat, count := range ds {
					if count == 0 {
						continue
					}
					totalDropped += count
					droppedReqs = append(droppedReqs, &endpointpb.ClusterStats_DroppedRequests{
						Category:     cat,
						DroppedCount: count,
					})
				}
				want := []*endpointpb.ClusterStats{
					{
						ClusterName:          testService,
						TotalDroppedRequests: totalDropped,
						DroppedRequests:      droppedReqs,
					},
				}

				var wg sync.WaitGroup
				for c, count := range ds {
					for i := 0; i < int(count); i++ {
						wg.Add(1)
						go func(i int, c string) {
							ls.CallDropped(c)
							wg.Done()
						}(i, c)
					}
				}
				wg.Wait()

				if got := ls.buildStats(testService); !equalClusterStats(got, want) {
					t.Errorf("lrsStore.buildStats() = %v, want %v", got, want)
					t.Errorf("%s", cmp.Diff(got, want))
				}
			}
		})
	}
}

func Test_lrsStore_buildStats_rpcCounts(t *testing.T) {
	tests := []struct {
		name string
		rpcs []map[internal.Locality]struct {
			start, success, failure uint64
			serverData              map[string]float64 // Will be reported with successful RPCs.
		}
	}{
		{
			name: "one rpcCount report",
			rpcs: []map[internal.Locality]struct {
				start, success, failure uint64
				serverData              map[string]float64
			}{{
				localities[0]: {8, 3, 1, nil},
			}},
		},
		{
			name: "two localities one rpcCount report",
			rpcs: []map[internal.Locality]struct {
				start, success, failure uint64
				serverData              map[string]float64
			}{{
				localities[0]: {8, 3, 1, nil},
				localities[1]: {15, 1, 5, nil},
			}},
		},
		{
			name: "three rpcCount reports",
			rpcs: []map[internal.Locality]struct {
				start, success, failure uint64
				serverData              map[string]float64
			}{{
				localities[0]: {8, 3, 1, nil},
				localities[1]: {15, 1, 5, nil},
			}, {
				localities[0]: {8, 3, 1, nil},
			}, {
				localities[1]: {15, 1, 5, nil},
			}},
		},
		{
			name: "no empty report",
			rpcs: []map[internal.Locality]struct {
				start, success, failure uint64
				serverData              map[string]float64
			}{{
				localities[0]: {4, 3, 1, nil},
				localities[1]: {7, 1, 5, nil},
			}, {
				localities[0]: {0, 0, 0, nil}, // This shouldn't cause an empty report for locality[0].
				localities[1]: {1, 1, 0, nil},
			}},
		},
		{
			name: "two localities one report with server loads",
			rpcs: []map[internal.Locality]struct {
				start, success, failure uint64
				serverData              map[string]float64
			}{{
				localities[0]: {8, 3, 1, map[string]float64{"cpu": 15, "mem": 20}},
				localities[1]: {15, 4, 5, map[string]float64{"net": 5, "disk": 0.8}},
			}},
		},
		{
			name: "three reports with server loads",
			rpcs: []map[internal.Locality]struct {
				start, success, failure uint64
				serverData              map[string]float64
			}{{
				localities[0]: {8, 3, 1, map[string]float64{"cpu": 15, "mem": 20}},
				localities[1]: {15, 4, 5, map[string]float64{"net": 5, "disk": 0.8}},
			}, {
				localities[0]: {8, 3, 1, map[string]float64{"cpu": 1, "mem": 2}},
			}, {
				localities[1]: {15, 4, 5, map[string]float64{"net": 13, "disk": 1.4}},
			}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ls := NewStore().(*lrsStore)

			// InProgress count doesn't get cleared at each buildStats, keep
			// them to carry over.
			inProgressCounts := make(map[internal.Locality]uint64)

			for _, counts := range tt.rpcs {
				var upstreamLocalityStats []*endpointpb.UpstreamLocalityStats

				for l, count := range counts {
					tempInProgress := count.start - count.success - count.failure + inProgressCounts[l]
					inProgressCounts[l] = tempInProgress
					if count.success == 0 && tempInProgress == 0 && count.failure == 0 {
						continue
					}
					var loadMetricStats []*endpointpb.EndpointLoadMetricStats
					for n, d := range count.serverData {
						loadMetricStats = append(loadMetricStats,
							&endpointpb.EndpointLoadMetricStats{
								MetricName:                    n,
								NumRequestsFinishedWithMetric: count.success,
								TotalMetricValue:              d * float64(count.success),
							},
						)
					}
					upstreamLocalityStats = append(upstreamLocalityStats, &endpointpb.UpstreamLocalityStats{
						Locality:                l.ToProto(),
						TotalSuccessfulRequests: count.success,
						TotalRequestsInProgress: tempInProgress,
						TotalErrorRequests:      count.failure,
						LoadMetricStats:         loadMetricStats,
					})
				}
				// InProgress count doesn't get cleared at each buildStats, and
				// needs to be carried over to the next result.
				for l, c := range inProgressCounts {
					if _, ok := counts[l]; !ok {
						upstreamLocalityStats = append(upstreamLocalityStats, &endpointpb.UpstreamLocalityStats{
							Locality:                l.ToProto(),
							TotalRequestsInProgress: c,
						})
					}
				}
				want := []*endpointpb.ClusterStats{
					{
						ClusterName:           testService,
						UpstreamLocalityStats: upstreamLocalityStats,
					},
				}

				var wg sync.WaitGroup
				for l, count := range counts {
					for i := 0; i < int(count.success); i++ {
						wg.Add(1)
						go func(l internal.Locality, serverData map[string]float64) {
							ls.CallStarted(l)
							ls.CallFinished(l, nil)
							for n, d := range serverData {
								ls.CallServerLoad(l, n, d)
							}
							wg.Done()
						}(l, count.serverData)
					}
					for i := 0; i < int(count.failure); i++ {
						wg.Add(1)
						go func(l internal.Locality) {
							ls.CallStarted(l)
							ls.CallFinished(l, errTest)
							wg.Done()
						}(l)
					}
					for i := 0; i < int(count.start-count.success-count.failure); i++ {
						wg.Add(1)
						go func(l internal.Locality) {
							ls.CallStarted(l)
							wg.Done()
						}(l)
					}
				}
				wg.Wait()

				if got := ls.buildStats(testService); !equalClusterStats(got, want) {
					t.Errorf("lrsStore.buildStats() = %v, want %v", got, want)
					t.Errorf("%s", cmp.Diff(got, want))
				}
			}
		})
	}
}

type lrsServer struct {
	reportingInterval *durationpb.Duration

	mu        sync.Mutex
	dropTotal uint64
	drops     map[string]uint64
	rpcs      map[internal.Locality]*rpcCountDataForTest
}

func (lrss *lrsServer) StreamLoadStats(stream lrsgrpc.LoadReportingService_StreamLoadStatsServer) error {
	req, err := stream.Recv()
	if err != nil {
		return err
	}
	if !proto.Equal(req, &lrspb.LoadStatsRequest{
		ClusterStats: []*endpointpb.ClusterStats{{
			ClusterName: testService,
		}},
	}) {
		return status.Errorf(codes.FailedPrecondition, "unexpected req: %+v", req)
	}
	if err := stream.Send(&lrspb.LoadStatsResponse{
		Clusters:              []string{testService},
		LoadReportingInterval: lrss.reportingInterval,
	}); err != nil {
		return err
	}

	for {
		req, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		stats := req.ClusterStats[0]
		lrss.mu.Lock()
		lrss.dropTotal += stats.TotalDroppedRequests
		for _, d := range stats.DroppedRequests {
			lrss.drops[d.Category] += d.DroppedCount
		}
		for _, ss := range stats.UpstreamLocalityStats {
			l := internal.Locality{
				Region:  ss.Locality.Region,
				Zone:    ss.Locality.Zone,
				SubZone: ss.Locality.SubZone,
			}
			counts, ok := lrss.rpcs[l]
			if !ok {
				counts = newRPCCountDataForTest(0, 0, 0, nil)
				lrss.rpcs[l] = counts
			}
			counts.succeeded += ss.TotalSuccessfulRequests
			counts.inProgress = ss.TotalRequestsInProgress
			counts.errored += ss.TotalErrorRequests
			for _, ts := range ss.LoadMetricStats {
				if counts.serverLoads == nil {
					counts.serverLoads = make(map[string]float64)
				}
				counts.serverLoads[ts.MetricName] = ts.TotalMetricValue / float64(ts.NumRequestsFinishedWithMetric)
			}
		}
		lrss.mu.Unlock()
	}
}

func setupServer(t *testing.T, reportingInterval *durationpb.Duration) (addr string, lrss *lrsServer, cleanup func()) {
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("listen failed due to: %v", err)
	}
	svr := grpc.NewServer()
	lrss = &lrsServer{
		reportingInterval: reportingInterval,
		drops:             make(map[string]uint64),
		rpcs:              make(map[internal.Locality]*rpcCountDataForTest),
	}
	lrsgrpc.RegisterLoadReportingServiceServer(svr, lrss)
	go svr.Serve(lis)
	return lis.Addr().String(), lrss, func() {
		svr.Stop()
		lis.Close()
	}
}

func Test_lrsStore_ReportTo(t *testing.T) {
	const intervalNano = 1000 * 1000 * 50
	addr, lrss, cleanup := setupServer(t, &durationpb.Duration{
		Seconds: 0,
		Nanos:   intervalNano,
	})
	defer cleanup()

	ls := NewStore()
	cc, err := grpc.Dial(addr, grpc.WithInsecure())
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	done := make(chan struct{})
	go func() {
		ls.ReportTo(ctx, cc, testService, nil)
		close(done)
	}()

	drops := map[string]uint64{
		dropCategories[0]: 13,
		dropCategories[1]: 14,
	}
	for c, d := range drops {
		for i := 0; i < int(d); i++ {
			ls.CallDropped(c)
			time.Sleep(time.Nanosecond * intervalNano / 10)
		}
	}

	rpcs := map[internal.Locality]*rpcCountDataForTest{
		localities[0]: newRPCCountDataForTest(3, 1, 4, nil),
		localities[1]: newRPCCountDataForTest(1, 5, 9, map[string]float64{"pi": 3.14, "e": 2.71}),
	}
	for l, count := range rpcs {
		for i := 0; i < int(count.succeeded); i++ {
			go func(i int, l internal.Locality, count *rpcCountDataForTest) {
				ls.CallStarted(l)
				ls.CallFinished(l, nil)
				for n, d := range count.serverLoads {
					ls.CallServerLoad(l, n, d)
				}
			}(i, l, count)
		}
		for i := 0; i < int(count.inProgress); i++ {
			go func(i int, l internal.Locality) {
				ls.CallStarted(l)
			}(i, l)
		}
		for i := 0; i < int(count.errored); i++ {
			go func(i int, l internal.Locality) {
				ls.CallStarted(l)
				ls.CallFinished(l, errTest)
			}(i, l)
		}
	}

	time.Sleep(time.Nanosecond * intervalNano * 2)
	cancel()
	<-done

	lrss.mu.Lock()
	defer lrss.mu.Unlock()
	if !cmp.Equal(lrss.drops, drops) {
		t.Errorf("different: %v", cmp.Diff(lrss.drops, drops))
	}
	if !cmp.Equal(lrss.rpcs, rpcs) {
		t.Errorf("different: %v", cmp.Diff(lrss.rpcs, rpcs))
	}
}
