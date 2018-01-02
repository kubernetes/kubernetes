// Copyright 2015 The rkt Authors
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

// +build host coreos src kvm

package main

import (
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/api/v1alpha"
	"github.com/coreos/rkt/tests/testutils"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

func setup() (*testutils.RktRunCtx, *gexpect.ExpectSubprocess, v1alpha.PublicAPIClient, *grpc.ClientConn, string) {
	t := new(testing.T) // Print no messages.
	ctx := testutils.NewRktRunCtx()
	svc := startAPIService(t, ctx)
	c, conn := newAPIClientOrFail(t, "localhost:15441")
	imagePath := patchTestACI("rkt-inspect-print.aci", "--exec=/inspect --print-msg=HELLO_API")

	return ctx, svc, c, conn, imagePath
}

func cleanup(ctx *testutils.RktRunCtx, svc *gexpect.ExpectSubprocess, conn *grpc.ClientConn, imagePath string) {
	t := new(testing.T) // Print no messages.
	os.Remove(imagePath)
	conn.Close()
	stopAPIService(t, svc)
	ctx.Cleanup()
}

func launchPods(ctx *testutils.RktRunCtx, numOfPods int, imagePath string) {
	t := new(testing.T) // Print no messages.
	cmd := fmt.Sprintf("%s --insecure-options=all run %s", ctx.Cmd(), imagePath)

	var wg sync.WaitGroup
	wg.Add(numOfPods)
	for i := 0; i < numOfPods; i++ {
		go func() {
			spawnAndWaitOrFail(t, cmd, 0)
			wg.Done()
		}()
	}
	wg.Wait()
}

func benchListPods(b *testing.B, c v1alpha.PublicAPIClient, detail bool) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := c.ListPods(context.Background(), &v1alpha.ListPodsRequest{Detail: detail})
		if err != nil {
			b.Error(err)
		}
	}
	b.StopTimer()
}

func benchInspectPod(b *testing.B, c v1alpha.PublicAPIClient) {
	resp, err := c.ListPods(context.Background(), &v1alpha.ListPodsRequest{})
	if err != nil {
		b.Fatalf("Unexpected error: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := c.InspectPod(context.Background(), &v1alpha.InspectPodRequest{Id: resp.Pods[0].Id})
		if err != nil {
			b.Error(err)
		}
	}
	b.StopTimer()
}

func fetchImages(ctx *testutils.RktRunCtx, numOfImages int) {
	t := new(testing.T) // Print no messages.

	var wg sync.WaitGroup
	wg.Add(numOfImages)
	for i := 0; i < numOfImages; i++ {
		go func(i int) {
			_, err := patchImportAndFetchHash(fmt.Sprintf("rkt-inspect-sleep-%d.aci", i), []string{"--exec=/inspect"}, t, ctx)
			if err != nil {
				t.Fatalf("%v", err)
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
}

func benchListImages(b *testing.B, c v1alpha.PublicAPIClient, detail bool) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := c.ListImages(context.Background(), &v1alpha.ListImagesRequest{Detail: detail})
		if err != nil {
			b.Error(err)
		}
	}
	b.StopTimer()
}

func benchInspectImage(b *testing.B, c v1alpha.PublicAPIClient) {
	resp, err := c.ListImages(context.Background(), &v1alpha.ListImagesRequest{})
	if err != nil {
		b.Fatalf("Unexpected error: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := c.InspectImage(context.Background(), &v1alpha.InspectImageRequest{Id: resp.Images[0].Id})
		if err != nil {
			b.Error(err)
		}
	}
	b.StopTimer()
}

func BenchmarkList1PodNoDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	launchPods(ctx, 1, imagePath)
	benchListPods(b, client, false)
}

func BenchmarkList10PodsNoDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	launchPods(ctx, 10, imagePath)
	benchListPods(b, client, false)
}

func BenchmarkList100PodsNoDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	launchPods(ctx, 100, imagePath)
	benchListPods(b, client, false)
}

func BenchmarkList1PodDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	launchPods(ctx, 1, imagePath)
	benchListPods(b, client, true)
}

func BenchmarkList10PodsDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	launchPods(ctx, 10, imagePath)
	benchListPods(b, client, true)
}

func BenchmarkList100PodsDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	launchPods(ctx, 100, imagePath)
	benchListPods(b, client, true)
}

func BenchmarkInspectPodIn10Pods(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	launchPods(ctx, 10, imagePath)
	benchInspectPod(b, client)
}

func BenchmarkInspectPodIn100Pods(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	launchPods(ctx, 100, imagePath)
	benchInspectPod(b, client)
}

func BenchmarkList1ImageNoDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	fetchImages(ctx, 1)
	benchListImages(b, client, false)
}

func BenchmarkList10ImagesNoDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	fetchImages(ctx, 10)
	benchListImages(b, client, false)
}

func BenchmarkList100ImagesNoDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	fetchImages(ctx, 100)
	benchListImages(b, client, false)
}

func BenchmarkList1ImageDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	fetchImages(ctx, 1)
	benchListImages(b, client, true)
}

func BenchmarkList10ImagesDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	fetchImages(ctx, 10)
	benchListImages(b, client, true)
}

func BenchmarkList100ImagesDetail(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	fetchImages(ctx, 100)
	benchListImages(b, client, true)
}

func BenchmarkInspectImageIn10Images(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	fetchImages(ctx, 10)
	benchInspectImage(b, client)
}

func BenchmarkInspectImageIn100Images(b *testing.B) {
	ctx, svc, client, conn, imagePath := setup()
	defer cleanup(ctx, svc, conn, imagePath)

	fetchImages(ctx, 100)
	benchInspectImage(b, client)
}
