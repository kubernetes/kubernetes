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

// +build ignore

package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/coreos/rkt/api/v1alpha"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

func getLogsWithoutFollow(c v1alpha.PublicAPIClient, p *v1alpha.Pod) {
	if len(p.Apps) == 0 {
		fmt.Printf("Pod %q has no apps\n", p.Id)
		return
	}

	logsResp, err := c.GetLogs(context.Background(), &v1alpha.GetLogsRequest{
		PodId:     p.Id,
		Follow:    false,
		AppName:   p.Apps[0].Name,
		SinceTime: time.Now().Add(-time.Second * 5).Unix(),
		Lines:     10,
	})

	if err != nil {
		fmt.Println(err)
		os.Exit(254)
	}

	logsRecvResp, err := logsResp.Recv()

	if err == io.EOF {
		return
	}

	if err != nil {
		fmt.Println(err)
		return
	}

	for _, l := range logsRecvResp.Lines {
		fmt.Println(l)
	}
}

func getLogsWithFollow(c v1alpha.PublicAPIClient, p *v1alpha.Pod) {
	if len(p.Apps) == 0 {
		fmt.Printf("Pod %q has no apps\n", p.Id)
		return
	}

	logsResp, err := c.GetLogs(context.Background(), &v1alpha.GetLogsRequest{
		PodId:   p.Id,
		Follow:  true,
		AppName: p.Apps[0].Name,
	})
	if err != nil {
		fmt.Println(err)
		os.Exit(254)
	}

	for {
		logsRecvResp, err := logsResp.Recv()
		if err == io.EOF {
			return
		}

		if err != nil {
			fmt.Println(err)
			return
		}

		for _, l := range logsRecvResp.Lines {
			fmt.Println(l)
		}
	}
}

func main() {
	followFlag := flag.Bool("follow", false, "enable 'follow' option on GetLogs")
	flag.Parse()

	conn, err := grpc.Dial("localhost:15441", grpc.WithInsecure())
	if err != nil {
		fmt.Println(err)
		os.Exit(254)
	}
	c := v1alpha.NewPublicAPIClient(conn)
	defer conn.Close()

	// List pods.
	podResp, err := c.ListPods(context.Background(), &v1alpha.ListPodsRequest{
		// Specify the request: Fetch and print only running pods and their details.
		Detail: true,
		Filters: []*v1alpha.PodFilter{
			{
				States: []v1alpha.PodState{v1alpha.PodState_POD_STATE_RUNNING},
			},
		},
	})
	if err != nil {
		fmt.Println(err)
		os.Exit(254)
	}

	for _, p := range podResp.Pods {
		if *followFlag {
			fmt.Printf("Pod %q is running. Following logs:\n", p.Id)
			getLogsWithFollow(c, p)
		} else {
			fmt.Printf("Pod %q is running.\n", p.Id)
			getLogsWithoutFollow(c, p)
		}
	}

	// List images.
	imgResp, err := c.ListImages(context.Background(), &v1alpha.ListImagesRequest{
		// In this request, we fetch the details of images whose names are prefixed with "coreos.com".
		Detail: true,
		Filters: []*v1alpha.ImageFilter{
			{
				Prefixes: []string{"coreos.com"},
			},
		},
	})
	if err != nil {
		fmt.Println(err)
		os.Exit(254)
	}

	for _, im := range imgResp.Images {
		fmt.Printf("Found image %q\n", im.Name)
	}
}
