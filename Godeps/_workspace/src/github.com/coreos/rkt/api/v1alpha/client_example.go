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
	"fmt"
	"os"

	"github.com/coreos/rkt/api/v1alpha"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

func main() {
	conn, err := grpc.Dial("localhost:15441", grpc.WithInsecure())
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
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
		os.Exit(2)
	}

	for _, p := range podResp.Pods {
		fmt.Printf("Pod %q is running\n", p.Id)
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
		os.Exit(3)
	}

	for _, im := range imgResp.Images {
		fmt.Printf("Found image %q\n", im.Name)
	}
}
