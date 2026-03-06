/*
Copyright 2017 The Kubernetes Authors.

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
	"flag"
	"fmt"
	"math/big"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
	"k8s.io/sample-controller/pkg/signals"

	// Uncomment the following line to load the gcp plugin (only required to authenticate against GKE clusters).
	// _ "k8s.io/client-go/plugin/pkg/client/auth/gcp"

	clientset "k8s.io/sample-controller/pkg/generated/clientset/versioned"
	informers "k8s.io/sample-controller/pkg/generated/informers/externalversions"
)

var (
	masterURL  string
	kubeconfig string
	shard      int
	shardTotal int
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	// set up signals so we handle the shutdown signal gracefully
	ctx := signals.SetupSignalHandler()
	logger := klog.FromContext(ctx)

	var shardSelector string
	if shard != -1 && shardTotal > 0 {
		if shard >= shardTotal {
			logger.Error(nil, "shard index must be less than shard total", "shard", shard, "total", shardTotal)
			return
		}
		start, end := CalculateShardRange(shard, shardTotal)
		shardSelector = fmt.Sprintf("shardRange(object.metadata.uid,%s,%s)", start, end)
	}

	cfg, err := clientcmd.BuildConfigFromFlags(masterURL, kubeconfig)
	if err != nil {
		logger.Error(err, "Error building kubeconfig")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	kubeClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		logger.Error(err, "Error building kubernetes clientset")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	exampleClient, err := clientset.NewForConfig(cfg)
	if err != nil {
		logger.Error(err, "Error building kubernetes clientset")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	tweakListOptions := func(options *metav1.ListOptions) {
		if shardSelector != "" {
			options.Selector = shardSelector
		}
	}

	kubeInformerFactory := kubeinformers.NewSharedInformerFactoryWithOptions(kubeClient, time.Second*30, kubeinformers.WithTweakListOptions(tweakListOptions))
	exampleInformerFactory := informers.NewSharedInformerFactoryWithOptions(exampleClient, time.Second*30, informers.WithTweakListOptions(tweakListOptions))

	controller := NewController(ctx, kubeClient, exampleClient,
		kubeInformerFactory.Apps().V1().Deployments(),
		exampleInformerFactory.Samplecontroller().V1alpha1().Foos())

	// notice that there is no need to run Start methods in a separate goroutine. (i.e. go kubeInformerFactory.Start(ctx.done())
	// Start method is non-blocking and runs all registered informers in a dedicated goroutine.
	kubeInformerFactory.Start(ctx.Done())
	exampleInformerFactory.Start(ctx.Done())

	if err = controller.Run(ctx, 2); err != nil {
		logger.Error(err, "Error running controller")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
}

func init() {
	flag.StringVar(&kubeconfig, "kubeconfig", "", "Path to a kubeconfig. Only required if out-of-cluster.")
	flag.StringVar(&masterURL, "master", "", "The address of the Kubernetes API server. Overrides any value in kubeconfig. Only required if out-of-cluster.")
	flag.IntVar(&shard, "shard", -1, "If set, the shard index to run. Must be used with --shard-total.")
	flag.IntVar(&shardTotal, "shard-total", 0, "If set, the total number of shards. Must be used with --shard.")
}

// CalculateShardRange computes the [start, end) 16-char hex prefixes for a given shard
// in the FNV-1a 64-bit hash space (0x0000000000000000 to 0xffffffffffffffff).
// index is 0-based (0 to total-1).
func CalculateShardRange(index, total int) (start, end string) {
	if total <= 1 {
		return "", "" // Match everything
	}

	// FNV-1a 64-bit: full range is [0, 2^64)
	maxVal := new(big.Int).Lsh(big.NewInt(1), 64) // 2^64

	span := new(big.Int).Div(maxVal, big.NewInt(int64(total)))
	startVal := new(big.Int).Mul(span, big.NewInt(int64(index)))
	endVal := new(big.Int).Mul(span, big.NewInt(int64(index+1)))

	if index == total-1 {
		end = "" // Unbounded upper
	} else {
		end = fmt.Sprintf("%016x", endVal)
	}

	if index == 0 {
		start = "" // Unbounded lower
	} else {
		start = fmt.Sprintf("%016x", startVal)
	}

	return start, end
}
