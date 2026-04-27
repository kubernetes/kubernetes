/*
Copyright 2025 The Kubernetes Authors.

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

// Package podcertificatesigner is an agnhost subcommand implementing a toy
// PodCertificateRequest signer.  It is meant to run continuously in an
// in-cluster pod.
package podcertificatesigner

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/component-base/logs"
	"k8s.io/kubernetes/test/utils/hermeticpodcertificatesigner"
	"k8s.io/utils/clock"
)

var CmdPodCertificateSigner = &cobra.Command{
	Use:   "podcertificatesigner",
	Short: "Sign PodCertificateRequests addressed to a given signer",
	Args:  cobra.MaximumNArgs(0),
	RunE:  run,
}

var kubeconfigPath string
var signerName string

func init() {
	CmdPodCertificateSigner.Flags().StringVar(&kubeconfigPath, "kubeconfig", "", "Path to kubeconfig file to use for connection.  If omitted, in-cluster config will be used.")
	CmdPodCertificateSigner.Flags().StringVar(&signerName, "signer-name", "", "The signer name to sign certificates for")
}

func run(cmd *cobra.Command, args []string) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	logs.InitLogs()
	defer logs.FlushLogs()

	cfg, err := clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	if err != nil {
		return fmt.Errorf("while building client config: %w", err)
	}

	kc, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return fmt.Errorf("while creating kubernetes client: %w", err)
	}

	caKeys, caCerts, err := hermeticpodcertificatesigner.GenerateCAHierarchy(1)
	if err != nil {
		return fmt.Errorf("while generating CA hierarchy: %w", err)
	}

	c := hermeticpodcertificatesigner.New(clock.RealClock{}, signerName, caKeys, caCerts, kc)
	go c.Run(ctx)

	// Wait for a shutdown signal.
	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, syscall.SIGINT, syscall.SIGTERM)
	<-signalCh

	// Canceling the context will begin exiting all of our controllers.
	cancel()

	return nil
}
