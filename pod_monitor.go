
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

func main() {
	// This example demonstrates how to monitor Kubernetes pods and their status.
	// It can be used to observe behaviors like crashloop backoff, though it won't
	// directly fix the underlying Kubernetes bug described in issue #133472.

	// Path to your kubeconfig file (e.g., ~/.kube/config)
	// In a real application, consider using in-cluster config if running inside a cluster.
	kubeconfigPath := clientcmd.RecommendedHomeFile

	// Build config from kubeconfig file
	config, err := clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	if err != nil {
		log.Fatalf("Error building kubeconfig: %v", err)
	}

	// Create a Kubernetes clientset
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		log.Fatalf("Error creating clientset: %v", err)
	}

	fmt.Println("Monitoring pods... Press Ctrl+C to stop.")

	for {
		pods, err := clientset.CoreV1().Pods("").List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			log.Printf("Error listing pods: %v", err)
		} else {
			fmt.Printf("\n--- Pod Status (%s) ---\n", time.Now().Format("15:04:05"))
			if len(pods.Items) == 0 {
				fmt.Println("No pods found.")
			} else {
				for _, pod := range pods.Items {
					fmt.Printf("Name: %s, Namespace: %s, Status: %s\n", pod.Name, pod.Namespace, pod.Status.Phase)

					// Check container statuses for more details, e.g., CrashLoopBackOff
					for _, containerStatus := range pod.Status.ContainerStatuses {
						if containerStatus.State.Waiting != nil && containerStatus.State.Waiting.Reason == "CrashLoopBackOff" {
							fmt.Printf("  Container %s is in CrashLoopBackOff: %s\n", containerStatus.Name, containerStatus.State.Waiting.Message)
						} else if containerStatus.State.Terminated != nil && containerStatus.State.Terminated.ExitCode != 0 {
							fmt.Printf("  Container %s terminated with exit code %d: %s\n", containerStatus.Name, containerStatus.State.Terminated.ExitCode, containerStatus.State.Terminated.Reason)
						}
					}
				}
			}
		}

		time.Sleep(5 * time.Second) // Poll every 5 seconds
	}
}


