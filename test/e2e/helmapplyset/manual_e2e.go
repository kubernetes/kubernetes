//go:build ignore

/*
Manual E2E test for Helm ApplySet Integration.
Run with: go run test/e2e/helmapplyset/manual_test.go

Prerequisites:
- minikube running
- Helm chart installed: helm install test-nginx oci://registry-1.docker.io/bitnamicharts/nginx
*/

package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"k8s.io/kubernetes/pkg/controller/helmapplyset"
	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
)

func main() {
	ctx := context.Background()

	// Build kubeconfig
	home, _ := os.UserHomeDir()
	kubeconfig := filepath.Join(home, ".kube", "config")

	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		fmt.Printf("❌ Failed to build kubeconfig: %v\n", err)
		os.Exit(1)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		fmt.Printf("❌ Failed to create clientset: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("🔍 Helm ApplySet Integration E2E Test")
	fmt.Println("=====================================")

	// Step 1: Find Helm release secrets
	fmt.Println("\n📦 Step 1: Finding Helm release secrets...")
	secrets, err := clientset.CoreV1().Secrets("default").List(ctx, metav1.ListOptions{
		LabelSelector: "owner=helm",
	})
	if err != nil {
		fmt.Printf("❌ Failed to list secrets: %v\n", err)
		os.Exit(1)
	}

	if len(secrets.Items) == 0 {
		fmt.Println("❌ No Helm release secrets found. Install a chart first:")
		fmt.Println("   helm install test-nginx oci://registry-1.docker.io/bitnamicharts/nginx")
		os.Exit(1)
	}

	fmt.Printf("✅ Found %d Helm release secret(s)\n", len(secrets.Items))

	for _, secret := range secrets.Items {
		fmt.Printf("\n📋 Processing: %s\n", secret.Name)
		fmt.Printf("   Type: %s\n", secret.Type)

		// Step 2: Verify it's a valid Helm release secret
		fmt.Println("\n📦 Step 2: Validating Helm release secret format...")
		if !helmapplyset.IsHelmReleaseSecret(&secret) {
			fmt.Printf("⚠️  Secret %s is not a valid Helm release secret\n", secret.Name)
			continue
		}
		fmt.Println("✅ Valid Helm release secret format")

		// Step 3: Parse the Helm release
		fmt.Println("\n📦 Step 3: Parsing Helm release data...")
		release, err := helmapplyset.ParseHelmReleaseSecret(&secret)
		if err != nil {
			fmt.Printf("❌ Failed to parse release: %v\n", err)
			continue
		}

		fmt.Printf("✅ Parsed Helm release:\n")
		fmt.Printf("   Name: %s\n", release.Name)
		fmt.Printf("   Namespace: %s\n", release.Namespace)
		fmt.Printf("   Version: %d\n", release.Version)
		fmt.Printf("   Status: %s\n", release.Status)
		fmt.Printf("   Chart: %s (version %s)\n", release.Chart, release.ChartVersion)

		// Step 4: Compute ApplySet ID
		fmt.Println("\n📦 Step 4: Computing ApplySet ID...")
		applySetID := parent.ComputeApplySetID(release.Name, release.Namespace)
		fmt.Printf("✅ ApplySet ID: %s\n", applySetID)

		// Step 5: Show what ApplySet parent would look like
		fmt.Println("\n📦 Step 5: ApplySet Parent Secret (would be created):")
		fmt.Printf("   Name: applyset-%s\n", release.Name)
		fmt.Printf("   Namespace: %s\n", release.Namespace)
		fmt.Printf("   Labels:\n")
		fmt.Printf("     applyset.kubernetes.io/id: %s\n", applySetID)
		fmt.Printf("   Annotations:\n")
		fmt.Printf("     applyset.kubernetes.io/tooling: helm/v3\n")

		// Step 6: Parse manifest to find resources
		fmt.Println("\n📦 Step 6: Resources in Helm release manifest:")
		if release.Manifest != "" {
			// Just show first 500 chars of manifest
			manifest := release.Manifest
			if len(manifest) > 500 {
				manifest = manifest[:500] + "..."
			}
			fmt.Printf("   Manifest preview:\n%s\n", manifest)
		}

		// Show raw release data size
		if releaseData, ok := secret.Data["release"]; ok {
			decoded, _ := base64.StdEncoding.DecodeString(string(releaseData))
			fmt.Printf("\n📊 Release data size: %d bytes (compressed: %d bytes)\n",
				len(releaseData), len(decoded))
		}
	}

	fmt.Println("\n=====================================")
	fmt.Println("✅ E2E Test Complete!")
	fmt.Println("\nNext steps to fully test:")
	fmt.Println("1. Deploy the controller to the cluster")
	fmt.Println("2. Verify ApplySet parent secrets are created")
	fmt.Println("3. Verify resources are labeled with applyset.kubernetes.io/part-of")
}
