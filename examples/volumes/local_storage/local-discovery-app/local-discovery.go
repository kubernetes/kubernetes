package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/rest"
)

func generatePVName(file, node string) string {
	return fmt.Sprintf("%v-%v", node, file)
}

func pvExists(existingPVs map[string]bool, file, node string) bool {
	value, exists := existingPVs[generatePVName(file, node)]
	return exists && value
}

func populateExistingPVs(c *kubernetes.Clientset, node string) (map[string]bool, error) {
	pvs, err := c.Core().PersistentVolumes().List(metav1.ListOptions{})
	if err != nil {
		fmt.Printf("Error listing PVs: %v\n", err.Error())
		return nil, err
	}

	existingPVMap := make(map[string]bool)
	for _, pv := range pvs.Items {
		localPV := pv.Spec.PersistentVolumeSource.LocalStorage
		if localPV != nil && localPV.NodeName == node {
			existingPVMap[pv.Name] = true
		}
	}
	return existingPVMap, nil
}

func getFullPath(file string) string {
	return fmt.Sprintf("/mnt/disks/%v", file)
}

func createPV(c *kubernetes.Clientset, file, node string) {
	pvName := generatePVName(file, node)
	pvSpec := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: pvName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("10Gi"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				LocalStorage: &v1.LocalStorageVolumeSource{
					Path:     getFullPath(file),
					NodeName: node,
				},
			},
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			StorageClassName: "local-fast",
		},
	}

	_, err := c.Core().PersistentVolumes().Create(pvSpec)
	if err != nil {
		fmt.Printf("Error creating PV %v: %v\n", pvName, err.Error())
	}
}

func processLocalSSDs(c *kubernetes.Clientset, existingPVs map[string]bool, node string) {
	files, err := ioutil.ReadDir("/local-disks/mnt/disks")
	if err != nil {
		fmt.Printf("Error reading directory: %v\n", err.Error())
		return
	}

	for _, fileInfo := range files {
		// Check if PV already exists for it
		if !pvExists(existingPVs, fileInfo.Name(), node) {
			// If not, create PV
			createPV(c, fileInfo.Name(), node)
		}
	}
}

func setupClient() *kubernetes.Clientset {
	config, err := rest.InClusterConfig()
	if err != nil {
		fmt.Printf("Error creating InCluster config: %v\n", err.Error())
		os.Exit(1)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		fmt.Printf("Error creating InCluster config: %v\n", err.Error())
		os.Exit(1)
	}
	return clientset
}

func main() {
	node := os.Getenv("MY_NODE_NAME")
	if node == "" {
		fmt.Printf("MY_NODE_NAME environment variable not set\n")
		os.Exit(1)
	}
	// Wait for kubectl proxy
	time.Sleep(5 * time.Second)
	// Setup client
	client := setupClient()
	for {
		existingPVMap, err := populateExistingPVs(client, node)
		if err == nil {
			processLocalSSDs(client, existingPVMap, node)
		}
		time.Sleep(30 * time.Second)
	}
}
