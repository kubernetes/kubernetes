package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/rest"
)

type pvMap map[string]*v1.PersistentVolume

// TODO: There is a name conflict if the same base file name is at two different paths
func generatePVName(file, node string) string {
	return fmt.Sprintf("%v-%v", node, file)
}

func pvExists(existingPVs pvMap, file, node string) bool {
	_, exists := existingPVs[generatePVName(file, node)]
	return exists
}

func deletePV(existingPVs pvMap, pvName string) {
	delete(existingPVs, pvName)
}

// TODO: make this an informer instead
func populateExistingPVs(c *kubernetes.Clientset, node string) (pvMap, error) {
	pvs, err := c.Core().PersistentVolumes().List(metav1.ListOptions{})
	if err != nil {
		fmt.Printf("Error listing PVs: %v\n", err.Error())
		return nil, err
	}

	existingPVMap := make(pvMap)
	for _, pv := range pvs.Items {
		localPV := pv.Spec.PersistentVolumeSource.LocalStorage
		if localPV != nil && localPV.NodeName == node {
			clone, err := api.Scheme.DeepCopy(&pv)
			if err != nil {
				fmt.Printf("Error cloning pv: %v\n", err)
				continue
			}
			pvClone, ok := clone.(*v1.PersistentVolume)
			if !ok {
				fmt.Printf("Error casting pv\n")
				continue
			}

			existingPVMap[pv.Name] = pvClone
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
			Annotations: map[string]string{
				"pv.kubernetes.io/provisioned-by": "local-storage-provisioner",
			},
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

	fmt.Printf("Found available local volume: %v\n", getFullPath(file))
	_, err := c.Core().PersistentVolumes().Create(pvSpec)
	if err != nil {
		fmt.Printf("Error creating PV %v: %v\n", pvName, err.Error())
	}
	fmt.Printf("Created PV: %v\n", pvName)
}

func processLocalSSDs(c *kubernetes.Clientset, existingPVs pvMap, node string) {
	dir, err := os.Open("/local-disks/mnt/disks")
	if err != nil {
		fmt.Printf("Error opening directory: %v\n", err.Error())
		return
	}
	defer dir.Close()

	files, err := dir.Readdirnames(-1)
	if err != nil {
		fmt.Printf("Error reading directory: %v\n", err.Error())
		return
	}

	for _, file := range files {
		// Check if PV already exists for it
		if !pvExists(existingPVs, file, node) {
			// If not, create PV
			createPV(c, file, node)
		}
	}
}

func deleteContents(path string) error {
	fullPath := filepath.Join("/local-disks", path)
	dir, err := os.Open(fullPath)
	if err != nil {
		return err
	}
	defer dir.Close()

	files, err := dir.Readdirnames(-1)
	if err != nil {
		return err
	}

	for _, file := range files {
		// TODO: investigate potential security implications
		err = os.RemoveAll(filepath.Join(fullPath, file))
		if err != nil {
			// TODO: accumulate errors
			return err
		}
	}
	return nil
}

func deletePVs(c *kubernetes.Clientset, existingPVs pvMap) {
	for _, pv := range existingPVs {
		if pv.Status.Phase == v1.VolumeReleased &&
			pv.Annotations["pv.kubernetes.io/provisioned-by"] == "local-storage-provisioner" {

			name := pv.Name
			fmt.Printf("Deleting PV: %v\n", name)
			// Remove all contents under directory
			err := deleteContents(pv.Spec.LocalStorage.Path)
			if err != nil {
				// Log event on PV
				fmt.Printf("Error reading directory: %v\n", err.Error())
				continue
			}

			// Remove API object
			err = c.Core().PersistentVolumes().Delete(name, &metav1.DeleteOptions{})
			if err != nil {
				// Log event on PV
				fmt.Printf("Error reading directory: %v\n", err.Error())
				continue
			}
			fmt.Printf("Done deleting PV: %v\n", name)
			deletePV(existingPVs, name)
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
	// TODO: error handling
	client := setupClient()
	for {
		existingPVMap, err := populateExistingPVs(client, node)
		if err == nil {
			deletePVs(client, existingPVMap)
			processLocalSSDs(client, existingPVMap, node)
		}
		time.Sleep(30 * time.Second)
	}
}
