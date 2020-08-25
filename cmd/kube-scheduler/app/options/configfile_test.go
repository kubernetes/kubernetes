package options

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func TestMergeInstanceConfiguration(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "scheduler")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	testCases := []struct {
		name                      string
		configFile                string
		instanceFile              string
		kubeConfig                string
		instanceKubeConfig        string
		expectedInstaceKubeConfig string
	}{
		{
			name:                      "kubeconfig without instance",
			configFile:                filepath.Join(tmpDir, "config.yaml"),
			kubeConfig:                "config.yaml",
			expectedInstaceKubeConfig: "config.yaml",
		},
		{
			name:                      "kubeconfig with instance overwrites",
			configFile:                filepath.Join(tmpDir, "config.yaml"),
			instanceFile:              filepath.Join(tmpDir, "instance-config.yaml"),
			kubeConfig:                "config.yaml",
			instanceKubeConfig:        "instance-config.yaml",
			expectedInstaceKubeConfig: "instance-config.yaml",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.configFile != "" {
				createConfigurationFile(t, tc.configFile, tc.kubeConfig)
			}

			if tc.instanceFile != "" {
				createConfigurationFile(t, tc.instanceFile, tc.instanceKubeConfig)
			}

			config, err := loadConfigFromFile(tc.configFile, tc.instanceFile)
			if err != nil {
				t.Fatalf("unexpected error for %s: %v", tc.name, err)
			}

			assert.Equal(t, config.ClientConnection.Kubeconfig, tc.expectedInstaceKubeConfig, config.ClientConnection.Kubeconfig)
		})
	}
}

func createConfigurationFile(t *testing.T, file, kubeConfig string) []byte {
	data := []byte(fmt.Sprintf(`
apiVersion: kubescheduler.config.k8s.io/v1beta1
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: "%s"`, kubeConfig))
	if err := ioutil.WriteFile(file, data, os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}
	return data
}
