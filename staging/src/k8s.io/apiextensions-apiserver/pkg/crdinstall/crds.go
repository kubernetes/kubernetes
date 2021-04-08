package crdinstall

import (
	"embed"
	"fmt"
	"k8s.io/apimachinery/pkg/runtime/serializer/yaml"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

//go:embed configs/*.yaml
var content embed.FS

const configsDir = "configs"

var installCRDs []*unstructured.Unstructured

func init() {

	decoder := yaml.NewDecodingSerializer(unstructured.UnstructuredJSONScheme)

	items, err := content.ReadDir(configsDir)
	if err != nil {
		fmt.Println("failed to the content of the config dir")
		return
	}

	for _, item := range items {
		// read the config
		b, err := content.ReadFile(fmt.Sprintf("%v/%v", configsDir, item.Name()))
		if err != nil {
			fmt.Printf("failed to readfile %v : %v\n", item.Name(), err)
			return
		}
		obj := &unstructured.Unstructured{}
		_, _, err = decoder.Decode(b, nil, obj)
		if err != nil {
			fmt.Printf("failed to decode %v : %v\n", item.Name(), err)
			return
		}

		// add to list
		installCRDs = append(installCRDs, obj)
	}
}

func Objects() []*unstructured.Unstructured {
	return installCRDs
}
