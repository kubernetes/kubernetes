package crdinstall

import (
	"bytes"
	"embed"
	"fmt"
	"io"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	yamlserializer "k8s.io/apimachinery/pkg/runtime/serializer/yaml"
	"k8s.io/apimachinery/pkg/util/yaml"
)

//go:embed configs/*.yaml
var content embed.FS

const configsDir = "configs"

var prorityOrder = []string{
	"vmset_crd_definition.yaml",
}

func getObjectsFromReader(r io.Reader) ([]*unstructured.Unstructured, error) {
	var objList []*unstructured.Unstructured

	yamlDecoder := yaml.NewYAMLOrJSONDecoder(r, 4098)
	decoder := yamlserializer.NewDecodingSerializer(unstructured.UnstructuredJSONScheme)
	for {
		ext := &runtime.RawExtension{}
		if err := yamlDecoder.Decode(ext); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		ext.Raw = bytes.TrimSpace(ext.Raw)
		if len(ext.Raw) == 0 || bytes.Equal(ext.Raw, []byte("null")) {
			continue
		}
		obj := &unstructured.Unstructured{}
		_, _, err := decoder.Decode(ext.Raw, nil, obj)
		if err != nil {
			return nil, err
		}
		objList = append(objList, obj)
	}

	return objList, nil
}

func getInPriorityOrder(content embed.FS, priorityOrder []string) ([]*unstructured.Unstructured, error) {
	read := sets.NewString()
	var objList []*unstructured.Unstructured

	// first read the files in the priority order and then read the remaining files in the order presented

	// Read files in priority order
	for _, name := range priorityOrder {
		f, err := content.Open(getFilePath(name))
		if err != nil {
			return nil, err
		}
		defer f.Close()
		objs, err := getObjectsFromReader(f)
		if err != nil {
			return nil, err
		}
		objList = append(objList, objs...)
		// mark this file as read
		read.Insert(name)
	}

	// now read all the other files that are not yet read
	files, err := content.ReadDir(configsDir)
	if err != nil {
		return nil, err
	}
	for _, file := range files {
		if read.Has(file.Name()) {
			continue
		}

		f, err := content.Open(getFilePath(file.Name()))
		if err != nil {
			return nil, err
		}
		defer f.Close()
		objs, err := getObjectsFromReader(f)
		if err != nil {
			return nil, err
		}
		objList = append(objList, objs...)
	}

	return objList, nil
}

func getObjects() ([]*unstructured.Unstructured, error) {
	return getInPriorityOrder(content, prorityOrder)
}

func getFilePath(name string) string {
	return fmt.Sprintf("%v/%v", configsDir, name)
}
