package main

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"

	"github.com/davecgh/go-spew/spew"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	runtimejson "k8s.io/apimachinery/pkg/runtime/serializer/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	multiclusterv1alpha1 "k8s.io/apimachinery/use-raw-extension/pkg/apis/multicluster/v1alpha1"
)

var exampleObj = `
apiVersion: multicluster.x-k8s.io/v1alpha1
kind: Work
metadata:
  name: example
  namespace: some-ns
spec:
  workload:
    manifests:
    - apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: my-deployment
        namespace: different-ns
    - apiVersion: v1
      kind: Secret
      metadata:
        name: my-secret
        namespace: different-ns
`

var scheme = runtime.NewScheme()
var codecFactory = serializer.NewCodecFactory(scheme)

func init() {
	utilruntime.Must(multiclusterv1alpha1.Install(scheme))
}

func main() {
	rand.Seed(time.Now().UnixNano())

	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "USAGE: read-work <filename>")
		os.Exit(1)
	}

	if err := runMain(); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v", err)
		os.Exit(1)
	}
}

func runMain() error {
	workBytes, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		return err
	}

	// yeah, this isn't easy.  I don't know how I know.  Generated clients do this for you.
	jsonSerializer := runtimejson.NewSerializerWithOptions(runtimejson.DefaultMetaFactory, scheme, scheme, runtimejson.SerializerOptions{Yaml: true, Pretty: true, Strict: true})
	decoder := codecFactory.DecoderToVersion(jsonSerializer, multiclusterv1alpha1.GroupVersion)

	decodedObj, err := runtime.Decode(decoder, workBytes)
	if err != nil {
		return err
	}

	fmt.Printf("decoded to type %T\n", decodedObj)
	fmt.Printf("spews as %v\n\n", spew.Sdump(decodedObj))

	for i, curr := range decodedObj.(*multiclusterv1alpha1.Work).Spec.Workload.Manifests {
		metadata, err := meta.Accessor(curr.Object)
		if err != nil {
			return err
		}
		fmt.Printf("  manifest[%d] is %q in %q of type %v \n", i, metadata.GetName(), metadata.GetNamespace(), curr.Object.GetObjectKind().GroupVersionKind())
	}

	return nil
}
