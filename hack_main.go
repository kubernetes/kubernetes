package main

import (
	"context"
	"encoding/json"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	appsv1ac "k8s.io/client-go/applyconfigurations/apps/v1"
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	_ "k8s.io/client-go/plugin/pkg/client/auth/gcp"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kube-openapi/pkg/util/proto"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

func main() {
	fmt.Println("vim-go")
	defaultNS := "default"
	mydep := "mydep"
	mgr := "mymanager"
	kubeconfig := "/usr/local/google/home/kevindelgado/.kube/config"

	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		panic(err)
	}
	client := clientset.NewForConfigOrDie(config)
	deployClient := client.AppsV1().Deployments(defaultNS)

	/// sanity check that I know how to apply and extract a builtin type
	deployApply := appsv1ac.Deployment(mydep, "default").
		WithLabels(map[string]string{"label1": "value1"}).
		WithSpec(appsv1ac.DeploymentSpec().
			WithSelector(metav1ac.LabelSelector().
				WithMatchLabels(map[string]string{"app": mydep}),
			).
			WithTemplate(corev1ac.PodTemplateSpec().
				WithLabels(map[string]string{"app": mydep}).
				WithSpec(corev1ac.PodSpec().
					WithContainers(
						corev1ac.Container().
							WithName("initial-container").
							WithImage("nginx:1.14.2"),
					),
				),
			),
		)

	applied, err := deployClient.Apply(context.TODO(), deployApply, metav1.ApplyOptions{FieldManager: mgr})
	if err != nil {
		panic(err)
	}
	fmt.Printf("applied = %+v\n", applied)
	fmt.Println("---")

	extracted, err := appsv1ac.ExtractDeployment(applied, mgr)
	if err != nil {
		panic(err)
	}
	fmt.Printf("extracted = %+v\n", extracted)
	fmt.Println("---")

	extracted.WithLabels(map[string]string{"label2": "value2"})
	finalDep, err := deployClient.Apply(context.TODO(), extracted, metav1.ApplyOptions{FieldManager: mgr})

	if err != nil {
		panic(err)
	}
	fmt.Printf("finalDep = %+v\n", finalDep)
	fmt.Println("---")

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		panic(err)
	}

	guestbookGVR := schema.GroupVersionResource{
		Group:    "webapp.my.domain",
		Version:  "v1",
		Resource: "guestbooks",
	}

	guestbookClient := dynamicClient.Resource(guestbookGVR).Namespace(defaultNS)

	////// DEBUGGING

	//// list all guestbooks as part of debugging the guestbookClient

	//listed, err := guestbookClient.List(context.TODO(), metav1.ListOptions{})
	//if err != nil {
	//	panic(err)
	//}
	//fmt.Printf("listed = %+v\n", listed)
	//fmt.Println("---")

	//// perform a Create first as part of debugging the guestbookClient
	//gbName := "created-gb-1"
	//gb := &unstructured.Unstructured{
	//	Object: map[string]interface{}{
	//		//"apiVersion": guestbookGVR.Group + "/" + guestbookGVR.Version,
	//		"apiVersion": "webapp.my.domain/v1",
	//		"kind":       "Guestbook",
	//		"metadata": map[string]interface{}{
	//			"name": gbName,
	//			//"namespace": defaultNS,
	//		},
	//		"spec": map[string]interface{}{
	//			"foo":              "bar1",
	//			"minuteMultiplier": 101,
	//		},
	//	},
	//}

	//createdGuestbook, err := guestbookClient.Create(
	//	context.TODO(),
	//	gb,
	//	metav1.CreateOptions{},
	//)

	//if err != nil {
	//	panic(err)
	//}
	//fmt.Println("---")
	//fmt.Printf("createdGuestbook = %+v\n", createdGuestbook)

	gbName2 := "applied-gb-2"
	gb2 := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "webapp.my.domain/v1",
			"kind":       "Guestbook",
			"metadata": map[string]interface{}{
				"name": gbName2,
			},
			"spec": map[string]interface{}{
				"foo":              "bar2",
				"minuteMultiplier": 102,
			},
		},
	}

	gbData, err := json.Marshal(gb2)
	if err != nil {
		panic(err)
	}

	appliedGuestbook, err := guestbookClient.Patch(
		context.TODO(),
		gbName2,
		types.ApplyPatchType,
		gbData,
		metav1.PatchOptions{FieldManager: mgr},
	)
	if err != nil {
		panic(err)
	}

	fmt.Printf("appliedGuestbook = %+v\n", appliedGuestbook)
	fmt.Println("---")

	discoveryClient := discovery.NewDiscoveryClientForConfigOrDie(config)
	//// OLD WAY
	//extractedGuestbook, err := extractUnstructured(appliedGuestbook, guestbookGVR, discoveryClient, mgr)
	//if err != nil {
	//	panic(err)
	//}

	//// NEW WAY
	fmt.Println("NEW!")
	extractor := metav1ac.NewUnstructuredExtractor(discoveryClient)
	extractedGuestbook, err := extractor.ExtractUnstructured(appliedGuestbook, mgr)
	if err != nil {
		panic(err)
	}
	fmt.Println("---")
	fmt.Printf("extractedGuestbook = %+v\n", extractedGuestbook)

}

func parserFromGVR(gvk schema.GroupVersionKind, dc *discovery.DiscoveryClient) *typed.ParseableType {
	doc, err := dc.OpenAPISchema()
	if err != nil {
		panic(err)
	}

	models, err := proto.NewOpenAPIData(doc)
	if err != nil {
		panic(err)
	}

	gvkParser, err := fieldmanager.NewGVKParser(models, false)
	if err != nil {
		panic(err)
	}

	return gvkParser.Type(gvk)

}

// TODO: test with different mgr or something that doesn't just copy from applied into the output
// TODO: what to do about subresource? maybe we need a separate extractUnstrcutruedStatus
func extractUnstructured(applied *unstructured.Unstructured, gvr schema.GroupVersionResource, dc *discovery.DiscoveryClient, fieldManager string) (*unstructured.Unstructured, error) {

	//TODO: acquire gvk in a better way
	gvk := schema.GroupVersionKind{
		Group:   gvr.Group,
		Version: gvr.Version,
		Kind:    "Guestbook",
	}
	crdParser := parserFromGVR(gvk, dc)

	result := &unstructured.Unstructured{}
	err := managedfields.ExtractInto(applied, *crdParser, fieldManager, result, "")
	if err != nil {
		return nil, err
	}
	result.SetName(applied.GetName())
	result.SetNamespace(applied.GetNamespace())
	result.SetKind(applied.GetKind())
	result.SetAPIVersion(applied.GetAPIVersion())
	return result, nil
}

type UnstructuredApplyConfiguration struct {
	metav1ac.TypeMetaApplyConfiguration    `json:",inline"`
	*metav1ac.ObjectMetaApplyConfiguration `json:"metadata,omitempty"`
	Spec                                   map[string]interface{} `json:"spec,omitempty"`
	Status                                 map[string]interface{} `json:"status,omitempty"`
}
