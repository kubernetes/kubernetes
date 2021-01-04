package scheduler

import (
	"context"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
)

func deletePodHandler(c clientset.Interface, emitEventFunc func(types.NamespacedName)) func(args *WorkArgs) error {
	return func(args *WorkArgs) error {
		ns := args.NamespacedName.Namespace
		name := args.NamespacedName.Name
		if emitEventFunc != nil {
			emitEventFunc(args.NamespacedName)
		}
		var err error
		for i := 0; i < retries; i++ {
			err = c.CoreV1().Pods(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
			if err == nil {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		return err
	}
}
