package main

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/pkg/api/v1"
)

func podForTest() *v1.Pod {
	p := &v1.Pod{}
	p.SetName(fmt.Sprintf("%d", rand.Int()))
	p.SetNamespace(fmt.Sprintf("%d", rand.Int()))
	return p
}

func TestMain(t *testing.T) {
	pod1 := podForTest()
	pod2 := podForTest()
	pod3 := podForTest()
	type testCase struct {
		description       string
		initial           []runtime.Object
		add               []runtime.Object
		update            []runtime.Object
		delete            []runtime.Object
		expectedLineCount int64
	}
	testCases := []testCase{
		{
			description:       "Single initial pod",
			initial:           []runtime.Object{pod1},
			expectedLineCount: 1,
		},
		{
			description:       "Multiple initial pods",
			initial:           []runtime.Object{pod1, pod2},
			expectedLineCount: 2,
		},
		{
			description:       "Pod added later",
			initial:           []runtime.Object{pod1, pod2},
			add:               []runtime.Object{pod3},
			expectedLineCount: 3,
		},
	}

	for _, tc := range testCases {
		t.Run(
			tc.description,
			func(t *testing.T) {
				initialLineCount := glog.Stats.Info.Lines()
				clientset := fake.NewSimpleClientset(tc.initial...)
				factory := informers.NewSharedInformerFactory(
					clientset,
					time.Hour*24,
				)
				c := NewPodLoggingController(factory)

				stop := make(chan struct{})
				defer close(stop)

				err := c.Run(stop)
				if err != nil {
					t.Error(err)
				}
				for _, podToAdd := range tc.add {
					// Type conversion + type assertion? Is this the only way?
					p := interface{}(podToAdd).(*v1.Pod)
					// XXX: I expected this to trigger the
					// cache.ResourceEventHandlerFuncs but it doesn't.
					_, err = clientset.Core().Pods(p.Namespace).Create(p)
					if err != nil {
						t.Error(err)
					}
				}
				actualLineCount := glog.Stats.Info.Lines() - initialLineCount

				if tc.expectedLineCount != actualLineCount {
					t.Errorf(
						"Line count mismatch. Expected %d. Got %d",
						tc.expectedLineCount,
						actualLineCount,
					)
				}
			},
		)
	}
}
