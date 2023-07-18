/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package flowcontrol

import (
	"testing"
	"time"

	"k8s.io/api/flowcontrol/v1beta3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fqs "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/queueset"
	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing/eventclock"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
)

// Test_GetMaxSeats tests max seats retrieved from MaxSeatsTracker
func Test_GetMaxSeats(t *testing.T) {
	testcases := []struct {
		name             string
		nominalCL        int
		handSize         int32
		expectedMaxSeats uint64
	}{
		{
			name:             "nominalCL=5, handSize=6",
			nominalCL:        5,
			handSize:         6,
			expectedMaxSeats: 1,
		},
		{
			name:             "nominalCL=10, handSize=6",
			nominalCL:        10,
			handSize:         6,
			expectedMaxSeats: 1,
		},
		{
			name:             "nominalCL=15, handSize=6",
			nominalCL:        15,
			handSize:         6,
			expectedMaxSeats: 2,
		},
		{
			name:             "nominalCL=20, handSize=6",
			nominalCL:        20,
			handSize:         6,
			expectedMaxSeats: 3,
		},
		{
			name:             "nominalCL=35, handSize=6",
			nominalCL:        35,
			handSize:         6,
			expectedMaxSeats: 5,
		},
		{
			name:             "nominalCL=100, handSize=6",
			nominalCL:        100,
			handSize:         6,
			expectedMaxSeats: 15,
		},
		{
			name:             "nominalCL=200, handSize=6",
			nominalCL:        200,
			handSize:         6,
			expectedMaxSeats: 30,
		},
		{
			name:             "nominalCL=10, handSize=1",
			nominalCL:        10,
			handSize:         1,
			expectedMaxSeats: 2,
		},
		{
			name:             "nominalCL=100, handSize=20",
			nominalCL:        100,
			handSize:         20,
			expectedMaxSeats: 5,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			clientset := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(clientset, time.Second)
			flowcontrolClient := clientset.FlowcontrolV1beta3()
			startTime := time.Now()
			clk, _ := eventclock.NewFake(startTime, 0, nil)
			c := newTestableController(TestableConfig{
				Name:              "Controller",
				Clock:             clk,
				InformerFactory:   informerFactory,
				FlowcontrolClient: flowcontrolClient,
				// for the purposes of this test, serverCL ~= nominalCL since there is
				// only 1 PL with large concurrency shares, making mandatory PLs negligible.
				ServerConcurrencyLimit: testcase.nominalCL,
				RequestWaitLimit:       time.Minute,
				ReqsGaugeVec:           metrics.PriorityLevelConcurrencyGaugeVec,
				ExecSeatsGaugeVec:      metrics.PriorityLevelExecutionSeatsGaugeVec,
				QueueSetFactory:        fqs.NewQueueSetFactory(clk),
			})

			testPriorityLevel := &v1beta3.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pl",
				},
				Spec: v1beta3.PriorityLevelConfigurationSpec{
					Type: v1beta3.PriorityLevelEnablementLimited,
					Limited: &v1beta3.LimitedPriorityLevelConfiguration{
						NominalConcurrencyShares: 10000,
						LimitResponse: v1beta3.LimitResponse{
							Queuing: &v1beta3.QueuingConfiguration{
								HandSize: testcase.handSize,
							},
						},
					},
				},
			}
			c.digestConfigObjects([]*v1beta3.PriorityLevelConfiguration{testPriorityLevel}, nil)
			maxSeats := c.GetMaxSeats("test-pl")
			if maxSeats != testcase.expectedMaxSeats {
				t.Errorf("unexpected max seats, got=%d, want=%d", maxSeats, testcase.expectedMaxSeats)
			}
		})
	}
}
