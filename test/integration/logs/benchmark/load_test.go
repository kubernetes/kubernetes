/*
Copyright 2021 The Kubernetes Authors.

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

package benchmark

import (
	"bytes"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

func TestData(t *testing.T) {
	container := v1.Container{
		Command:                []string{"sh -c \nf=/restart-count/restartCount\ncount=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})\nif [ $count -eq 1 ]; then\n\texit 1\nfi\nif [ $count -eq 2 ]; then\n\texit 0\nfi\nwhile true; do sleep 1; done\n"},
		Image:                  "registry.k8s.io/e2e-test-images/busybox:1.29-2",
		Name:                   "terminate-cmd-rpn",
		TerminationMessagePath: "/dev/termination-log",
	}

	testcases := map[string]struct {
		messages                 []logMessage
		printf, structured, json string
		stats                    logStats
	}{
		"data/simple.log": {
			messages: []logMessage{
				{
					msg: "Pod status updated",
				},
			},
			printf: `Pod status updated: []
`,
			structured: `"Pod status updated"
`,
			json: `{"msg":"Pod status updated","v":0}
`,
			stats: logStats{
				TotalLines: 1,
				JsonLines:  1,
				ArgCounts:  map[string]int{},
			},
		},
		"data/error.log": {
			messages: []logMessage{
				{
					msg:     "Pod status update",
					err:     errors.New("failed"),
					isError: true,
				},
			},
			printf: `Pod status update: failed []
`,
			structured: `"Pod status update" err="failed"
`,
			json: `{"msg":"Pod status update","err":"failed"}
`,
			stats: logStats{
				TotalLines:    1,
				JsonLines:     1,
				ErrorMessages: 1,
				ArgCounts: map[string]int{
					stringArg: 1,
					totalArg:  1,
				},
			},
		},
		"data/error-value.log": {
			messages: []logMessage{
				{
					msg: "Pod status update",
					kvs: []interface{}{"err", errors.New("failed")},
				},
			},
			printf: `Pod status update: [err failed]
`,
			structured: `"Pod status update" err="failed"
`,
			json: `{"msg":"Pod status update","v":0,"err":"failed"}
`,
			stats: logStats{
				TotalLines: 1,
				JsonLines:  1,
				ArgCounts: map[string]int{
					stringArg: 1,
					totalArg:  1,
				},
			},
		},
		"data/values.log": {
			messages: []logMessage{
				{
					msg: "Example",
					kvs: []interface{}{
						"pod", klog.KRef("system", "kube-scheduler"),
						"pv", klog.KRef("", "volume"),
						"someString", "hello world",
						"someValue", 1.0,
					},
				},
			},
			printf: `Example: [pod system/kube-scheduler pv volume someString hello world someValue 1]
`,
			structured: `"Example" pod="system/kube-scheduler" pv="volume" someString="hello world" someValue=1
`,
			json: `{"msg":"Example","v":0,"pod":{"name":"kube-scheduler","namespace":"system"},"pv":{"name":"volume"},"someString":"hello world","someValue":1}
`,
			stats: logStats{
				TotalLines: 1,
				JsonLines:  1,
				ArgCounts: map[string]int{
					stringArg: 1,
					krefArg:   2,
					numberArg: 1,
					totalArg:  4,
				},
			},
		},
		"data/container.log": {
			messages: []logMessage{
				{
					msg: "Creating container in pod",
					kvs: []interface{}{
						"container", &container,
					},
				},
			},
			printf: `Creating container in pod: [container &Container{Name:terminate-cmd-rpn,Image:registry.k8s.io/e2e-test-images/busybox:1.29-2,Command:[sh -c 
f=/restart-count/restartCount
count=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})
if [ $count -eq 1 ]; then
	exit 1
fi
if [ $count -eq 2 ]; then
	exit 0
fi
while true; do sleep 1; done
],Args:[],WorkingDir:,Ports:[]ContainerPort{},Env:[]EnvVar{},Resources:ResourceRequirements{Limits:ResourceList{},Requests:ResourceList{},},VolumeMounts:[]VolumeMount{},LivenessProbe:nil,ReadinessProbe:nil,Lifecycle:nil,TerminationMessagePath:/dev/termination-log,ImagePullPolicy:,SecurityContext:nil,Stdin:false,StdinOnce:false,TTY:false,EnvFrom:[]EnvFromSource{},TerminationMessagePolicy:,VolumeDevices:[]VolumeDevice{},StartupProbe:nil,}]
`,
			structured: `"Creating container in pod" container=<
	&Container{Name:terminate-cmd-rpn,Image:registry.k8s.io/e2e-test-images/busybox:1.29-2,Command:[sh -c 
	f=/restart-count/restartCount
	count=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})
	if [ $count -eq 1 ]; then
		exit 1
	fi
	if [ $count -eq 2 ]; then
		exit 0
	fi
	while true; do sleep 1; done
	],Args:[],WorkingDir:,Ports:[]ContainerPort{},Env:[]EnvVar{},Resources:ResourceRequirements{Limits:ResourceList{},Requests:ResourceList{},},VolumeMounts:[]VolumeMount{},LivenessProbe:nil,ReadinessProbe:nil,Lifecycle:nil,TerminationMessagePath:/dev/termination-log,ImagePullPolicy:,SecurityContext:nil,Stdin:false,StdinOnce:false,TTY:false,EnvFrom:[]EnvFromSource{},TerminationMessagePolicy:,VolumeDevices:[]VolumeDevice{},StartupProbe:nil,}
 >
`,
			// This is what the output would look like with JSON object. Because of https://github.com/kubernetes/kubernetes/issues/106652 we get the string instead.
			// 			json: `{"msg":"Creating container in pod","v":0,"container":{"name":"terminate-cmd-rpn","image":"registry.k8s.io/e2e-test-images/busybox:1.29-2","command":["sh -c \nf=/restart-count/restartCount\ncount=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})\nif [ $count -eq 1 ]; then\n\texit 1\nfi\nif [ $count -eq 2 ]; then\n\texit 0\nfi\nwhile true; do sleep 1; done\n"],"resources":{},"terminationMessagePath":"/dev/termination-log"}}
			// `,
			json: `{"msg":"Creating container in pod","v":0,"container":"&Container{Name:terminate-cmd-rpn,Image:registry.k8s.io/e2e-test-images/busybox:1.29-2,Command:[sh -c \nf=/restart-count/restartCount\ncount=$(echo 'hello' >> $f ; wc -l $f | awk {'print $1'})\nif [ $count -eq 1 ]; then\n\texit 1\nfi\nif [ $count -eq 2 ]; then\n\texit 0\nfi\nwhile true; do sleep 1; done\n],Args:[],WorkingDir:,Ports:[]ContainerPort{},Env:[]EnvVar{},Resources:ResourceRequirements{Limits:ResourceList{},Requests:ResourceList{},},VolumeMounts:[]VolumeMount{},LivenessProbe:nil,ReadinessProbe:nil,Lifecycle:nil,TerminationMessagePath:/dev/termination-log,ImagePullPolicy:,SecurityContext:nil,Stdin:false,StdinOnce:false,TTY:false,EnvFrom:[]EnvFromSource{},TerminationMessagePolicy:,VolumeDevices:[]VolumeDevice{},StartupProbe:nil,}"}
`,
			stats: logStats{
				TotalLines: 2,
				JsonLines:  1,
				ArgCounts: map[string]int{
					totalArg: 1,
					otherArg: 1,
				},
				OtherLines: []string{
					"0: # This is a manually created message. See https://github.com/kubernetes/kubernetes/issues/106652 for the real one.",
				},
				OtherArgs: []interface{}{
					&container,
				},
			},
		},
	}

	for path, expected := range testcases {
		t.Run(path, func(t *testing.T) {
			messages, stats, err := loadLog(path)
			if err != nil {
				t.Fatalf("unexpected load error: %v", err)
			}
			assert.Equal(t, expected.messages, messages)
			assert.Equal(t, expected.stats, stats)
			print := func(format func(item logMessage)) {
				for _, item := range expected.messages {
					format(item)
				}
			}
			testBuffered := func(t *testing.T, expected string, format func(item logMessage)) {
				var buffer bytes.Buffer
				klog.SetOutput(&buffer)
				defer klog.SetOutput(&output)

				print(format)
				klog.Flush()
				assert.Equal(t, expected, buffer.String())
			}

			t.Run("printf", func(t *testing.T) {
				testBuffered(t, expected.printf, printf)
			})
			t.Run("structured", func(t *testing.T) {
				testBuffered(t, expected.structured, prints)
			})
			t.Run("json", func(t *testing.T) {
				var buffer bytes.Buffer
				logger := newJSONLogger(&buffer)
				klog.SetLogger(logger)
				defer klog.ClearLogger()
				print(prints)
				klog.Flush()
				assert.Equal(t, expected.json, buffer.String())
			})
		})
	}
}
