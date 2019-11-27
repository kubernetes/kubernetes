/*
Copyright 2016 The Kubernetes Authors.

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

package e2enode

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e_node/services"

	"github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Container Runtime Conformance Test", func() {
	f := framework.NewDefaultFramework("runtime-conformance")

	ginkgo.Describe("container runtime conformance blackbox test", func() {

		ginkgo.Context("when running a container with a new image", func() {
			// The service account only has pull permission
			auth := `
{
	"auths": {
		"https://gcr.io": {
			"auth": "X2pzb25fa2V5OnsKICAidHlwZSI6ICJzZXJ2aWNlX2FjY291bnQiLAogICJwcm9qZWN0X2lkIjogImF1dGhlbnRpY2F0ZWQtaW1hZ2UtcHVsbGluZyIsCiAgInByaXZhdGVfa2V5X2lkIjogImI5ZjJhNjY0YWE5YjIwNDg0Y2MxNTg2MDYzZmVmZGExOTIyNGFjM2IiLAogICJwcml2YXRlX2tleSI6ICItLS0tLUJFR0lOIFBSSVZBVEUgS0VZLS0tLS1cbk1JSUV2UUlCQURBTkJna3Foa2lHOXcwQkFRRUZBQVNDQktjd2dnU2pBZ0VBQW9JQkFRQzdTSG5LVEVFaVlMamZcbkpmQVBHbUozd3JCY2VJNTBKS0xxS21GWE5RL3REWGJRK2g5YVl4aldJTDhEeDBKZTc0bVovS01uV2dYRjVLWlNcbm9BNktuSU85Yi9SY1NlV2VpSXRSekkzL1lYVitPNkNjcmpKSXl4anFWam5mVzJpM3NhMzd0OUE5VEZkbGZycm5cbjR6UkpiOWl4eU1YNGJMdHFGR3ZCMDNOSWl0QTNzVlo1ODhrb1FBZmgzSmhhQmVnTWorWjRSYko0aGVpQlFUMDNcbnZVbzViRWFQZVQ5RE16bHdzZWFQV2dydDZOME9VRGNBRTl4bGNJek11MjUzUG4vSzgySFpydEx4akd2UkhNVXhcbng0ZjhwSnhmQ3h4QlN3Z1NORit3OWpkbXR2b0wwRmE3ZGducFJlODZWRDY2ejNZenJqNHlLRXRqc2hLZHl5VWRcbkl5cVhoN1JSQWdNQkFBRUNnZ0VBT3pzZHdaeENVVlFUeEFka2wvSTVTRFVidi9NazRwaWZxYjJEa2FnbmhFcG9cbjFJajJsNGlWMTByOS9uenJnY2p5VlBBd3pZWk1JeDFBZVF0RDdoUzRHWmFweXZKWUc3NkZpWFpQUm9DVlB6b3VcbmZyOGRDaWFwbDV0enJDOWx2QXNHd29DTTdJWVRjZmNWdDdjRTEyRDNRS3NGNlo3QjJ6ZmdLS251WVBmK0NFNlRcbmNNMHkwaCtYRS9kMERvSERoVy96YU1yWEhqOFRvd2V1eXRrYmJzNGYvOUZqOVBuU2dET1lQd2xhbFZUcitGUWFcbkpSd1ZqVmxYcEZBUW14M0Jyd25rWnQzQ2lXV2lGM2QrSGk5RXRVYnRWclcxYjZnK1JRT0licWFtcis4YlJuZFhcbjZWZ3FCQWtKWjhSVnlkeFVQMGQxMUdqdU9QRHhCbkhCbmM0UW9rSXJFUUtCZ1FEMUNlaWN1ZGhXdGc0K2dTeGJcbnplanh0VjFONDFtZHVjQnpvMmp5b1dHbzNQVDh3ckJPL3lRRTM0cU9WSi9pZCs4SThoWjRvSWh1K0pBMDBzNmdcblRuSXErdi9kL1RFalk4MW5rWmlDa21SUFdiWHhhWXR4UjIxS1BYckxOTlFKS2ttOHRkeVh5UHFsOE1veUdmQ1dcbjJ2aVBKS05iNkhabnY5Q3lqZEo5ZzJMRG5RS0JnUUREcVN2eURtaGViOTIzSW96NGxlZ01SK205Z2xYVWdTS2dcbkVzZlllbVJmbU5XQitDN3ZhSXlVUm1ZNU55TXhmQlZXc3dXRldLYXhjK0krYnFzZmx6elZZdFpwMThNR2pzTURcbmZlZWZBWDZCWk1zVXQ3Qmw3WjlWSjg1bnRFZHFBQ0xwWitaLzN0SVJWdWdDV1pRMWhrbmxHa0dUMDI0SkVFKytcbk55SDFnM2QzUlFLQmdRQ1J2MXdKWkkwbVBsRklva0tGTkh1YTBUcDNLb1JTU1hzTURTVk9NK2xIckcxWHJtRjZcbkMwNGNTKzQ0N0dMUkxHOFVUaEpKbTRxckh0Ti9aK2dZOTYvMm1xYjRIakpORDM3TVhKQnZFYTN5ZUxTOHEvK1JcbjJGOU1LamRRaU5LWnhQcG84VzhOSlREWTVOa1BaZGh4a2pzSHdVNGRTNjZwMVRESUU0MGd0TFpaRFFLQmdGaldcbktyblFpTnEzOS9iNm5QOFJNVGJDUUFKbmR3anhTUU5kQTVmcW1rQTlhRk9HbCtqamsxQ1BWa0tNSWxLSmdEYkpcbk9heDl2OUc2Ui9NSTFIR1hmV3QxWU56VnRocjRIdHNyQTB0U3BsbWhwZ05XRTZWejZuQURqdGZQSnMyZUdqdlhcbmpQUnArdjhjY21MK3dTZzhQTGprM3ZsN2VlNXJsWWxNQndNdUdjUHhBb0dBZWRueGJXMVJMbVZubEFpSEx1L0xcbmxtZkF3RFdtRWlJMFVnK1BMbm9Pdk81dFE1ZDRXMS94RU44bFA0cWtzcGtmZk1Rbk5oNFNZR0VlQlQzMlpxQ1RcbkpSZ2YwWGpveXZ2dXA5eFhqTWtYcnBZL3ljMXpmcVRaQzBNTzkvMVVjMWJSR2RaMmR5M2xSNU5XYXA3T1h5Zk9cblBQcE5Gb1BUWGd2M3FDcW5sTEhyR3pNPVxuLS0tLS1FTkQgUFJJVkFURSBLRVktLS0tLVxuIiwKICAiY2xpZW50X2VtYWlsIjogImltYWdlLXB1bGxpbmdAYXV0aGVudGljYXRlZC1pbWFnZS1wdWxsaW5nLmlhbS5nc2VydmljZWFjY291bnQuY29tIiwKICAiY2xpZW50X2lkIjogIjExMzc5NzkxNDUzMDA3MzI3ODcxMiIsCiAgImF1dGhfdXJpIjogImh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbS9vL29hdXRoMi9hdXRoIiwKICAidG9rZW5fdXJpIjogImh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbS9vL29hdXRoMi90b2tlbiIsCiAgImF1dGhfcHJvdmlkZXJfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9vYXV0aDIvdjEvY2VydHMiLAogICJjbGllbnRfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9yb2JvdC92MS9tZXRhZGF0YS94NTA5L2ltYWdlLXB1bGxpbmclNDBhdXRoZW50aWNhdGVkLWltYWdlLXB1bGxpbmcuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iCn0=",
			"email": "image-pulling@authenticated-image-pulling.iam.gserviceaccount.com"
		}
	}
}`
			// The following images are not added into NodeImageWhiteList, because this test is
			// testing image pulling, these images don't need to be prepulled. The ImagePullPolicy
			// is v1.PullAlways, so it won't be blocked by framework image white list check.
			for _, testCase := range []struct {
				description string
				image       string
				phase       v1.PodPhase
				waiting     bool
			}{
				{
					description: "should be able to pull from private registry with credential provider",
					image:       "gcr.io/authenticated-image-pulling/alpine:3.7",
					phase:       v1.PodRunning,
					waiting:     false,
				},
			} {
				testCase := testCase
				ginkgo.It(testCase.description+" [NodeConformance]", func() {
					name := "image-pull-test"
					command := []string{"/bin/sh", "-c", "while true; do sleep 1; done"}
					container := common.ConformanceContainer{
						PodClient: f.PodClient(),
						Container: v1.Container{
							Name:    name,
							Image:   testCase.image,
							Command: command,
							// PullAlways makes sure that the image will always be pulled even if it is present before the test.
							ImagePullPolicy: v1.PullAlways,
						},
						RestartPolicy: v1.RestartPolicyNever,
					}

					configFile := filepath.Join(services.KubeletRootDirectory, "config.json")
					err := ioutil.WriteFile(configFile, []byte(auth), 0644)
					framework.ExpectNoError(err)
					defer os.Remove(configFile)

					// checkContainerStatus checks whether the container status matches expectation.
					checkContainerStatus := func() error {
						status, err := container.GetStatus()
						if err != nil {
							return fmt.Errorf("failed to get container status: %v", err)
						}
						// We need to check container state first. The default pod status is pending, If we check
						// pod phase first, and the expected pod phase is Pending, the container status may not
						// even show up when we check it.
						// Check container state
						if !testCase.waiting {
							if status.State.Running == nil {
								return fmt.Errorf("expected container state: Running, got: %q",
									common.GetContainerState(status.State))
							}
						}
						if testCase.waiting {
							if status.State.Waiting == nil {
								return fmt.Errorf("expected container state: Waiting, got: %q",
									common.GetContainerState(status.State))
							}
							reason := status.State.Waiting.Reason
							if reason != images.ErrImagePull.Error() &&
								reason != images.ErrImagePullBackOff.Error() {
								return fmt.Errorf("unexpected waiting reason: %q", reason)
							}
						}
						// Check pod phase
						phase, err := container.GetPhase()
						if err != nil {
							return fmt.Errorf("failed to get pod phase: %v", err)
						}
						if phase != testCase.phase {
							return fmt.Errorf("expected pod phase: %q, got: %q", testCase.phase, phase)
						}
						return nil
					}
					// The image registry is not stable, which sometimes causes the test to fail. Add retry mechanism to make this
					// less flaky.
					const flakeRetry = 3
					for i := 1; i <= flakeRetry; i++ {
						var err error
						ginkgo.By("create the container")
						container.Create()
						ginkgo.By("check the container status")
						for start := time.Now(); time.Since(start) < common.ContainerStatusRetryTimeout; time.Sleep(common.ContainerStatusPollInterval) {
							if err = checkContainerStatus(); err == nil {
								break
							}
						}
						ginkgo.By("delete the container")
						container.Delete()
						if err == nil {
							break
						}
						if i < flakeRetry {
							framework.Logf("No.%d attempt failed: %v, retrying...", i, err)
						} else {
							framework.Failf("All %d attempts failed: %v", flakeRetry, err)
						}
					}
				})
			}
		})
	})
})
