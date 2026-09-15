/*
Copyright The Kubernetes Authors.

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

package node

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

const (
	mutableProbeTimeout      = 2 * time.Minute
	mutableProbeStableWindow = 10 * time.Second
)

// var _ = SIGDescribe("Mutable container probes [LinuxOnly]", framework.WithFeatureGate(features.MutableContainerProbes), func() {
var _ = SIGDescribe("Mutable container probes [LinuxOnly]", func() {

	f := framework.NewDefaultFramework("mutable-container-probes")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	for _, sidecar := range []bool{false, true} {
		kind := "regular container"
		if sidecar {
			kind = "restartable init container"
		}

		ginkgo.Context(kind, func() {
			ginkgo.It("should add, remove and re-add readiness without restarting", func(ctx context.Context) {
				t := newMutableProbeTest(f, sidecar)
				t.create(ctx)
				t.waitState(ctx, true, true)

				for _, enabled := range []bool{true, false, true} {
					count := len(t.requests("fail"))

					t.update(ctx, func(c *v1.Container) {
						c.ReadinessProbe = nil
						if enabled {
							c.ReadinessProbe = mutableHTTPProbe("fail")
						}
					})
					if enabled {
						t.waitRequests(ctx, "fail", count+1, mutableProbeTimeout)
					}

					t.waitState(ctx, true, !enabled)
					t.stableState(ctx, true, !enabled, mutableProbeStableWindow)
				}
			})

			ginkgo.It("should stop removed liveness probes and restart only after re-adding a failing probe", func(ctx context.Context) {
				t := newMutableProbeTest(f, sidecar)
				t.create(ctx)
				t.waitState(ctx, true, true)

				probe := mutableHTTPProbe("liveness")
				probe.FailureThreshold = 3

				t.update(ctx, func(c *v1.Container) { c.LivenessProbe = probe.DeepCopy() })
				t.waitRequests(ctx, "liveness", 2, mutableProbeTimeout)

				t.update(ctx, func(c *v1.Container) { c.LivenessProbe = nil })

				// Allow an already dispatched request to finish before checking that removal stopped the worker.
				t.stableState(ctx, true, true, mutableProbeStableWindow)
				count := len(t.requests("liveness"))
				initialCount := len(t.requests("liveness-initial"))

				t.exec("touch /state/liveness.fail")

				gomega.Consistently(ctx, func() int {
					t.checkIdentity(ctx)
					return len(t.requests("liveness"))
				}, 15*time.Second, time.Second).Should(gomega.Equal(count))

				t.update(ctx, func(c *v1.Container) { c.LivenessProbe = probe.DeepCopy() })
				t.waitRestart(ctx)

				gomega.Expect(len(t.requests("liveness-initial"))-initialCount).To(gomega.BeNumerically(">=", 3), "the original container must receive three failures before restarting")
			})

			for _, remove := range []bool{false, true} {
				action := "replace a failing startup handler"
				if remove {
					action = "remove a pending startup probe"
				}

				ginkgo.It("should "+action+" and enable readiness and liveness", func(ctx context.Context) {
					t := newMutableProbeTest(f, sidecar)
					c := t.container(t.pod)
					c.StartupProbe = mutableStartupProbe("fail")
					c.ReadinessProbe = mutableHTTPProbe("readiness")
					c.LivenessProbe = mutableHTTPProbe("liveness")

					t.create(ctx)
					t.waitRequests(ctx, "fail", 2, mutableProbeTimeout)
					t.waitState(ctx, false, false)
					t.stableState(ctx, false, false, 5*time.Second)

					gomega.Expect(t.requests("readiness")).To(gomega.BeEmpty())
					gomega.Expect(t.requests("liveness")).To(gomega.BeEmpty())

					t.update(ctx, func(c *v1.Container) {
						if remove {
							c.StartupProbe = nil
						} else {
							c.StartupProbe.ProbeHandler = mutableHTTPProbe("startup").ProbeHandler
						}
					})
					t.waitState(ctx, true, true)
					t.waitRequests(ctx, "readiness", 1, mutableProbeTimeout)
					t.waitRequests(ctx, "liveness", 1, mutableProbeTimeout)
					t.stableState(ctx, true, true, mutableProbeStableWindow)
				})
			}

			ginkgo.It("should honor startup added before the first container start", func(ctx context.Context) {
				t := newMutableProbeTest(f, sidecar)
				blocker := *t.container(t.pod).DeepCopy()
				blocker.Name = "blocker"
				blocker.RestartPolicy = nil
				blocker.Command = []string{"sh", "-c", "until test -e /state/release-init; do sleep 1; done"}
				t.pod.Spec.InitContainers = append([]v1.Container{blocker}, t.pod.Spec.InitContainers...)

				t.pod = t.client.Create(ctx, t.pod)
				framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, t.pod.Name,
					"blocking init container running", mutableProbeTimeout, func(p *v1.Pod) (bool, error) {
						for _, s := range p.Status.InitContainerStatuses {
							if s.Name == "blocker" {
								return s.State.Running != nil, nil
							}
						}

						return false, nil
					}))

				t.update(ctx, func(c *v1.Container) {
					c.StartupProbe = mutableStartupProbe("fail")
					c.ReadinessProbe = mutableHTTPProbe("readiness")
					c.LivenessProbe = mutableHTTPProbe("liveness")
				})

				// Let the updated spec propagate before the init container allows the first start.
				gomega.Consistently(ctx, func() *v1.ContainerStateRunning {
					return t.status(ctx).State.Running
				}, mutableProbeStableWindow, time.Second).Should(gomega.BeNil())

				_, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, t.pod.Name, "blocker", "touch", "/state/release-init")
				framework.ExpectNoError(err, "release blocking init container: %s", stderr)
				t.captureIdentity(ctx)
				t.waitRequests(ctx, "fail", 2, mutableProbeTimeout)
				t.waitState(ctx, false, false)
				t.stableState(ctx, false, false, mutableProbeStableWindow)

				gomega.Expect(t.requests("readiness")).To(gomega.BeEmpty())
				gomega.Expect(t.requests("liveness")).To(gomega.BeEmpty())

				t.update(ctx, func(c *v1.Container) { c.StartupProbe = nil })
				t.waitState(ctx, true, true)
			})

			for _, action := range []string{"add", "replace", "remove and re-add"} {
				ginkgo.It("should "+action+" startup after Started and use the latest probe on restart", func(ctx context.Context) {
					t := newMutableProbeTest(f, sidecar)
					c := t.container(t.pod)
					if action != "add" {
						c.StartupProbe = mutableStartupProbe("startup")
					}

					c.ReadinessProbe = mutableHTTPProbe("readiness")
					c.LivenessProbe = mutableHTTPProbe("liveness")

					t.create(ctx)
					t.waitState(ctx, true, true)

					if action == "remove and re-add" {

						t.update(ctx, func(c *v1.Container) { c.StartupProbe = nil })
						t.stableState(ctx, true, true, mutableProbeStableWindow)
					}

					t.update(ctx, func(c *v1.Container) { c.StartupProbe = mutableStartupProbe("fail") })
					t.stableState(ctx, true, true, mutableProbeStableWindow)

					gomega.Expect(t.requests("fail")).To(gomega.BeEmpty(), "startup must not run again in an already started container")

					t.exec("touch /state/restart")
					t.waitRestart(ctx)
					t.waitRequests(ctx, "fail", 2, mutableProbeTimeout)
					t.waitState(ctx, false, false)
					t.stableState(ctx, false, false, mutableProbeStableWindow)

					for _, name := range []string{"readiness", "liveness"} {
						gomega.Expect(t.requests(name+"-replacement")).To(gomega.BeEmpty(), "%s ran before replacement startup succeeded", name)
					}
				})
			}
		})
	}

	ginkgo.It("should promptly replace HTTP readiness parameters and then switch to exec", func(ctx context.Context) {
		t := newMutableProbeTest(f, false)
		probe := mutableHTTPProbe("readiness")
		probe.PeriodSeconds = 120
		t.container(t.pod).ReadinessProbe = probe

		t.create(ctx)
		t.waitState(ctx, true, true)

		t.update(ctx, func(c *v1.Container) { c.ReadinessProbe.HTTPGet.Path = "/cgi-bin/probe?fail" })
		t.waitRequests(ctx, "fail", 1, 30*time.Second)
		t.waitState(ctx, true, false)

		t.update(ctx, func(c *v1.Container) {
			c.ReadinessProbe.ProbeHandler = v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"sh", "-c", "date +%s >> /state/exec.requests; exit 0"}}}
		})
		t.waitRequests(ctx, "exec", 1, 30*time.Second)
		t.waitState(ctx, true, true)
		t.stableState(ctx, true, true, mutableProbeStableWindow)
	})

	ginkgo.It("should reschedule healthy liveness when its period is shortened and extended", func(ctx context.Context) {
		t := newMutableProbeTest(f, false)
		probe := mutableHTTPProbe("liveness")
		probe.PeriodSeconds = 30
		t.container(t.pod).LivenessProbe = probe

		t.create(ctx)
		stamps := t.waitRequests(ctx, "liveness", 2, mutableProbeTimeout)
		completed := t.waitResults(ctx, "liveness", len(stamps))
		expectMutableProbeInterval(completed[0], stamps[1], 25*time.Second, 45*time.Second)

		for _, period := range []int32{5, 30} {
			previous := completed[len(stamps)-1]
			count := len(stamps)

			t.update(ctx, func(c *v1.Container) { c.LivenessProbe.PeriodSeconds = period })
			stamps = t.waitRequests(ctx, "liveness", count+2, mutableProbeTimeout)
			completed = t.waitResults(ctx, "liveness", len(stamps))
			low, high := 3*time.Second, 15*time.Second
			if period == 30 {
				low, high = 25*time.Second, 45*time.Second
			}

			expectMutableProbeInterval(previous, stamps[count], low, high)
			expectMutableProbeInterval(completed[count], stamps[count+1], low, high)
		}

		t.stableState(ctx, true, true, mutableProbeStableWindow)
	})

	for _, shorten := range []bool{true, false} {
		action := "extend"
		if shorten {
			action = "shorten"
		}

		ginkgo.It("should "+action+" initial delay before the first probe", func(ctx context.Context) {
			t := newMutableProbeTest(f, false)
			oldDelay, newDelay := int32(60), int32(90)
			if shorten {
				oldDelay, newDelay = 120, 30
			}

			probe := mutableHTTPProbe("liveness")
			probe.InitialDelaySeconds = oldDelay
			t.container(t.pod).LivenessProbe = probe

			t.create(ctx)
			started := t.identity.State.Running.StartedAt.Time

			gomega.Expect(time.Since(started)).To(gomega.BeNumerically("<", 25*time.Second), "container must be observed before either initial delay expires")
			gomega.Expect(t.requests("liveness")).To(gomega.BeEmpty())

			t.update(ctx, func(c *v1.Container) { c.LivenessProbe.InitialDelaySeconds = newDelay })
			stamps := t.waitRequests(ctx, "liveness", 1, mutableProbeTimeout)
			expectMutableProbeInterval(started, stamps[0], time.Duration(newDelay)*time.Second-time.Second, time.Duration(newDelay)*time.Second+15*time.Second)
			t.stableState(ctx, true, true, mutableProbeStableWindow)
		})
	}

	ginkgo.It("should not reapply initial delay once probing has begun", func(ctx context.Context) {
		t := newMutableProbeTest(f, false)
		t.container(t.pod).LivenessProbe = mutableHTTPProbe("liveness")

		t.create(ctx)
		stamps := t.waitRequests(ctx, "liveness", 2, mutableProbeTimeout)

		t.update(ctx, func(c *v1.Container) { c.LivenessProbe.InitialDelaySeconds = 3600 })
		t.waitRequests(ctx, "liveness", len(stamps)+3, 20*time.Second)
		t.stableState(ctx, true, true, mutableProbeStableWindow)
	})

	for _, success := range []bool{false, true} {
		field := "failureThreshold"
		if success {
			field = "successThreshold"
		}

		ginkgo.It("should reset accumulated readiness results when updating "+field, func(ctx context.Context) {
			t := newMutableProbeTest(f, false)
			probe := mutableHTTPProbe("readiness")

			// Each request waits for a numbered response file, so extra readiness triggers cannot consume uncounted results.
			probe.TimeoutSeconds = 60
			if success {
				probe.SuccessThreshold = 10
			} else {
				probe.FailureThreshold = 10
			}

			t.container(t.pod).ReadinessProbe = probe
			t.setup = "touch /state/readiness.controlled"

			t.create(ctx)

			result := "fail"
			if success {
				result = "success"
			}

			initialResult := "success"
			if success {
				initialResult = "fail"
			}

			t.releaseRequest(ctx, 1, initialResult)
			count := 1
			t.waitState(ctx, true, !success)

			for range 2 {
				count++
				t.releaseRequest(ctx, count, result)
			}

			// Park the next attempt while the spec update propagates. Its response is the first result under the new threshold.
			t.waitRequests(ctx, "readiness", count+1, mutableProbeTimeout)

			t.update(ctx, func(c *v1.Container) {
				if success {
					c.ReadinessProbe.SuccessThreshold = 3
				} else {
					c.ReadinessProbe.FailureThreshold = 3
				}
			})
			t.stableState(ctx, true, !success, mutableProbeStableWindow)

			for range 2 {
				count++
				t.releaseRequest(ctx, count, result)
				t.stableState(ctx, true, !success, 5*time.Second)
			}

			count++
			t.releaseRequest(ctx, count, result)
			t.waitState(ctx, true, success)
			t.stableState(ctx, true, success, 5*time.Second)
		})
	}

	ginkgo.It("should promptly apply readiness timeout changes to a delayed handler", func(ctx context.Context) {
		t := newMutableProbeTest(f, false)
		probe := mutableHTTPProbe("readiness")
		probe.PeriodSeconds = 120
		t.container(t.pod).ReadinessProbe = probe
		t.setup = "echo 3 > /state/readiness.delay"

		t.create(ctx)
		t.waitRequests(ctx, "readiness", 1, mutableProbeTimeout)
		t.waitState(ctx, true, false)
		t.stableState(ctx, true, false, 5*time.Second)

		for _, timeout := range []int32{5, 1} {
			count := len(t.requests("readiness"))

			t.update(ctx, func(c *v1.Container) { c.ReadinessProbe.TimeoutSeconds = timeout })
			t.waitRequests(ctx, "readiness", count+1, 30*time.Second)
			t.waitState(ctx, true, timeout == 5)
			t.stableState(ctx, true, timeout == 5, 5*time.Second)
		}
	})

	for _, remove := range []bool{false, true} {
		action := "update the probe grace period"
		if remove {
			action = "fall back to the Pod grace period after removing the probe override"
		}

		ginkgo.It("should "+action+" on the next liveness failure", func(ctx context.Context) {
			t := newMutableProbeTest(f, false)
			probe := mutableHTTPProbe("liveness")
			probe.TerminationGracePeriodSeconds = ptr.To[int64](30)
			t.container(t.pod).LivenessProbe = probe
			t.pod.Spec.TerminationGracePeriodSeconds = ptr.To[int64](20)
			t.setup = "touch /state/hold-term"

			t.create(ctx)
			t.waitRequests(ctx, "liveness", 2, mutableProbeTimeout)

			t.update(ctx, func(c *v1.Container) {
				c.LivenessProbe.TerminationGracePeriodSeconds = ptr.To[int64](5)
				if remove {
					c.LivenessProbe.TerminationGracePeriodSeconds = nil
				}
			})
			t.stableState(ctx, true, true, mutableProbeStableWindow)

			t.exec("touch /state/liveness.fail")
			t.waitRestart(ctx)
			terminated := t.identity.LastTerminationState.Terminated

			gomega.Expect(terminated).NotTo(gomega.BeNil())
			gomega.Expect(terminated.ExitCode).To(gomega.Equal(int32(137)), "target must survive SIGTERM and require SIGKILL")

			term, err := strconv.ParseInt(strings.TrimSpace(t.exec("cat /state/term")), 10, 64)
			framework.ExpectNoError(err, "parse SIGTERM timestamp")
			expected := 5 * time.Second
			if remove {
				expected = 20 * time.Second
			}

			expectMutableProbeInterval(time.Unix(term, 0), terminated.FinishedAt.Time, expected-2*time.Second, expected+7*time.Second)
		})
	}
})

// The existing BusyBox probe pods provide a shell and httpd. The CGI adds request timestamps
// and controlled responses without requiring a new test image or routing through a Service.
const mutableProbeServer = `
mkdir -p /www/cgi-bin
echo ready > /www/index.html
if test -e /state/seen; then touch /state/replacement; fi
touch /state/seen
# A requested liveness restart should recover, leaving time to inspect the terminated instance.
rm -f /state/liveness.fail
cat > /www/cgi-bin/probe <<'EOF'
#!/bin/sh
name="$QUERY_STRING"
date +%s >> /state/$name.requests
if test -e /state/replacement; then
  date +%s >> /state/$name-replacement.requests
else
  date +%s >> /state/$name-initial.requests
fi
result=success
if test "$name" = fail || test -e /state/$name.fail; then result=fail; fi
if test -e /state/$name.controlled; then
  number=$(wc -l < /state/$name.requests | tr -d ' ')
  until test -e /state/$name.release.$number; do sleep 0.1; done
  result=$(cat /state/$name.release.$number)
fi
if test -e /state/$name.delay; then sleep "$(cat /state/$name.delay)"; fi
date +%s >> /state/$name.results
if test "$result" = fail; then
  printf 'Status: 500 Internal Server Error\r\n'
fi
printf 'Content-Type: text/plain\r\n\r\n%s\n' "$result"
EOF
chmod +x /www/cgi-bin/probe
trap 'date +%s > /state/term; if ! test -e /state/hold-term; then exit 0; fi' TERM
httpd -f -p 8080 -h /www &
while true; do
  if test -e /state/restart; then rm /state/restart; exit 1; fi
  sleep 1 &
  wait $!
done
`

type mutableProbeTest struct {
	f        *framework.Framework
	client   *e2epod.PodClient
	pod      *v1.Pod
	identity v1.ContainerStatus
	setup    string
}

func newMutableProbeTest(f *framework.Framework, sidecar bool) *mutableProbeTest {
	command := []string{"sh", "-c", mutableProbeServer}
	pod := busyBoxPodSpec(nil, nil, command)
	if sidecar {
		pod = busyBoxSidecarPodSpec(nil, nil, command)
	}

	pod.Spec.NodeSelector = map[string]string{"kubernetes.io/os": "linux"}
	pod.Spec.TerminationGracePeriodSeconds = ptr.To[int64](1)
	pod.Spec.Volumes = []v1.Volume{{Name: "state", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}}
	t := &mutableProbeTest{f: f, client: e2epod.NewPodClient(f), pod: pod}
	t.container(pod).VolumeMounts = []v1.VolumeMount{{Name: "state", MountPath: "/state"}}

	return t
}

func mutableHTTPProbe(name string) *v1.Probe {
	return &v1.Probe{
		ProbeHandler: v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{Path: "/cgi-bin/probe?" + name, Port: intstr.FromInt32(8080), Scheme: v1.URISchemeHTTP}},
		// Leave time for httpd to bind before liveness can fail the initial container.
		InitialDelaySeconds: 5,
		PeriodSeconds:       2, TimeoutSeconds: 1, SuccessThreshold: 1, FailureThreshold: 1,
	}
}

func mutableStartupProbe(name string) *v1.Probe {
	probe := mutableHTTPProbe(name)

	// Keep failing startup alive while API propagation and negative assertions complete.
	probe.FailureThreshold = 300
	return probe
}

func (t *mutableProbeTest) container(pod *v1.Pod) *v1.Container {
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == "busybox" {
			return &pod.Spec.Containers[i]
		}
	}

	for i := range pod.Spec.InitContainers {
		if pod.Spec.InitContainers[i].Name == "busybox" {
			return &pod.Spec.InitContainers[i]
		}
	}

	framework.Failf("Pod %s has no busybox probe target", pod.Name)

	return nil
}

func (t *mutableProbeTest) create(ctx context.Context) {
	if t.setup != "" {
		t.container(t.pod).Command[2] = t.setup + "\n" + mutableProbeServer
	}

	t.pod = t.client.Create(ctx, t.pod)
	t.captureIdentity(ctx)
}

func (t *mutableProbeTest) status(ctx context.Context) v1.ContainerStatus {
	pod, err := t.client.Get(ctx, t.pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get mutable probe Pod")

	gomega.Expect(pod.UID).To(gomega.Equal(t.pod.UID), "probe update must preserve Pod UID")

	for _, statuses := range [][]v1.ContainerStatus{pod.Status.ContainerStatuses, pod.Status.InitContainerStatuses} {
		for _, s := range statuses {
			if s.Name == "busybox" {
				return s
			}
		}
	}

	return v1.ContainerStatus{}
}

func (t *mutableProbeTest) captureIdentity(ctx context.Context) {
	gomega.Eventually(ctx, func() bool {
		t.identity = t.status(ctx)
		return t.identity.State.Running != nil && t.identity.ContainerID != ""
	}, mutableProbeTimeout, time.Second).Should(gomega.BeTrue(), "probe target must be running")
	gomega.Expect(t.identity.RestartCount).To(gomega.BeZero(), "target restarted before the test observed it")

	// Avoid changing configuration before the HTTP server can serve probes.
	gomega.Eventually(ctx, func() error {
		_, _, err := e2epod.ExecCommandInContainerWithFullOutput(t.f, t.pod.Name, "busybox", "sh", "-c", "wget -q -O /dev/null http://127.0.0.1:8080/")
		return err
	}, 30*time.Second, time.Second).Should(gomega.Succeed())
}

func (t *mutableProbeTest) checkIdentity(ctx context.Context) v1.ContainerStatus {
	s := t.status(ctx)

	gomega.Expect(s.ContainerID).To(gomega.Equal(t.identity.ContainerID), "probe update must preserve ContainerID")
	gomega.Expect(s.RestartCount).To(gomega.Equal(t.identity.RestartCount), "unexpected restart during probe update")
	gomega.Expect(s.State.Running).NotTo(gomega.BeNil(), "probe target unexpectedly stopped")

	return s
}

func (t *mutableProbeTest) waitState(ctx context.Context, started, ready bool) {
	gomega.Eventually(ctx, func() []any {
		s := t.checkIdentity(ctx)
		return []any{s.Started, s.Ready}
	}, mutableProbeTimeout, time.Second).Should(gomega.Equal([]any{ptr.To(started), ready}))
}

func (t *mutableProbeTest) stableState(ctx context.Context, started, ready bool, duration time.Duration) {
	gomega.Consistently(ctx, func() []any {
		s := t.checkIdentity(ctx)
		return []any{s.Started, s.Ready}
	}, duration, time.Second).Should(gomega.Equal([]any{ptr.To(started), ready}))
}

func (t *mutableProbeTest) update(ctx context.Context, mutate func(*v1.Container)) {
	ginkgo.By("Updating probes in the original Pod spec and reading them back")
	var expected *v1.Container
	t.client.Update(ctx, t.pod.Name, func(pod *v1.Pod) {
		c := t.container(pod)
		mutate(c)
		expected = c.DeepCopy()
	})

	pod, err := t.client.Get(ctx, t.pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	gomega.Expect(pod.UID).To(gomega.Equal(t.pod.UID))

	actual := t.container(pod)
	for i, probe := range []*v1.Probe{expected.ReadinessProbe, expected.LivenessProbe, expected.StartupProbe} {
		got := []*v1.Probe{actual.ReadinessProbe, actual.LivenessProbe, actual.StartupProbe}[i]

		gomega.Expect(apiequality.Semantic.DeepEqual(got, probe)).To(gomega.BeTrue(), "probe update was not persisted: got %#v, want %#v", got, probe)
	}
}

func (t *mutableProbeTest) exec(command string) string {
	stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(t.f, t.pod.Name, "busybox", "sh", "-c", command)
	framework.ExpectNoError(err, "probe fixture command %q failed: %s", command, stderr)
	return stdout
}

func (t *mutableProbeTest) requests(name string) []time.Time {
	return t.readTimes(name + ".requests")
}

func (t *mutableProbeTest) readTimes(file string) []time.Time {
	output := t.exec("if test -e /state/" + file + "; then cat /state/" + file + "; fi")
	var stamps []time.Time
	for _, field := range strings.Fields(output) {
		seconds, err := strconv.ParseInt(field, 10, 64)
		framework.ExpectNoError(err, "parse probe timestamp %q from %s", field, file)
		stamps = append(stamps, time.Unix(seconds, 0))
	}

	return stamps
}

func (t *mutableProbeTest) waitRequests(ctx context.Context, name string, count int, timeout time.Duration) []time.Time {
	var stamps []time.Time

	gomega.Eventually(ctx, func() int {
		t.checkIdentity(ctx)
		stamps = t.requests(name)
		return len(stamps)
	}, timeout, time.Second).Should(gomega.BeNumerically(">=", count), "waiting for %d %s probe requests", count, name)

	return stamps
}

func (t *mutableProbeTest) waitResults(ctx context.Context, name string, count int) []time.Time {
	var stamps []time.Time

	gomega.Eventually(ctx, func() int {
		t.checkIdentity(ctx)
		stamps = t.readTimes(name + ".results")
		return len(stamps)
	}, 10*time.Second, time.Second).Should(gomega.BeNumerically(">=", count), "waiting for %d %s probe responses", count, name)

	return stamps
}

func (t *mutableProbeTest) releaseRequest(ctx context.Context, number int, result string) {
	t.waitRequests(ctx, "readiness", number, mutableProbeTimeout)

	// Publish the response atomically so the CGI cannot read an empty, newly created file.
	t.exec(fmt.Sprintf("echo %s > /state/response.tmp && mv /state/response.tmp /state/readiness.release.%d", result, number))

	gomega.Expect(t.waitResults(ctx, "readiness", number)).To(gomega.HaveLen(number), "controlled readiness response must complete")
}

func (t *mutableProbeTest) waitRestart(ctx context.Context) {
	previous := t.identity

	gomega.Eventually(ctx, func() bool {
		s := t.status(ctx)

		gomega.Expect(s.RestartCount).To(gomega.BeNumerically("<=", previous.RestartCount+1), "target restarted more than once")

		if s.RestartCount == previous.RestartCount+1 && s.State.Running != nil && s.ContainerID != previous.ContainerID && s.ContainerID != "" {
			t.identity = s
			return true
		}

		return false
	}, mutableProbeTimeout, time.Second).Should(gomega.BeTrue(), "waiting for one replacement container in the same Pod")
}

func expectMutableProbeInterval(start, end time.Time, minimum, maximum time.Duration) {
	interval := end.Sub(start)

	gomega.Expect(interval).To(gomega.BeNumerically(">=", minimum), "probe interval from %s to %s was too short", start, end)
	gomega.Expect(interval).To(gomega.BeNumerically("<=", maximum), "probe interval from %s to %s was too long", start, end)
}
