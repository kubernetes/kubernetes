/*
Copyright 2014 The Kubernetes Authors.

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

package e2e

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	inf "gopkg.in/inf.v0"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/apps"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/petset"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	utilyaml "k8s.io/kubernetes/pkg/util/yaml"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	petsetPoll = 10 * time.Second
	// Some pets install base packages via wget
	petsetTimeout           = 10 * time.Minute
	zookeeperManifestPath   = "test/e2e/testing-manifests/petset/zookeeper"
	mysqlGaleraManifestPath = "test/e2e/testing-manifests/petset/mysql-galera"
	redisManifestPath       = "test/e2e/testing-manifests/petset/redis"
	// Should the test restart petset clusters?
	// TODO: enable when we've productionzed bringup of pets in this e2e.
	restartCluster = false
)

// Time: 25m, slow by design.
// GCE Quota requirements: 3 pds, one per pet manifest declared above.
// GCE Api requirements: nodes and master need storage r/w permissions.
var _ = framework.KubeDescribe("PetSet [Slow] [Feature:PetSet]", func() {
	f := framework.NewDefaultFramework("petset")
	var ns string
	var c *client.Client

	BeforeEach(func() {
		// PetSet is in alpha, so it's disabled on all platforms except GCE.
		// In theory, tests that restart pets should pass on any platform with a
		// dynamic volume provisioner.
		framework.SkipUnlessProviderIs("gce")

		c = f.Client
		ns = f.Namespace.Name
	})

	framework.KubeDescribe("Basic PetSet functionality", func() {
		psName := "pet"
		labels := map[string]string{
			"foo": "bar",
			"baz": "blah",
		}
		headlessSvcName := "test"

		BeforeEach(func() {
			By("creating service " + headlessSvcName + " in namespace " + ns)
			headlessService := createServiceSpec(headlessSvcName, "", true, labels)
			_, err := c.Services(ns).Create(headlessService)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				dumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all petset in ns %v", ns)
			deleteAllPetSets(c, ns)
		})

		It("should provide basic identity [Feature:PetSet]", func() {
			By("creating petset " + psName + " in namespace " + ns)
			petMounts := []api.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
			podMounts := []api.VolumeMount{{Name: "home", MountPath: "/home"}}
			ps := newPetSet(psName, ns, headlessSvcName, 3, petMounts, podMounts, labels)
			_, err := c.Apps().PetSets(ns).Create(ps)
			Expect(err).NotTo(HaveOccurred())

			pst := petSetTester{c: c}

			By("Saturating pet set " + ps.Name)
			pst.saturate(ps)

			By("Verifying petset mounted data directory is usable")
			ExpectNoError(pst.checkMount(ps, "/data"))

			cmd := "echo $(hostname) > /data/hostname; sync;"
			By("Running " + cmd + " in all pets")
			ExpectNoError(pst.execInPets(ps, cmd))

			By("Restarting pet set " + ps.Name)
			pst.restart(ps)
			pst.saturate(ps)

			By("Verifying petset mounted data directory is usable")
			ExpectNoError(pst.checkMount(ps, "/data"))

			cmd = "if [ \"$(cat /data/hostname)\" = \"$(hostname)\" ]; then exit 0; else exit 1; fi"
			By("Running " + cmd + " in all pets")
			ExpectNoError(pst.execInPets(ps, cmd))
		})

		It("should handle healthy pet restarts during scale [Feature:PetSet]", func() {
			By("creating petset " + psName + " in namespace " + ns)

			petMounts := []api.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
			podMounts := []api.VolumeMount{{Name: "home", MountPath: "/home"}}
			ps := newPetSet(psName, ns, headlessSvcName, 2, petMounts, podMounts, labels)
			_, err := c.Apps().PetSets(ns).Create(ps)
			Expect(err).NotTo(HaveOccurred())

			pst := petSetTester{c: c}

			pst.waitForRunning(1, ps)

			By("Marking pet at index 0 as healthy.")
			pst.setHealthy(ps)

			By("Waiting for pet at index 1 to enter running.")
			pst.waitForRunning(2, ps)

			// Now we have 1 healthy and 1 unhealthy pet. Deleting the healthy pet should *not*
			// create a new pet till the remaining pet becomes healthy, which won't happen till
			// we set the healthy bit.

			By("Deleting healthy pet at index 0.")
			pst.deletePetAtIndex(0, ps)

			By("Confirming pet at index 0 is not recreated.")
			pst.confirmPetCount(1, ps, 10*time.Second)

			By("Deleting unhealthy pet at index 1.")
			pst.deletePetAtIndex(1, ps)

			By("Confirming all pets in petset are created.")
			pst.saturate(ps)
		})
	})

	framework.KubeDescribe("Deploy clustered applications [Slow] [Feature:PetSet]", func() {
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce")
		})

		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				dumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all petset in ns %v", ns)
			deleteAllPetSets(c, ns)
		})

		It("should creating a working zookeeper cluster [Feature:PetSet]", func() {
			pst := &petSetTester{c: c}
			pet := &zookeeperTester{tester: pst}
			By("Deploying " + pet.name())
			ps := pet.deploy(ns)

			By("Creating foo:bar in member with index 0")
			pet.write(0, map[string]string{"foo": "bar"})

			if restartCluster {
				By("Restarting pet set " + ps.Name)
				pst.restart(ps)
				pst.waitForRunning(ps.Spec.Replicas, ps)
			}

			By("Reading value under foo from member with index 2")
			if v := pet.read(2, "foo"); v != "bar" {
				framework.Failf("Read unexpected value %v, expected bar under key foo", v)
			}
		})

		It("should creating a working redis cluster [Feature:PetSet]", func() {
			pst := &petSetTester{c: c}
			pet := &redisTester{tester: pst}
			By("Deploying " + pet.name())
			ps := pet.deploy(ns)

			By("Creating foo:bar in member with index 0")
			pet.write(0, map[string]string{"foo": "bar"})

			if restartCluster {
				By("Restarting pet set " + ps.Name)
				pst.restart(ps)
				pst.waitForRunning(ps.Spec.Replicas, ps)
			}

			By("Reading value under foo from member with index 2")
			if v := pet.read(2, "foo"); v != "bar" {
				framework.Failf("Read unexpected value %v, expected bar under key foo", v)
			}
		})

		It("should creating a working mysql cluster [Feature:PetSet]", func() {
			pst := &petSetTester{c: c}
			pet := &mysqlGaleraTester{tester: pst}
			By("Deploying " + pet.name())
			ps := pet.deploy(ns)

			By("Creating foo:bar in member with index 0")
			pet.write(0, map[string]string{"foo": "bar"})

			if restartCluster {
				By("Restarting pet set " + ps.Name)
				pst.restart(ps)
				pst.waitForRunning(ps.Spec.Replicas, ps)
			}

			By("Reading value under foo from member with index 2")
			if v := pet.read(2, "foo"); v != "bar" {
				framework.Failf("Read unexpected value %v, expected bar under key foo", v)
			}
		})
	})
})

func dumpDebugInfo(c *client.Client, ns string) {
	pl, _ := c.Pods(ns).List(api.ListOptions{LabelSelector: labels.Everything()})
	for _, p := range pl.Items {
		desc, _ := framework.RunKubectl("describe", "po", p.Name, fmt.Sprintf("--namespace=%v", ns))
		framework.Logf("\nOutput of kubectl describe %v:\n%v", p.Name, desc)

		l, _ := framework.RunKubectl("logs", p.Name, fmt.Sprintf("--namespace=%v", ns), "--tail=100")
		framework.Logf("\nLast 100 log lines of %v:\n%v", p.Name, l)
	}
}

func kubectlExecWithRetries(args ...string) (out string) {
	var err error
	for i := 0; i < 3; i++ {
		if out, err = framework.RunKubectl(args...); err == nil {
			return
		}
		framework.Logf("Retrying %v:\nerror %v\nstdout %v", args, err, out)
	}
	framework.Failf("Failed to execute \"%v\" with retries: %v", args, err)
	return
}

type petTester interface {
	deploy(ns string) *apps.PetSet
	write(petIndex int, kv map[string]string)
	read(petIndex int, key string) string
	name() string
}

type zookeeperTester struct {
	ps     *apps.PetSet
	tester *petSetTester
}

func (z *zookeeperTester) name() string {
	return "zookeeper"
}

func (z *zookeeperTester) deploy(ns string) *apps.PetSet {
	z.ps = z.tester.createPetSet(zookeeperManifestPath, ns)
	return z.ps
}

func (z *zookeeperTester) write(petIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", z.ps.Name, petIndex)
	ns := fmt.Sprintf("--namespace=%v", z.ps.Namespace)
	for k, v := range kv {
		cmd := fmt.Sprintf("/opt/zookeeper/bin/zkCli.sh create /%v %v", k, v)
		framework.Logf(framework.RunKubectlOrDie("exec", ns, name, "--", "/bin/sh", "-c", cmd))
	}
}

func (z *zookeeperTester) read(petIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", z.ps.Name, petIndex)
	ns := fmt.Sprintf("--namespace=%v", z.ps.Namespace)
	cmd := fmt.Sprintf("/opt/zookeeper/bin/zkCli.sh get /%v", key)
	return lastLine(framework.RunKubectlOrDie("exec", ns, name, "--", "/bin/sh", "-c", cmd))
}

type mysqlGaleraTester struct {
	ps     *apps.PetSet
	tester *petSetTester
}

func (m *mysqlGaleraTester) name() string {
	return "mysql: galera"
}

func (m *mysqlGaleraTester) mysqlExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/usr/bin/mysql -u root -B -e '%v'", cmd)
	// TODO: Find a readiness probe for mysql that guarantees writes will
	// succeed and ditch retries. Current probe only reads, so there's a window
	// for a race.
	return kubectlExecWithRetries(fmt.Sprintf("--namespace=%v", ns), "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (m *mysqlGaleraTester) deploy(ns string) *apps.PetSet {
	m.ps = m.tester.createPetSet(mysqlGaleraManifestPath, ns)

	framework.Logf("Deployed petset %v, initializing database", m.ps.Name)
	for _, cmd := range []string{
		"create database petset;",
		"use petset; create table pet (k varchar(20), v varchar(20));",
	} {
		framework.Logf(m.mysqlExec(cmd, ns, fmt.Sprintf("%v-0", m.ps.Name)))
	}
	return m.ps
}

func (m *mysqlGaleraTester) write(petIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", m.ps.Name, petIndex)
	for k, v := range kv {
		cmd := fmt.Sprintf("use  petset; insert into pet (k, v) values (\"%v\", \"%v\");", k, v)
		framework.Logf(m.mysqlExec(cmd, m.ps.Namespace, name))
	}
}

func (m *mysqlGaleraTester) read(petIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", m.ps.Name, petIndex)
	return lastLine(m.mysqlExec(fmt.Sprintf("use petset; select v from pet where k=\"%v\";", key), m.ps.Namespace, name))
}

type redisTester struct {
	ps     *apps.PetSet
	tester *petSetTester
}

func (m *redisTester) name() string {
	return "redis: master/slave"
}

func (m *redisTester) redisExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/opt/redis/redis-cli -h %v %v", podName, cmd)
	return framework.RunKubectlOrDie(fmt.Sprintf("--namespace=%v", ns), "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (m *redisTester) deploy(ns string) *apps.PetSet {
	m.ps = m.tester.createPetSet(redisManifestPath, ns)
	return m.ps
}

func (m *redisTester) write(petIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", m.ps.Name, petIndex)
	for k, v := range kv {
		framework.Logf(m.redisExec(fmt.Sprintf("SET %v %v", k, v), m.ps.Namespace, name))
	}
}

func (m *redisTester) read(petIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", m.ps.Name, petIndex)
	return lastLine(m.redisExec(fmt.Sprintf("GET %v", key), m.ps.Namespace, name))
}

func lastLine(out string) string {
	outLines := strings.Split(strings.Trim(out, "\n"), "\n")
	return outLines[len(outLines)-1]
}

func petSetFromManifest(fileName, ns string) *apps.PetSet {
	var ps apps.PetSet
	framework.Logf("Parsing petset from %v", fileName)
	data, err := ioutil.ReadFile(fileName)
	Expect(err).NotTo(HaveOccurred())
	json, err := utilyaml.ToJSON(data)
	Expect(err).NotTo(HaveOccurred())

	Expect(runtime.DecodeInto(api.Codecs.UniversalDecoder(), json, &ps)).NotTo(HaveOccurred())
	ps.Namespace = ns
	if ps.Spec.Selector == nil {
		ps.Spec.Selector = &unversioned.LabelSelector{
			MatchLabels: ps.Spec.Template.Labels,
		}
	}
	return &ps
}

// petSetTester has all methods required to test a single petset.
type petSetTester struct {
	c *client.Client
}

func (p *petSetTester) createPetSet(manifestPath, ns string) *apps.PetSet {
	mkpath := func(file string) string {
		return filepath.Join(framework.TestContext.RepoRoot, manifestPath, file)
	}
	ps := petSetFromManifest(mkpath("petset.yaml"), ns)

	framework.Logf(fmt.Sprintf("creating " + ps.Name + " service"))
	framework.RunKubectlOrDie("create", "-f", mkpath("service.yaml"), fmt.Sprintf("--namespace=%v", ns))

	framework.Logf(fmt.Sprintf("creating petset %v/%v with %d replicas and selector %+v", ps.Namespace, ps.Name, ps.Spec.Replicas, ps.Spec.Selector))
	framework.RunKubectlOrDie("create", "-f", mkpath("petset.yaml"), fmt.Sprintf("--namespace=%v", ns))
	p.waitForRunning(ps.Spec.Replicas, ps)
	return ps
}

func (p *petSetTester) checkMount(ps *apps.PetSet, mountPath string) error {
	for _, cmd := range []string{
		// Print inode, size etc
		fmt.Sprintf("ls -idlh %v", mountPath),
		// Print subdirs
		fmt.Sprintf("find %v", mountPath),
		// Try writing
		fmt.Sprintf("touch %v", filepath.Join(mountPath, fmt.Sprintf("%v", time.Now().UnixNano()))),
	} {
		if err := p.execInPets(ps, cmd); err != nil {
			return fmt.Errorf("failed to execute %v, error: %v", cmd, err)
		}
	}
	return nil
}

func (p *petSetTester) execInPets(ps *apps.PetSet, cmd string) error {
	podList := p.getPodList(ps)
	for _, pet := range podList.Items {
		stdout, err := framework.RunHostCmd(pet.Namespace, pet.Name, cmd)
		framework.Logf("stdout of %v on %v: %v", cmd, pet.Name, stdout)
		if err != nil {
			return err
		}
	}
	return nil
}

func (p *petSetTester) saturate(ps *apps.PetSet) {
	// TODO: Watch events and check that creation timestamps don't overlap
	for i := 0; i < ps.Spec.Replicas; i++ {
		framework.Logf("Waiting for pet at index " + fmt.Sprintf("%v", i+1) + " to enter Running")
		p.waitForRunning(i+1, ps)
		framework.Logf("Marking pet at index " + fmt.Sprintf("%v", i) + " healthy")
		p.setHealthy(ps)
	}
}

func (p *petSetTester) deletePetAtIndex(index int, ps *apps.PetSet) {
	// TODO: we won't use "-index" as the name strategy forever,
	// pull the name out from an identity mapper.
	name := fmt.Sprintf("%v-%v", ps.Name, index)
	noGrace := int64(0)
	if err := p.c.Pods(ps.Namespace).Delete(name, &api.DeleteOptions{GracePeriodSeconds: &noGrace}); err != nil {
		framework.Failf("Failed to delete pet %v for PetSet %v: %v", name, ps.Name, ps.Namespace, err)
	}
}

func (p *petSetTester) scale(ps *apps.PetSet, count int) error {
	name := ps.Name
	ns := ps.Namespace
	p.update(ns, name, func(ps *apps.PetSet) { ps.Spec.Replicas = count })

	var petList *api.PodList
	pollErr := wait.PollImmediate(petsetPoll, petsetTimeout, func() (bool, error) {
		petList = p.getPodList(ps)
		if len(petList.Items) == count {
			return true, nil
		}
		return false, nil
	})
	if pollErr != nil {
		unhealthy := []string{}
		for _, pet := range petList.Items {
			delTs, phase, readiness := pet.DeletionTimestamp, pet.Status.Phase, api.IsPodReady(&pet)
			if delTs != nil || phase != api.PodRunning || !readiness {
				unhealthy = append(unhealthy, fmt.Sprintf("%v: deletion %v, phase %v, readiness %v", pet.Name, delTs, phase, readiness))
			}
		}
		return fmt.Errorf("Failed to scale petset to %d in %v. Remaining pods:\n%v", count, petsetTimeout, unhealthy)
	}
	return nil
}

func (p *petSetTester) restart(ps *apps.PetSet) {
	oldReplicas := ps.Spec.Replicas
	ExpectNoError(p.scale(ps, 0))
	p.update(ps.Namespace, ps.Name, func(ps *apps.PetSet) { ps.Spec.Replicas = oldReplicas })
}

func (p *petSetTester) update(ns, name string, update func(ps *apps.PetSet)) {
	for i := 0; i < 3; i++ {
		ps, err := p.c.Apps().PetSets(ns).Get(name)
		if err != nil {
			framework.Failf("failed to get petset %q: %v", name, err)
		}
		update(ps)
		ps, err = p.c.Apps().PetSets(ns).Update(ps)
		if err == nil {
			return
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			framework.Failf("failed to update petset %q: %v", name, err)
		}
	}
	framework.Failf("too many retries draining petset %q", name)
}

func (p *petSetTester) getPodList(ps *apps.PetSet) *api.PodList {
	selector, err := unversioned.LabelSelectorAsSelector(ps.Spec.Selector)
	ExpectNoError(err)
	podList, err := p.c.Pods(ps.Namespace).List(api.ListOptions{LabelSelector: selector})
	ExpectNoError(err)
	return podList
}

func (p *petSetTester) confirmPetCount(count int, ps *apps.PetSet, timeout time.Duration) {
	start := time.Now()
	deadline := start.Add(timeout)
	for t := time.Now(); t.Before(deadline); t = time.Now() {
		podList := p.getPodList(ps)
		petCount := len(podList.Items)
		if petCount != count {
			framework.Failf("PetSet %v scaled unexpectedly scaled to %d -> %d replicas: %+v", ps.Name, count, len(podList.Items), podList)
		}
		framework.Logf("Verifying petset %v doesn't scale past %d for another %+v", ps.Name, count, deadline.Sub(t))
		time.Sleep(1 * time.Second)
	}
}

func (p *petSetTester) waitForRunning(numPets int, ps *apps.PetSet) {
	pollErr := wait.PollImmediate(petsetPoll, petsetTimeout,
		func() (bool, error) {
			podList := p.getPodList(ps)
			if len(podList.Items) < numPets {
				framework.Logf("Found %d pets, waiting for %d", len(podList.Items), numPets)
				return false, nil
			}
			if len(podList.Items) > numPets {
				return false, fmt.Errorf("Too many pods scheduled, expected %d got %d", numPets, len(podList.Items))
			}
			for _, p := range podList.Items {
				isReady := api.IsPodReady(&p)
				if p.Status.Phase != api.PodRunning || !isReady {
					framework.Logf("Waiting for pod %v to enter %v - Ready=True, currently %v - Ready=%v", p.Name, api.PodRunning, p.Status.Phase, isReady)
					return false, nil
				}
			}
			return true, nil
		})
	if pollErr != nil {
		framework.Failf("Failed waiting for pods to enter running: %v", pollErr)
	}
}

func (p *petSetTester) setHealthy(ps *apps.PetSet) {
	podList := p.getPodList(ps)
	markedHealthyPod := ""
	for _, pod := range podList.Items {
		if pod.Status.Phase != api.PodRunning {
			framework.Failf("Found pod in %v cannot set health", pod.Status.Phase)
		}
		if isInitialized(pod) {
			continue
		}
		if markedHealthyPod != "" {
			framework.Failf("Found multiple non-healthy pets: %v and %v", pod.Name, markedHealthyPod)
		}
		p, err := framework.UpdatePodWithRetries(p.c, pod.Namespace, pod.Name, func(up *api.Pod) {
			up.Annotations[petset.PetSetInitAnnotation] = "true"
		})
		ExpectNoError(err)
		framework.Logf("Set annotation %v to %v on pod %v", petset.PetSetInitAnnotation, p.Annotations[petset.PetSetInitAnnotation], pod.Name)
		markedHealthyPod = pod.Name
	}
}

func deleteAllPetSets(c *client.Client, ns string) {
	pst := &petSetTester{c: c}
	psList, err := c.Apps().PetSets(ns).List(api.ListOptions{LabelSelector: labels.Everything()})
	ExpectNoError(err)

	// Scale down each petset, then delete it completely.
	// Deleting a pvc without doing this will leak volumes, #25101.
	errList := []string{}
	for _, ps := range psList.Items {
		framework.Logf("Scaling petset %v to 0", ps.Name)
		if err := pst.scale(&ps, 0); err != nil {
			errList = append(errList, fmt.Sprintf("%v", err))
		}
		framework.Logf("Deleting petset %v", ps.Name)
		if err := c.Apps().PetSets(ps.Namespace).Delete(ps.Name, nil); err != nil {
			errList = append(errList, fmt.Sprintf("%v", err))
		}
	}

	// pvs are global, so we need to wait for the exact ones bound to the petset pvcs.
	pvNames := sets.NewString()
	// TODO: Don't assume all pvcs in the ns belong to a petset
	pvcPollErr := wait.PollImmediate(petsetPoll, petsetTimeout, func() (bool, error) {
		pvcList, err := c.PersistentVolumeClaims(ns).List(api.ListOptions{LabelSelector: labels.Everything()})
		if err != nil {
			framework.Logf("WARNING: Failed to list pvcs, retrying %v", err)
			return false, nil
		}
		for _, pvc := range pvcList.Items {
			pvNames.Insert(pvc.Spec.VolumeName)
			// TODO: Double check that there are no pods referencing the pvc
			framework.Logf("Deleting pvc: %v with volume %v", pvc.Name, pvc.Spec.VolumeName)
			if err := c.PersistentVolumeClaims(ns).Delete(pvc.Name); err != nil {
				return false, nil
			}
		}
		return true, nil
	})
	if pvcPollErr != nil {
		errList = append(errList, fmt.Sprintf("Timeout waiting for pvc deletion."))
	}

	pollErr := wait.PollImmediate(petsetPoll, petsetTimeout, func() (bool, error) {
		pvList, err := c.PersistentVolumes().List(api.ListOptions{LabelSelector: labels.Everything()})
		if err != nil {
			framework.Logf("WARNING: Failed to list pvs, retrying %v", err)
			return false, nil
		}
		waitingFor := []string{}
		for _, pv := range pvList.Items {
			if pvNames.Has(pv.Name) {
				waitingFor = append(waitingFor, fmt.Sprintf("%v: %+v", pv.Name, pv.Status))
			}
		}
		if len(waitingFor) == 0 {
			return true, nil
		}
		framework.Logf("Still waiting for pvs of petset to disappear:\n%v", strings.Join(waitingFor, "\n"))
		return false, nil
	})
	if pollErr != nil {
		errList = append(errList, fmt.Sprintf("Timeout waiting for pv provisioner to delete pvs, this might mean the test leaked pvs."))
	}
	if len(errList) != 0 {
		ExpectNoError(fmt.Errorf("%v", strings.Join(errList, "\n")))
	}
}

func ExpectNoError(err error) {
	Expect(err).NotTo(HaveOccurred())
}

func isInitialized(pod api.Pod) bool {
	initialized, ok := pod.Annotations[petset.PetSetInitAnnotation]
	if !ok {
		return false
	}
	inited, err := strconv.ParseBool(initialized)
	if err != nil {
		framework.Failf("Couldn't parse petset init annotations %v", initialized)
	}
	return inited
}

func dec(i int64, exponent int) *inf.Dec {
	return inf.NewDec(i, inf.Scale(-exponent))
}

func newPVC(name string) api.PersistentVolumeClaim {
	return api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name: name,
			Annotations: map[string]string{
				"volume.alpha.kubernetes.io/storage-class": "anything",
			},
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceStorage: *resource.NewQuantity(1, resource.BinarySI),
				},
			},
		},
	}
}

func newPetSet(name, ns, governingSvcName string, replicas int, petMounts []api.VolumeMount, podMounts []api.VolumeMount, labels map[string]string) *apps.PetSet {
	mounts := append(petMounts, podMounts...)
	claims := []api.PersistentVolumeClaim{}
	for _, m := range petMounts {
		claims = append(claims, newPVC(m.Name))
	}

	vols := []api.Volume{}
	for _, m := range podMounts {
		vols = append(vols, api.Volume{
			Name: m.Name,
			VolumeSource: api.VolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: fmt.Sprintf("/tmp/%v", m.Name),
				},
			},
		})
	}

	return &apps.PetSet{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "PetSet",
			APIVersion: "apps/v1beta1",
		},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: apps.PetSetSpec{
			Selector: &unversioned.LabelSelector{
				MatchLabels: labels,
			},
			Replicas: replicas,
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: labels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:         "nginx",
							Image:        "gcr.io/google_containers/nginx-slim:0.7",
							VolumeMounts: mounts,
						},
					},
					Volumes: vols,
				},
			},
			VolumeClaimTemplates: claims,
			ServiceName:          governingSvcName,
		},
	}
}
