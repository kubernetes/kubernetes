package scheduler

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	// storagev1 removed: test uses empty storage class on PVs
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	testutil "k8s.io/kubernetes/test/utils"
)

// Reproducer for issue #116629: interaction between PodTopologySpread and VolumeBinding
func TestTopologySpread_VolumeBinding_Issue116629(t *testing.T) {
	testCtx := testutils.InitTestAPIServer(t, "issue-116629", nil)
	defer testCtx.CloseFn()

	// Start scheduler with default configuration
	testCtx = testutils.InitTestScheduler(t, testCtx)

	// Start informers; we'll start the scheduler AFTER PV/PVC statuses are observed
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	testCtx.InformerFactory.Start(ctx.Done())
	testCtx.InformerFactory.WaitForCacheSync(ctx.Done())

	// Create three nodes in different zones
	nodeA := st.MakeNode().Name("node-a").Label("topology.kubernetes.io/zone", "A").Obj()
	nodeB := st.MakeNode().Name("node-b").Label("topology.kubernetes.io/zone", "B").Obj()
	nodeC := st.MakeNode().Name("node-c").Label("topology.kubernetes.io/zone", "C").Obj()

	if _, err := testutils.CreateNode(testCtx.ClientSet, nodeA); err != nil {
		t.Fatalf("failed creating node-a: %v", err)
	}
	if _, err := testutils.CreateNode(testCtx.ClientSet, nodeB); err != nil {
		t.Fatalf("failed creating node-b: %v", err)
	}
	if _, err := testutils.CreateNode(testCtx.ClientSet, nodeC); err != nil {
		t.Fatalf("failed creating node-c: %v", err)
	}

	// Create three PVs with zone nodeAffinity (do NOT bind). Use empty StorageClassName.
	pvA := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "pv-a"},
		Spec: v1.PersistentVolumeSpec{
			StorageClassName: "",
			Capacity: v1.ResourceList{
				v1.ResourceStorage: resource.MustParse("1Gi"),
			},
			VolumeMode:  func() *v1.PersistentVolumeMode { m := v1.PersistentVolumeFilesystem; return &m }(),
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "/tmp/pv-a"},
			},
			NodeAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "topology.kubernetes.io/zone",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"A"},
						}},
					}},
				},
			},
		},
	}

	pvB := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "pv-b"},
		Spec: v1.PersistentVolumeSpec{
			StorageClassName: "",
			Capacity: v1.ResourceList{
				v1.ResourceStorage: resource.MustParse("1Gi"),
			},
			VolumeMode:  func() *v1.PersistentVolumeMode { m := v1.PersistentVolumeFilesystem; return &m }(),
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "/tmp/pv-b"},
			},
			NodeAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "topology.kubernetes.io/zone",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"B"},
						}},
					}},
				},
			},
		},
	}

	pvC := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "pv-c"},
		Spec: v1.PersistentVolumeSpec{
			StorageClassName: "",
			Capacity: v1.ResourceList{
				v1.ResourceStorage: resource.MustParse("1Gi"),
			},
			VolumeMode:  func() *v1.PersistentVolumeMode { m := v1.PersistentVolumeFilesystem; return &m }(),
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "/tmp/pv-c"},
			},
			NodeAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "topology.kubernetes.io/zone",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"C"},
						}},
					}},
				},
			},
		},
	}

	if err := testutil.CreatePersistentVolumeWithRetries(testCtx.ClientSet, pvA); err != nil {
		t.Fatalf("failed creating pv-a: %v", err)
	}
	if err := testutil.CreatePersistentVolumeWithRetries(testCtx.ClientSet, pvB); err != nil {
		t.Fatalf("failed creating pv-b: %v", err)
	}
	if err := testutil.CreatePersistentVolumeWithRetries(testCtx.ClientSet, pvC); err != nil {
		t.Fatalf("failed creating pv-c: %v", err)
	}

	// Create PVCs in the test namespace that match the PVs. Do not prebind or set ClaimRef/VolumeName.
	pvcA := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc-a", Namespace: testCtx.NS.Name},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			Resources: v1.VolumeResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceStorage: resource.MustParse("1Gi")},
			},
			VolumeMode:       func() *v1.PersistentVolumeMode { m := v1.PersistentVolumeFilesystem; return &m }(),
			StorageClassName: func() *string { s := ""; return &s }(),
		},
	}

	pvcB := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc-b", Namespace: testCtx.NS.Name},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			Resources: v1.VolumeResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceStorage: resource.MustParse("1Gi")},
			},
			VolumeMode:       func() *v1.PersistentVolumeMode { m := v1.PersistentVolumeFilesystem; return &m }(),
			StorageClassName: func() *string { s := ""; return &s }(),
		},
	}

	pvcC := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc-c", Namespace: testCtx.NS.Name},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			Resources: v1.VolumeResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceStorage: resource.MustParse("1Gi")},
			},
			VolumeMode:       func() *v1.PersistentVolumeMode { m := v1.PersistentVolumeFilesystem; return &m }(),
			StorageClassName: func() *string { s := ""; return &s }(),
		},
	}

	// Create PVCs and wait until each becomes Bound (let PV controller perform binding).
	if err := testutil.CreatePersistentVolumeClaimWithRetries(testCtx.ClientSet, testCtx.NS.Name, pvcA); err != nil {
		t.Fatalf("failed creating pvc-a: %v", err)
	}
	if err := wait.PollUntilContextTimeout(ctx, time.Second, 30*time.Second, true, func(c context.Context) (bool, error) {
		p, err := testCtx.ClientSet.CoreV1().PersistentVolumeClaims(testCtx.NS.Name).Get(c, "pvc-a", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		t.Logf("waiting for pvc-a to become Bound: phase=%s volume=%s", p.Status.Phase, p.Spec.VolumeName)
		return p.Status.Phase == v1.ClaimBound, nil
	}); err != nil {
		t.Fatalf("timed out waiting for pvc-a to become Bound: %v", err)
	}

	if err := testutil.CreatePersistentVolumeClaimWithRetries(testCtx.ClientSet, testCtx.NS.Name, pvcB); err != nil {
		t.Fatalf("failed creating pvc-b: %v", err)
	}
	if err := wait.PollUntilContextTimeout(ctx, time.Second, 30*time.Second, true, func(c context.Context) (bool, error) {
		p, err := testCtx.ClientSet.CoreV1().PersistentVolumeClaims(testCtx.NS.Name).Get(c, "pvc-b", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		t.Logf("waiting for pvc-b to become Bound: phase=%s volume=%s", p.Status.Phase, p.Spec.VolumeName)
		return p.Status.Phase == v1.ClaimBound, nil
	}); err != nil {
		t.Fatalf("timed out waiting for pvc-b to become Bound: %v", err)
	}

	if err := testutil.CreatePersistentVolumeClaimWithRetries(testCtx.ClientSet, testCtx.NS.Name, pvcC); err != nil {
		t.Fatalf("failed creating pvc-c: %v", err)
	}
	if err := wait.PollUntilContextTimeout(ctx, time.Second, 30*time.Second, true, func(c context.Context) (bool, error) {
		p, err := testCtx.ClientSet.CoreV1().PersistentVolumeClaims(testCtx.NS.Name).Get(c, "pvc-c", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		t.Logf("waiting for pvc-c to become Bound: phase=%s volume=%s", p.Status.Phase, p.Spec.VolumeName)
		return p.Status.Phase == v1.ClaimBound, nil
	}); err != nil {
		t.Fatalf("timed out waiting for pvc-c to become Bound: %v", err)
	}

	// Create a dedicated pv-new and a matching pvc-new; do not prebind or modify existing pv-a/pvc-a
	pvNew := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "pv-new"},
		Spec: v1.PersistentVolumeSpec{
			StorageClassName: "",
			Capacity: v1.ResourceList{
				v1.ResourceStorage: resource.MustParse("1Gi"),
			},
			VolumeMode:  func() *v1.PersistentVolumeMode { m := v1.PersistentVolumeFilesystem; return &m }(),
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "/tmp/pv-new"},
			},
			NodeAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{{
						MatchExpressions: []v1.NodeSelectorRequirement{{
							Key:      "topology.kubernetes.io/zone",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"A"},
						}},
					}},
				},
			},
		},
	}
	if err := testutil.CreatePersistentVolumeWithRetries(testCtx.ClientSet, pvNew); err != nil {
		t.Fatalf("failed creating pv-new: %v", err)
	}

	pvcNew := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc-new", Namespace: testCtx.NS.Name},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
			Resources: v1.VolumeResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceStorage: resource.MustParse("1Gi")},
			},
			VolumeMode:       func() *v1.PersistentVolumeMode { m := v1.PersistentVolumeFilesystem; return &m }(),
			StorageClassName: func() *string { s := ""; return &s }(),
		},
	}
	if err := testutil.CreatePersistentVolumeClaimWithRetries(testCtx.ClientSet, testCtx.NS.Name, pvcNew); err != nil {
		t.Fatalf("failed creating pvc-new: %v", err)
	}
	if err := wait.PollUntilContextTimeout(ctx, time.Second, 30*time.Second, true, func(c context.Context) (bool, error) {
		p, err := testCtx.ClientSet.CoreV1().PersistentVolumeClaims(testCtx.NS.Name).Get(c, "pvc-new", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		t.Logf("waiting for pvc-new to become Bound: phase=%s volume=%s", p.Status.Phase, p.Spec.VolumeName)
		return p.Status.Phase == v1.ClaimBound, nil
	}); err != nil {
		t.Fatalf("timed out waiting for pvc-new to become Bound: %v", err)
	}

	// namespace for pod creation
	ns := testCtx.NS.Name

	// Wait for informers/scheduler caches to observe the PV/PVC status updates.
	if err := wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (bool, error) {
		// verify PVs
		for _, pvName := range []string{"pv-a", "pv-b", "pv-c"} {
			pv, err := testCtx.ClientSet.CoreV1().PersistentVolumes().Get(context.Background(), pvName, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			if pv.Status.Phase != v1.VolumeBound {
				return false, nil
			}
		}
		// verify PVCs
		for _, pvcName := range []string{"pvc-a", "pvc-b", "pvc-c"} {
			pvc, err := testCtx.ClientSet.CoreV1().PersistentVolumeClaims(testCtx.NS.Name).Get(context.Background(), pvcName, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			if pvc.Status.Phase != v1.ClaimBound {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		t.Fatalf("timed out waiting for PV/PVC status to be observed as Bound: %v", err)
	}

	// Now that PV/PVC statuses are observed as Bound by informers, start the scheduler
	go testCtx.Scheduler.Run(ctx)

	// Existing pods (one marked terminating)
	podA := st.MakePod().Name("pod-a").Namespace(testCtx.NS.Name).PVC("pvc-a").Label("app", "sset").Obj()
	podA.Spec.TopologySpreadConstraints = []v1.TopologySpreadConstraint{{
		MaxSkew:           1,
		TopologyKey:       "topology.kubernetes.io/zone",
		WhenUnsatisfiable: v1.DoNotSchedule,
		LabelSelector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "sset"}},
	}}
	podA.Spec.NodeName = "node-a"
	podB := st.MakePod().Name("pod-b").Namespace(testCtx.NS.Name).PVC("pvc-b").Label("app", "sset").Obj()
	podB.Spec.TopologySpreadConstraints = []v1.TopologySpreadConstraint{{
		MaxSkew:           1,
		TopologyKey:       "topology.kubernetes.io/zone",
		WhenUnsatisfiable: v1.DoNotSchedule,
		LabelSelector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "sset"}},
	}}
	podB.Spec.NodeName = "node-b"
	podC := st.MakePod().Name("pod-c").Namespace(testCtx.NS.Name).PVC("pvc-c").Label("app", "sset").Terminating().Obj()
	podC.Spec.TopologySpreadConstraints = []v1.TopologySpreadConstraint{{
		MaxSkew:           1,
		TopologyKey:       "topology.kubernetes.io/zone",
		WhenUnsatisfiable: v1.DoNotSchedule,
		LabelSelector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "sset"}},
	}}
	podC.Spec.NodeName = "node-c"

	// Ensure pods have at least one container (validation requires this)
	if len(podA.Spec.Containers) == 0 {
		podA.Spec.Containers = []v1.Container{{Name: "pause", Image: "k8s.gcr.io/pause:3.9"}}
	}
	if len(podB.Spec.Containers) == 0 {
		podB.Spec.Containers = []v1.Container{{Name: "pause", Image: "k8s.gcr.io/pause:3.9"}}
	}
	if len(podC.Spec.Containers) == 0 {
		podC.Spec.Containers = []v1.Container{{Name: "pause", Image: "k8s.gcr.io/pause:3.9"}}
	}

	if _, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Create(context.Background(), podA, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed creating pod-a: %v", err)
	}
	if _, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Create(context.Background(), podB, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed creating pod-b: %v", err)
	}
	if _, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Create(context.Background(), podC, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed creating pod-c: %v", err)
	}

	// New pod with topologySpreadConstraints using pvc-a
	newPod := st.MakePod().Name("pod-new").Namespace(ns).PVC("pvc-new").Label("app", "sset").Obj()
	newPod.Spec.TopologySpreadConstraints = []v1.TopologySpreadConstraint{{
		MaxSkew:           1,
		TopologyKey:       "topology.kubernetes.io/zone",
		WhenUnsatisfiable: v1.DoNotSchedule,
		LabelSelector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "sset"}},
	}}
	if len(newPod.Spec.Containers) == 0 {
		newPod.Spec.Containers = []v1.Container{{Name: "pause", Image: "k8s.gcr.io/pause:3.9"}}
	}

	// Ensure the pod is scheduled by the default scheduler in the test harness
	newPod.Spec.SchedulerName = "default-scheduler"
	if _, err := testCtx.ClientSet.CoreV1().Pods(ns).Create(context.Background(), newPod, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed creating new pod: %v", err)
	}

	// Run one scheduling cycle: wait for pod to be either scheduled or marked unschedulable
	if err := wait.PollImmediate(100*time.Millisecond, 20*time.Second, func() (bool, error) {
		p, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Get(context.Background(), "pod-new", metav1.GetOptions{})
		if err != nil {
			t.Logf("poll: failed getting pod-new: %v", err)
			return false, nil
		}

		// Diagnostic logging each iteration
		t.Logf("poll: pod phase=%s spec.nodeName=%q", p.Status.Phase, p.Spec.NodeName)
		for i := range p.Status.Conditions {
			c := p.Status.Conditions[i]
			t.Logf("poll: condition type=%s status=%s reason=%q message=%q", c.Type, c.Status, c.Reason, c.Message)
		}

		// Fetch and log pod events
		fieldSel := "involvedObject.name=pod-new"
		evList, evErr := testCtx.ClientSet.CoreV1().Events(testCtx.NS.Name).List(context.Background(), metav1.ListOptions{FieldSelector: fieldSel})
		if evErr != nil {
			t.Logf("poll: failed listing events for pod-new: %v", evErr)
		} else {
			for _, ev := range evList.Items {
				t.Logf("poll: event type=%s reason=%q message=%q source=%v count=%d", ev.Type, ev.Reason, ev.Message, ev.Source, ev.Count)
			}
		}

		// If NodeName is set or PodScheduled condition present (True/False), consider progress observed
		if p.Spec.NodeName != "" {
			t.Logf("poll: pod-new has been assigned to node %q", p.Spec.NodeName)
			return true, nil
		}
		for i := range p.Status.Conditions {
			c := &p.Status.Conditions[i]
			if c.Type == v1.PodScheduled {
				t.Logf("poll: PodScheduled condition seen: status=%s reason=%q message=%q", c.Status, c.Reason, c.Message)
				if c.Status == v1.ConditionTrue || c.Status == v1.ConditionFalse {
					return true, nil
				}
			}
		}
		return false, nil
	}); err != nil {
		t.Fatalf("expected pod-new to become scheduled or unschedulable within timeout: %v", err)
	}

	// Fetch pod and assert it was successfully scheduled to node-a (zone A).
	// Regression assertion for issue #116629: without the fix, PodTopologySpread would
	// set minMatchNum=0 (because pod-c on zone C is terminating and was excluded from
	// zone counts), making skew(A)=1+1-0=2 > maxSkew(1) → pod-new permanently
	// unschedulable even though pvc-new is bound to pv-new which only fits zone A.
	pod, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Get(context.Background(), "pod-new", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed getting pod-new: %v", err)
	}

	if pod.Spec.NodeName == "" {
		// Collect the PodScheduled condition reason for a clear failure message.
		var reason, msg string
		for i := range pod.Status.Conditions {
			c := &pod.Status.Conditions[i]
			if c.Type == v1.PodScheduled {
				reason, msg = c.Reason, c.Message
				break
			}
		}
		t.Errorf("pod-new was not scheduled (PodScheduled reason=%q msg=%q); "+
			"expected it to land on node-a (zone A) — this may indicate the #116629 "+
			"minMatchNum deflation regression: a terminating pod in zone C should not "+
			"inflate the skew of zone A", reason, msg)
	} else if pod.Spec.NodeName != "node-a" {
		t.Errorf("pod-new was scheduled to %q but expected node-a (zone A); "+
			"pvc-new is bound to pv-new which has zone-A node affinity, "+
			"VolumeBinding should restrict scheduling to node-a", pod.Spec.NodeName)
	} else {
		t.Logf("pod-new was correctly scheduled to node-a (zone A)")
	}
}
