package csiinlinevolumesecurity

import (
	"context"
	"fmt"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	corev1listers "k8s.io/client-go/listers/core/v1"
	storagev1listers "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	appsapi "k8s.io/kubernetes/pkg/apis/apps"
	batchapi "k8s.io/kubernetes/pkg/apis/batch"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	podsecapi "k8s.io/pod-security-admission/api"
)

const (
	defaultNamespaceName = "test-namespace"
	defaultCSIDriverName = "test-driver"

	// expected error string when privileged namespace is required
	privNamespaceRequiredError = "has a pod security enforce level that is lower than privileged"
)

func getMockCSIInlineVolSec(namespace *corev1.Namespace, driver *storagev1.CSIDriver) (*csiInlineVolSec, error) {
	c := &csiInlineVolSec{
		Handler: admission.NewHandler(admission.Create),
		defaultPolicy: podsecapi.Policy{
			Enforce: podsecapi.LevelVersion{
				Level:   defaultPodSecEnforceProfile,
				Version: podsecapi.GetAPIVersion(),
			},
			Warn: podsecapi.LevelVersion{
				Level:   defaultPodSecWarnProfile,
				Version: podsecapi.GetAPIVersion(),
			},
			Audit: podsecapi.LevelVersion{
				Level:   defaultPodSecAuditProfile,
				Version: podsecapi.GetAPIVersion(),
			},
		},
		nsLister:            fakeNamespaceLister(namespace),
		nsListerSynced:      func() bool { return true },
		csiDriverLister:     fakeCSIDriverLister(driver),
		csiDriverListSynced: func() bool { return true },
		podSpecExtractor:    &OCPPodSpecExtractor{},
	}
	if err := c.ValidateInitialization(); err != nil {
		return nil, err
	}

	return c, nil
}

func fakeNamespaceLister(ns *corev1.Namespace) corev1listers.NamespaceLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	_ = indexer.Add(ns)
	return corev1listers.NewNamespaceLister(indexer)
}

func fakeCSIDriverLister(driver *storagev1.CSIDriver) storagev1listers.CSIDriverLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	_ = indexer.Add(driver)
	return storagev1listers.NewCSIDriverLister(indexer)
}

func TestValidate(t *testing.T) {
	type TestStruct struct {
		name          string
		obj           runtime.Object
		namespace     *corev1.Namespace
		driver        *storagev1.CSIDriver
		expectedError error
	}

	tests := []TestStruct{
		{
			name:      "should allow pods with no volumes",
			obj:       testPod(),
			namespace: testNamespaceNoLabels(),
			driver:    testCSIDriverNoLabels(),
		},
		{
			name:      "should allow pods with inline volumes in a baseline namespace when the driver uses the baseline label",
			obj:       testPodWithInlineVol(),
			namespace: testNamespaceBaseline(),
			driver:    testCSIDriverBaseline(),
		},
		{
			name:      "should allow pods with inline volumes in a baseline namespace when the driver uses the restricted label",
			obj:       testPodWithInlineVol(),
			namespace: testNamespaceBaseline(),
			driver:    testCSIDriverRestricted(),
		},
		{
			name:          "should deny pod admission with inline volumes if the CSI driver is not found and namespace is restricted",
			obj:           testPodWithInvalidDriverName(),
			namespace:     testNamespaceRestricted(),
			driver:        testCSIDriverRestricted(),
			expectedError: fmt.Errorf(privNamespaceRequiredError),
		},
		{
			name:      "should allow pod admission with inline volumes if the CSI driver is not found and namespace is privileged",
			obj:       testPodWithInvalidDriverName(),
			namespace: testNamespacePrivileged(),
			driver:    testCSIDriverRestricted(),
		},
		{
			name:          "should deny pod admission if the CSI driver has an invalid profile label",
			obj:           testPodWithInlineVol(),
			namespace:     testNamespaceBaseline(),
			driver:        testCSIDriverInvalid(),
			expectedError: fmt.Errorf("invalid label security.openshift.io/csi-ephemeral-volume-profile for CSIDriver test-driver: must be one of privileged, baseline, restricted"),
		},
		{
			name:          "should deny pod admission if the namespace has an invalid profile label",
			obj:           testPodWithInlineVol(),
			namespace:     testNamespaceInvalid(),
			driver:        testCSIDriverRestricted(),
			expectedError: fmt.Errorf("Invalid value: \"invalid-value\": must be one of privileged, baseline, restricted"),
		},
		{
			name:      "should ignore types that do not have a pod spec",
			obj:       &coreapi.Service{},
			namespace: testNamespaceNoLabels(),
			driver:    testCSIDriverNoLabels(),
		},
	}

	podSpecableObjects := []struct {
		name string
		obj  runtime.Object
	}{
		{"Pod", &coreapi.Pod{}},
		{"PodTemplate", &coreapi.PodTemplate{}},
		{"ReplicationController", &coreapi.ReplicationController{}},
		{"ReplicaSet", &appsapi.ReplicaSet{}},
		{"Deployment", &appsapi.Deployment{}},
		{"DaemonSet", &appsapi.DaemonSet{}},
		{"StatefulSet", &appsapi.StatefulSet{}},
		{"Job", &batchapi.Job{}},
		{"CronJob", &batchapi.CronJob{}},
	}

	// Add a standard subset of the tests for each supported object type
	for _, pso := range podSpecableObjects {
		objTests := []TestStruct{
			{
				name:          fmt.Sprintf("should deny %s admission by default when it has an inline volume and no policy is defined", pso.name),
				obj:           createPodControllerObject(pso.obj, testPodWithInlineVol()),
				namespace:     testNamespaceNoLabels(),
				driver:        testCSIDriverNoLabels(),
				expectedError: fmt.Errorf(privNamespaceRequiredError),
			},
			{
				name:          fmt.Sprintf("should deny %s admission with inline volumes in a baseline namespace when the driver uses the privileged label", pso.name),
				obj:           createPodControllerObject(pso.obj, testPodWithInlineVol()),
				namespace:     testNamespaceBaseline(),
				driver:        testCSIDriverPrivileged(),
				expectedError: fmt.Errorf(privNamespaceRequiredError),
			},
			{
				name:      fmt.Sprintf("should allow %s with only persistent volume claims", pso.name),
				obj:       createPodControllerObject(pso.obj, testPodWithPVC()),
				namespace: testNamespaceNoLabels(),
				driver:    testCSIDriverNoLabels(),
			},
			{
				name:      fmt.Sprintf("should allow %s with inline volumes when running in a privileged namespace", pso.name),
				obj:       createPodControllerObject(pso.obj, testPodWithInlineVol()),
				namespace: testNamespacePrivileged(),
				driver:    testCSIDriverNoLabels(),
			},
			{
				name:      fmt.Sprintf("should allow %s with inline volumes in a restricted namespace when the driver uses the restricted label", pso.name),
				obj:       createPodControllerObject(pso.obj, testPodWithInlineVol()),
				namespace: testNamespaceRestricted(),
				driver:    testCSIDriverRestricted(),
			},
		}

		tests = append(tests, objTests...)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c, err := getMockCSIInlineVolSec(test.namespace, test.driver)
			if err != nil {
				t.Fatalf("%s: failed getMockCSIInlineVolSec: %v", test.name, err)
			}

			ns := test.namespace.Name
			name := test.obj.(metav1.Object).GetName()
			gvr := getObjectGroupVersionResource(test.obj)
			attrs := admission.NewAttributesRecord(test.obj, nil, schema.GroupVersionKind{}, ns, name, gvr, "", admission.Create, nil, false, fakeUser())

			err = c.Validate(context.TODO(), attrs, nil)
			if err != nil {
				if test.expectedError == nil {
					t.Fatalf("%s: admission controller returned error: %v", test.name, err)
				}

				if !strings.Contains(err.Error(), test.expectedError.Error()) {
					t.Fatalf("%s: the expected error %v, got %v", test.name, test.expectedError, err)
				}
			}

			if err == nil && test.expectedError != nil {
				t.Fatalf("%s: the expected error %v, got nil", test.name, test.expectedError)
			}
		})
	}
}

func fakeUser() user.Info {
	return &user.DefaultInfo{
		Name: "testuser",
	}
}

func testNamespaceNoLabels() *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: defaultNamespaceName,
		},
	}
}

func testNamespaceRestricted() *corev1.Namespace {
	ns := testNamespaceNoLabels()
	ns.Labels = map[string]string{
		"pod-security.kubernetes.io/audit":   "restricted",
		"pod-security.kubernetes.io/enforce": "restricted",
		"pod-security.kubernetes.io/warn":    "restricted",
	}
	return ns
}

func testNamespaceBaseline() *corev1.Namespace {
	ns := testNamespaceNoLabels()
	ns.Labels = map[string]string{
		"pod-security.kubernetes.io/audit":   "baseline",
		"pod-security.kubernetes.io/enforce": "baseline",
		"pod-security.kubernetes.io/warn":    "baseline",
	}
	return ns
}

func testNamespacePrivileged() *corev1.Namespace {
	ns := testNamespaceNoLabels()
	ns.Labels = map[string]string{
		"pod-security.kubernetes.io/audit":   "privileged",
		"pod-security.kubernetes.io/enforce": "privileged",
		"pod-security.kubernetes.io/warn":    "privileged",
	}
	return ns
}

func testNamespaceInvalid() *corev1.Namespace {
	ns := testNamespaceNoLabels()
	ns.Labels = map[string]string{
		"pod-security.kubernetes.io/audit":   "invalid-value",
		"pod-security.kubernetes.io/enforce": "invalid-value",
		"pod-security.kubernetes.io/warn":    "invalid-value",
	}
	return ns
}

func testCSIDriverNoLabels() *storagev1.CSIDriver {
	return &storagev1.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: defaultCSIDriverName,
		},
		Spec: storagev1.CSIDriverSpec{
			VolumeLifecycleModes: []storagev1.VolumeLifecycleMode{
				storagev1.VolumeLifecycleEphemeral,
			},
		},
	}
}

func testCSIDriverRestricted() *storagev1.CSIDriver {
	driver := testCSIDriverNoLabels()
	driver.Labels = map[string]string{
		csiInlineVolProfileLabel: "restricted",
	}
	return driver
}

func testCSIDriverBaseline() *storagev1.CSIDriver {
	driver := testCSIDriverNoLabels()
	driver.Labels = map[string]string{
		csiInlineVolProfileLabel: "baseline",
	}
	return driver
}

func testCSIDriverPrivileged() *storagev1.CSIDriver {
	driver := testCSIDriverNoLabels()
	driver.Labels = map[string]string{
		csiInlineVolProfileLabel: "privileged",
	}
	return driver
}

func testCSIDriverInvalid() *storagev1.CSIDriver {
	driver := testCSIDriverNoLabels()
	driver.Labels = map[string]string{
		csiInlineVolProfileLabel: "invalid-value",
	}
	return driver
}

func testPod() *coreapi.Pod {
	pod := &coreapi.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: defaultNamespaceName,
		},
		Spec: coreapi.PodSpec{
			InitContainers: []coreapi.Container{
				{
					Name: "initTest",
				},
			},
			Containers: []coreapi.Container{
				{
					Name: "test",
				},
			},
		},
	}

	return pod
}

func testPodWithInlineVol() *coreapi.Pod {
	pod := testPod()
	pod.Spec.Volumes = []coreapi.Volume{
		{
			Name: "test-vol",
			VolumeSource: coreapi.VolumeSource{
				CSI: &coreapi.CSIVolumeSource{
					Driver: defaultCSIDriverName,
				},
			},
		},
	}
	return pod
}

func testPodWithPVC() *coreapi.Pod {
	pod := testPod()
	pod.Spec.Volumes = []coreapi.Volume{
		{
			Name: "test-vol",
			VolumeSource: coreapi.VolumeSource{
				PersistentVolumeClaim: &coreapi.PersistentVolumeClaimVolumeSource{
					ClaimName: "test-pvc",
				},
			},
		},
	}
	return pod
}

func testPodWithInvalidDriverName() *coreapi.Pod {
	pod := testPod()
	pod.Spec.Volumes = []coreapi.Volume{
		{
			Name: "test-vol",
			VolumeSource: coreapi.VolumeSource{
				CSI: &coreapi.CSIVolumeSource{
					Driver: "invalid-csi-driver",
				},
			},
		},
	}
	return pod
}

// Creates a pod controller object, given an object type and a pod for the template
func createPodControllerObject(obj runtime.Object, pod *coreapi.Pod) runtime.Object {
	switch obj.(type) {
	case *coreapi.Pod:
		return pod
	case *coreapi.PodTemplate:
		return &coreapi.PodTemplate{
			ObjectMeta: metav1.ObjectMeta{Name: "test-pod-template"},
			Template: coreapi.PodTemplateSpec{
				ObjectMeta: pod.ObjectMeta,
				Spec:       pod.Spec,
			},
		}
	case *coreapi.ReplicationController:
		return &coreapi.ReplicationController{
			ObjectMeta: metav1.ObjectMeta{Name: "test-repl-controller"},
			Spec: coreapi.ReplicationControllerSpec{
				Template: &coreapi.PodTemplateSpec{
					ObjectMeta: pod.ObjectMeta,
					Spec:       pod.Spec,
				},
			},
		}
	case *appsapi.ReplicaSet:
		return &appsapi.ReplicaSet{
			ObjectMeta: metav1.ObjectMeta{Name: "test-replicaset"},
			Spec: appsapi.ReplicaSetSpec{
				Template: coreapi.PodTemplateSpec{
					ObjectMeta: pod.ObjectMeta,
					Spec:       pod.Spec,
				},
			},
		}
	case *appsapi.Deployment:
		return &appsapi.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: "test-deployment"},
			Spec: appsapi.DeploymentSpec{
				Template: coreapi.PodTemplateSpec{
					ObjectMeta: pod.ObjectMeta,
					Spec:       pod.Spec,
				},
			},
		}
	case *appsapi.DaemonSet:
		return &appsapi.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: "test-daemonset"},
			Spec: appsapi.DaemonSetSpec{
				Template: coreapi.PodTemplateSpec{
					ObjectMeta: pod.ObjectMeta,
					Spec:       pod.Spec,
				},
			},
		}
	case *appsapi.StatefulSet:
		return &appsapi.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{Name: "test-statefulset"},
			Spec: appsapi.StatefulSetSpec{
				Template: coreapi.PodTemplateSpec{
					ObjectMeta: pod.ObjectMeta,
					Spec:       pod.Spec,
				},
			},
		}
	case *batchapi.Job:
		return &batchapi.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "test-job"},
			Spec: batchapi.JobSpec{
				Template: coreapi.PodTemplateSpec{
					ObjectMeta: pod.ObjectMeta,
					Spec:       pod.Spec,
				},
			},
		}
	case *batchapi.CronJob:
		return &batchapi.CronJob{
			ObjectMeta: metav1.ObjectMeta{Name: "test-cronjob"},
			Spec: batchapi.CronJobSpec{
				JobTemplate: batchapi.JobTemplateSpec{
					Spec: batchapi.JobSpec{
						Template: coreapi.PodTemplateSpec{
							ObjectMeta: pod.ObjectMeta,
							Spec:       pod.Spec,
						},
					},
				},
			},
		}
	default:
		// If we can't add a pod template, just return the provided object.
		return obj
	}
}

func getObjectGroupVersionResource(obj runtime.Object) schema.GroupVersionResource {
	ver := "version"
	switch obj.(type) {
	case *coreapi.Pod:
		return coreapi.Resource("pods").WithVersion(ver)
	case *coreapi.PodTemplate:
		return coreapi.Resource("podtemplates").WithVersion(ver)
	case *coreapi.ReplicationController:
		return coreapi.Resource("replicationcontrollers").WithVersion(ver)
	case *appsapi.ReplicaSet:
		return appsapi.Resource("replicasets").WithVersion(ver)
	case *appsapi.Deployment:
		return appsapi.Resource("deployments").WithVersion(ver)
	case *appsapi.DaemonSet:
		return appsapi.Resource("daemonsets").WithVersion(ver)
	case *appsapi.StatefulSet:
		return appsapi.Resource("statefulsets").WithVersion(ver)
	case *batchapi.Job:
		return batchapi.Resource("jobs").WithVersion(ver)
	case *batchapi.CronJob:
		return batchapi.Resource("cronjobs").WithVersion(ver)
	default:
		// If it's not a recognized object, return something invalid.
		return coreapi.Resource("invalidresource").WithVersion("invalidversion")
	}
}
