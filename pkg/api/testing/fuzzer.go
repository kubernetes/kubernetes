/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"reflect"
	"strconv"

	"github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/resource"
	apitesting "k8s.io/apimachinery/pkg/api/testing"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/intstr"
	kubeadmfuzzer "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/fuzzer"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// overrideGenericFuncs override some generic fuzzer funcs from k8s.io/apiserver in order to have more realistic
// values in a Kubernetes context.
func overrideGenericFuncs(t apitesting.TestingCommon, codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(j *runtime.Object, c fuzz.Continue) {
			// TODO: uncomment when round trip starts from a versioned object
			if true { //c.RandBool() {
				*j = &runtime.Unknown{
					// We do not set TypeMeta here because it is not carried through a round trip
					Raw:         []byte(`{"apiVersion":"unknown.group/unknown","kind":"Something","someKey":"someValue"}`),
					ContentType: runtime.ContentTypeJSON,
				}
			} else {
				types := []runtime.Object{&api.Pod{}, &api.ReplicationController{}}
				t := types[c.Rand.Intn(len(types))]
				c.Fuzz(t)
				*j = t
			}
		},
		func(r *runtime.RawExtension, c fuzz.Continue) {
			// Pick an arbitrary type and fuzz it
			types := []runtime.Object{&api.Pod{}, &extensions.Deployment{}, &api.Service{}}
			obj := types[c.Rand.Intn(len(types))]
			c.Fuzz(obj)

			var codec runtime.Codec
			switch obj.(type) {
			case *extensions.Deployment:
				codec = apitesting.TestCodec(codecs, extensionsv1beta1.SchemeGroupVersion)
			default:
				codec = apitesting.TestCodec(codecs, v1.SchemeGroupVersion)
			}

			// Convert the object to raw bytes
			bytes, err := runtime.Encode(codec, obj)
			if err != nil {
				t.Errorf("Failed to encode object: %v", err)
				return
			}

			// Set the bytes field on the RawExtension
			r.Raw = bytes
		},
	}
}

func coreFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(q *resource.Quantity, c fuzz.Continue) {
			*q = *resource.NewQuantity(c.Int63n(1000), resource.DecimalExponent)
		},
		func(j *api.ObjectReference, c fuzz.Continue) {
			// We have to customize the randomization of TypeMetas because their
			// APIVersion and Kind must remain blank in memory.
			j.APIVersion = c.RandString()
			j.Kind = c.RandString()
			j.Namespace = c.RandString()
			j.Name = c.RandString()
			j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
			j.FieldPath = c.RandString()
		},
		func(j *api.ListOptions, c fuzz.Continue) {
			label, _ := labels.Parse("a=b")
			j.LabelSelector = label
			field, _ := fields.ParseSelector("a=b")
			j.FieldSelector = field
		},
		func(j *api.PodExecOptions, c fuzz.Continue) {
			j.Stdout = true
			j.Stderr = true
		},
		func(j *api.PodAttachOptions, c fuzz.Continue) {
			j.Stdout = true
			j.Stderr = true
		},
		func(j *api.PodPortForwardOptions, c fuzz.Continue) {
			if c.RandBool() {
				j.Ports = make([]int32, c.Intn(10))
				for i := range j.Ports {
					j.Ports[i] = c.Int31n(65535)
				}
			}
		},
		func(s *api.PodSpec, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			// has a default value
			ttl := int64(30)
			if c.RandBool() {
				ttl = int64(c.Uint32())
			}
			s.TerminationGracePeriodSeconds = &ttl

			c.Fuzz(s.SecurityContext)

			if s.SecurityContext == nil {
				s.SecurityContext = new(api.PodSecurityContext)
			}
			if s.Affinity == nil {
				s.Affinity = new(api.Affinity)
			}
			if s.SchedulerName == "" {
				s.SchedulerName = api.DefaultSchedulerName
			}
		},
		func(j *api.PodPhase, c fuzz.Continue) {
			statuses := []api.PodPhase{api.PodPending, api.PodRunning, api.PodFailed, api.PodUnknown}
			*j = statuses[c.Rand.Intn(len(statuses))]
		},
		func(j *api.Binding, c fuzz.Continue) {
			c.Fuzz(&j.ObjectMeta)
			j.Target.Name = c.RandString()
		},
		func(j *api.ReplicationControllerSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			//j.TemplateRef = nil // this is required for round trip
		},
		func(j *api.List, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// TODO: uncomment when round trip starts from a versioned object
			if false { //j.Items == nil {
				j.Items = []runtime.Object{}
			}
		},
		func(q *api.ResourceRequirements, c fuzz.Continue) {
			randomQuantity := func() resource.Quantity {
				var q resource.Quantity
				c.Fuzz(&q)
				// precalc the string for benchmarking purposes
				_ = q.String()
				return q
			}
			q.Limits = make(api.ResourceList)
			q.Requests = make(api.ResourceList)
			cpuLimit := randomQuantity()
			q.Limits[api.ResourceCPU] = *cpuLimit.Copy()
			q.Requests[api.ResourceCPU] = *cpuLimit.Copy()
			memoryLimit := randomQuantity()
			q.Limits[api.ResourceMemory] = *memoryLimit.Copy()
			q.Requests[api.ResourceMemory] = *memoryLimit.Copy()
			storageLimit := randomQuantity()
			q.Limits[api.ResourceStorage] = *storageLimit.Copy()
			q.Requests[api.ResourceStorage] = *storageLimit.Copy()
		},
		func(q *api.LimitRangeItem, c fuzz.Continue) {
			var cpuLimit resource.Quantity
			c.Fuzz(&cpuLimit)

			q.Type = api.LimitTypeContainer
			q.Default = make(api.ResourceList)
			q.Default[api.ResourceCPU] = *(cpuLimit.Copy())

			q.DefaultRequest = make(api.ResourceList)
			q.DefaultRequest[api.ResourceCPU] = *(cpuLimit.Copy())

			q.Max = make(api.ResourceList)
			q.Max[api.ResourceCPU] = *(cpuLimit.Copy())

			q.Min = make(api.ResourceList)
			q.Min[api.ResourceCPU] = *(cpuLimit.Copy())

			q.MaxLimitRequestRatio = make(api.ResourceList)
			q.MaxLimitRequestRatio[api.ResourceCPU] = resource.MustParse("10")
		},
		func(p *api.PullPolicy, c fuzz.Continue) {
			policies := []api.PullPolicy{api.PullAlways, api.PullNever, api.PullIfNotPresent}
			*p = policies[c.Rand.Intn(len(policies))]
		},
		func(rp *api.RestartPolicy, c fuzz.Continue) {
			policies := []api.RestartPolicy{api.RestartPolicyAlways, api.RestartPolicyNever, api.RestartPolicyOnFailure}
			*rp = policies[c.Rand.Intn(len(policies))]
		},
		// api.DownwardAPIVolumeFile needs to have a specific func since FieldRef has to be
		// defaulted to a version otherwise roundtrip will fail
		func(m *api.DownwardAPIVolumeFile, c fuzz.Continue) {
			m.Path = c.RandString()
			versions := []string{"v1"}
			m.FieldRef = &api.ObjectFieldSelector{}
			m.FieldRef.APIVersion = versions[c.Rand.Intn(len(versions))]
			m.FieldRef.FieldPath = c.RandString()
			c.Fuzz(m.Mode)
			if m.Mode != nil {
				*m.Mode &= 0777
			}
		},
		func(s *api.SecretVolumeSource, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again

			if c.RandBool() {
				opt := c.RandBool()
				s.Optional = &opt
			}
			// DefaultMode should always be set, it has a default
			// value and it is expected to be between 0 and 0777
			var mode int32
			c.Fuzz(&mode)
			mode &= 0777
			s.DefaultMode = &mode
		},
		func(cm *api.ConfigMapVolumeSource, c fuzz.Continue) {
			c.FuzzNoCustom(cm) // fuzz self without calling this function again

			if c.RandBool() {
				opt := c.RandBool()
				cm.Optional = &opt
			}
			// DefaultMode should always be set, it has a default
			// value and it is expected to be between 0 and 0777
			var mode int32
			c.Fuzz(&mode)
			mode &= 0777
			cm.DefaultMode = &mode
		},
		func(d *api.DownwardAPIVolumeSource, c fuzz.Continue) {
			c.FuzzNoCustom(d) // fuzz self without calling this function again

			// DefaultMode should always be set, it has a default
			// value and it is expected to be between 0 and 0777
			var mode int32
			c.Fuzz(&mode)
			mode &= 0777
			d.DefaultMode = &mode
		},
		func(s *api.ProjectedVolumeSource, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again

			// DefaultMode should always be set, it has a default
			// value and it is expected to be between 0 and 0777
			var mode int32
			c.Fuzz(&mode)
			mode &= 0777
			s.DefaultMode = &mode
		},
		func(k *api.KeyToPath, c fuzz.Continue) {
			c.FuzzNoCustom(k) // fuzz self without calling this function again
			k.Key = c.RandString()
			k.Path = c.RandString()

			// Mode is not mandatory, but if it is set, it should be
			// a value between 0 and 0777
			if k.Mode != nil {
				*k.Mode &= 0777
			}
		},
		func(vs *api.VolumeSource, c fuzz.Continue) {
			// Exactly one of the fields must be set.
			v := reflect.ValueOf(vs).Elem()
			i := int(c.RandUint64() % uint64(v.NumField()))
			t := v.Field(i).Addr()
			for v.Field(i).IsNil() {
				c.Fuzz(t.Interface())
			}
		},
		func(i *api.ISCSIVolumeSource, c fuzz.Continue) {
			i.ISCSIInterface = c.RandString()
			if i.ISCSIInterface == "" {
				i.ISCSIInterface = "default"
			}
		},
		func(d *api.DNSPolicy, c fuzz.Continue) {
			policies := []api.DNSPolicy{api.DNSClusterFirst, api.DNSDefault}
			*d = policies[c.Rand.Intn(len(policies))]
		},
		func(p *api.Protocol, c fuzz.Continue) {
			protocols := []api.Protocol{api.ProtocolTCP, api.ProtocolUDP}
			*p = protocols[c.Rand.Intn(len(protocols))]
		},
		func(p *api.ServiceAffinity, c fuzz.Continue) {
			types := []api.ServiceAffinity{api.ServiceAffinityClientIP, api.ServiceAffinityNone}
			*p = types[c.Rand.Intn(len(types))]
		},
		func(p *api.ServiceType, c fuzz.Continue) {
			types := []api.ServiceType{api.ServiceTypeClusterIP, api.ServiceTypeNodePort, api.ServiceTypeLoadBalancer}
			*p = types[c.Rand.Intn(len(types))]
		},
		func(p *api.ServiceExternalTrafficPolicyType, c fuzz.Continue) {
			types := []api.ServiceExternalTrafficPolicyType{api.ServiceExternalTrafficPolicyTypeCluster, api.ServiceExternalTrafficPolicyTypeLocal}
			*p = types[c.Rand.Intn(len(types))]
		},
		func(ct *api.Container, c fuzz.Continue) {
			c.FuzzNoCustom(ct)                                          // fuzz self without calling this function again
			ct.TerminationMessagePath = "/" + ct.TerminationMessagePath // Must be non-empty
			ct.TerminationMessagePolicy = "File"
		},
		func(p *api.Probe, c fuzz.Continue) {
			c.FuzzNoCustom(p)
			// These fields have default values.
			intFieldsWithDefaults := [...]string{"TimeoutSeconds", "PeriodSeconds", "SuccessThreshold", "FailureThreshold"}
			v := reflect.ValueOf(p).Elem()
			for _, field := range intFieldsWithDefaults {
				f := v.FieldByName(field)
				if f.Int() == 0 {
					f.SetInt(1)
				}
			}
		},
		func(ev *api.EnvVar, c fuzz.Continue) {
			ev.Name = c.RandString()
			if c.RandBool() {
				ev.Value = c.RandString()
			} else {
				ev.ValueFrom = &api.EnvVarSource{}
				ev.ValueFrom.FieldRef = &api.ObjectFieldSelector{}

				var versions []schema.GroupVersion
				for _, testGroup := range testapi.Groups {
					versions = append(versions, *testGroup.GroupVersion())
				}

				ev.ValueFrom.FieldRef.APIVersion = versions[c.Rand.Intn(len(versions))].String()
				ev.ValueFrom.FieldRef.FieldPath = c.RandString()
			}
		},
		func(ev *api.EnvFromSource, c fuzz.Continue) {
			if c.RandBool() {
				ev.Prefix = "p_"
			}
			if c.RandBool() {
				c.Fuzz(&ev.ConfigMapRef)
			} else {
				c.Fuzz(&ev.SecretRef)
			}
		},
		func(cm *api.ConfigMapEnvSource, c fuzz.Continue) {
			c.FuzzNoCustom(cm) // fuzz self without calling this function again
			if c.RandBool() {
				opt := c.RandBool()
				cm.Optional = &opt
			}
		},
		func(s *api.SecretEnvSource, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
		},
		func(sc *api.SecurityContext, c fuzz.Continue) {
			c.FuzzNoCustom(sc) // fuzz self without calling this function again
			if c.RandBool() {
				priv := c.RandBool()
				sc.Privileged = &priv
			}

			if c.RandBool() {
				sc.Capabilities = &api.Capabilities{
					Add:  make([]api.Capability, 0),
					Drop: make([]api.Capability, 0),
				}
				c.Fuzz(&sc.Capabilities.Add)
				c.Fuzz(&sc.Capabilities.Drop)
			}
		},
		func(s *api.Secret, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			s.Type = api.SecretTypeOpaque
		},
		func(r *api.RBDVolumeSource, c fuzz.Continue) {
			r.RBDPool = c.RandString()
			if r.RBDPool == "" {
				r.RBDPool = "rbd"
			}
			r.RadosUser = c.RandString()
			if r.RadosUser == "" {
				r.RadosUser = "admin"
			}
			r.Keyring = c.RandString()
			if r.Keyring == "" {
				r.Keyring = "/etc/ceph/keyring"
			}
		},
		func(pv *api.PersistentVolume, c fuzz.Continue) {
			c.FuzzNoCustom(pv) // fuzz self without calling this function again
			types := []api.PersistentVolumePhase{api.VolumeAvailable, api.VolumePending, api.VolumeBound, api.VolumeReleased, api.VolumeFailed}
			pv.Status.Phase = types[c.Rand.Intn(len(types))]
			pv.Status.Message = c.RandString()
			reclamationPolicies := []api.PersistentVolumeReclaimPolicy{api.PersistentVolumeReclaimRecycle, api.PersistentVolumeReclaimRetain}
			pv.Spec.PersistentVolumeReclaimPolicy = reclamationPolicies[c.Rand.Intn(len(reclamationPolicies))]
		},
		func(pvc *api.PersistentVolumeClaim, c fuzz.Continue) {
			c.FuzzNoCustom(pvc) // fuzz self without calling this function again
			types := []api.PersistentVolumeClaimPhase{api.ClaimBound, api.ClaimPending, api.ClaimLost}
			pvc.Status.Phase = types[c.Rand.Intn(len(types))]
		},
		func(obj *api.AzureDiskVolumeSource, c fuzz.Continue) {
			if obj.CachingMode == nil {
				obj.CachingMode = new(api.AzureDataDiskCachingMode)
				*obj.CachingMode = api.AzureDataDiskCachingReadWrite
			}
			if obj.Kind == nil {
				obj.Kind = new(api.AzureDataDiskKind)
				*obj.Kind = api.AzureSharedBlobDisk
			}
			if obj.FSType == nil {
				obj.FSType = new(string)
				*obj.FSType = "ext4"
			}
			if obj.ReadOnly == nil {
				obj.ReadOnly = new(bool)
				*obj.ReadOnly = false
			}
		},
		func(sio *api.ScaleIOVolumeSource, c fuzz.Continue) {
			sio.ProtectionDomain = c.RandString()
			if sio.ProtectionDomain == "" {
				sio.ProtectionDomain = "default"
			}
			sio.StoragePool = c.RandString()
			if sio.StoragePool == "" {
				sio.StoragePool = "default"
			}
			sio.StorageMode = c.RandString()
			if sio.StorageMode == "" {
				sio.StorageMode = "ThinProvisioned"
			}
			sio.FSType = c.RandString()
			if sio.FSType == "" {
				sio.FSType = "xfs"
			}
		},
		func(s *api.NamespaceSpec, c fuzz.Continue) {
			s.Finalizers = []api.FinalizerName{api.FinalizerKubernetes}
		},
		func(s *api.NamespaceStatus, c fuzz.Continue) {
			s.Phase = api.NamespaceActive
		},
		func(http *api.HTTPGetAction, c fuzz.Continue) {
			c.FuzzNoCustom(http)            // fuzz self without calling this function again
			http.Path = "/" + http.Path     // can't be blank
			http.Scheme = "x" + http.Scheme // can't be blank
		},
		func(ss *api.ServiceSpec, c fuzz.Continue) {
			c.FuzzNoCustom(ss) // fuzz self without calling this function again
			if len(ss.Ports) == 0 {
				// There must be at least 1 port.
				ss.Ports = append(ss.Ports, api.ServicePort{})
				c.Fuzz(&ss.Ports[0])
			}
			for i := range ss.Ports {
				switch ss.Ports[i].TargetPort.Type {
				case intstr.Int:
					ss.Ports[i].TargetPort.IntVal = 1 + ss.Ports[i].TargetPort.IntVal%65535 // non-zero
				case intstr.String:
					ss.Ports[i].TargetPort.StrVal = "x" + ss.Ports[i].TargetPort.StrVal // non-empty
				}
			}
		},
		func(n *api.Node, c fuzz.Continue) {
			c.FuzzNoCustom(n)
			n.Spec.ExternalID = "external"
		},
		func(s *api.NodeStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.Allocatable = s.Capacity
		},
	}
}

func extensionFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(j *extensions.DeploymentSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			rhl := int32(c.Rand.Int31())
			pds := int32(c.Rand.Int31())
			j.RevisionHistoryLimit = &rhl
			j.ProgressDeadlineSeconds = &pds
		},
		func(j *extensions.DeploymentStrategy, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// Ensure that strategyType is one of valid values.
			strategyTypes := []extensions.DeploymentStrategyType{extensions.RecreateDeploymentStrategyType, extensions.RollingUpdateDeploymentStrategyType}
			j.Type = strategyTypes[c.Rand.Intn(len(strategyTypes))]
			if j.Type != extensions.RollingUpdateDeploymentStrategyType {
				j.RollingUpdate = nil
			} else {
				rollingUpdate := extensions.RollingUpdateDeployment{}
				if c.RandBool() {
					rollingUpdate.MaxUnavailable = intstr.FromInt(int(c.Rand.Int31()))
					rollingUpdate.MaxSurge = intstr.FromInt(int(c.Rand.Int31()))
				} else {
					rollingUpdate.MaxSurge = intstr.FromString(fmt.Sprintf("%d%%", c.Rand.Int31()))
				}
				j.RollingUpdate = &rollingUpdate
			}
		},
		func(psp *extensions.PodSecurityPolicySpec, c fuzz.Continue) {
			c.FuzzNoCustom(psp) // fuzz self without calling this function again
			runAsUserRules := []extensions.RunAsUserStrategy{extensions.RunAsUserStrategyMustRunAsNonRoot, extensions.RunAsUserStrategyMustRunAs, extensions.RunAsUserStrategyRunAsAny}
			psp.RunAsUser.Rule = runAsUserRules[c.Rand.Intn(len(runAsUserRules))]
			seLinuxRules := []extensions.SELinuxStrategy{extensions.SELinuxStrategyRunAsAny, extensions.SELinuxStrategyMustRunAs}
			psp.SELinux.Rule = seLinuxRules[c.Rand.Intn(len(seLinuxRules))]
		},
		func(s *extensions.Scale, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			// TODO: Implement a fuzzer to generate valid keys, values and operators for
			// selector requirements.
			if s.Status.Selector != nil {
				s.Status.Selector = &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"testlabelkey": "testlabelval",
					},
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "testkey",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"val1", "val2", "val3"},
						},
					},
				}
			}
		},
		func(j *extensions.DaemonSetSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			rhl := int32(c.Rand.Int31())
			j.RevisionHistoryLimit = &rhl
		},
		func(j *extensions.DaemonSetUpdateStrategy, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// Ensure that strategyType is one of valid values.
			strategyTypes := []extensions.DaemonSetUpdateStrategyType{extensions.RollingUpdateDaemonSetStrategyType, extensions.OnDeleteDaemonSetStrategyType}
			j.Type = strategyTypes[c.Rand.Intn(len(strategyTypes))]
			if j.Type != extensions.RollingUpdateDaemonSetStrategyType {
				j.RollingUpdate = nil
			} else {
				rollingUpdate := extensions.RollingUpdateDaemonSet{}
				if c.RandBool() {
					if c.RandBool() {
						rollingUpdate.MaxUnavailable = intstr.FromInt(1 + int(c.Rand.Int31()))
					} else {
						rollingUpdate.MaxUnavailable = intstr.FromString(fmt.Sprintf("%d%%", 1+c.Rand.Int31()))
					}
				}
				j.RollingUpdate = &rollingUpdate
			}
		},
	}
}

func batchFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(j *batch.JobSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			completions := int32(c.Rand.Int31())
			parallelism := int32(c.Rand.Int31())
			j.Completions = &completions
			j.Parallelism = &parallelism
			if c.Rand.Int31()%2 == 0 {
				j.ManualSelector = newBool(true)
			} else {
				j.ManualSelector = nil
			}
		},
		func(sj *batch.CronJobSpec, c fuzz.Continue) {
			c.FuzzNoCustom(sj)
			suspend := c.RandBool()
			sj.Suspend = &suspend
			sds := int64(c.RandUint64())
			sj.StartingDeadlineSeconds = &sds
			sj.Schedule = c.RandString()
			if hasSuccessLimit := c.RandBool(); hasSuccessLimit {
				successfulJobsHistoryLimit := int32(c.Rand.Int31())
				sj.SuccessfulJobsHistoryLimit = &successfulJobsHistoryLimit
			}
			if hasFailedLimit := c.RandBool(); hasFailedLimit {
				failedJobsHistoryLimit := int32(c.Rand.Int31())
				sj.FailedJobsHistoryLimit = &failedJobsHistoryLimit
			}
		},
		func(cp *batch.ConcurrencyPolicy, c fuzz.Continue) {
			policies := []batch.ConcurrencyPolicy{batch.AllowConcurrent, batch.ForbidConcurrent, batch.ReplaceConcurrent}
			*cp = policies[c.Rand.Intn(len(policies))]
		},
	}
}

func autoscalingFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(s *autoscaling.HorizontalPodAutoscalerSpec, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			minReplicas := int32(c.Rand.Int31())
			s.MinReplicas = &minReplicas

			randomQuantity := func() resource.Quantity {
				var q resource.Quantity
				c.Fuzz(&q)
				// precalc the string for benchmarking purposes
				_ = q.String()
				return q
			}

			targetUtilization := int32(c.RandUint64())
			s.Metrics = []autoscaling.MetricSpec{
				{
					Type: autoscaling.PodsMetricSourceType,
					Pods: &autoscaling.PodsMetricSource{
						MetricName:         c.RandString(),
						TargetAverageValue: randomQuantity(),
					},
				},
				{
					Type: autoscaling.ResourceMetricSourceType,
					Resource: &autoscaling.ResourceMetricSource{
						Name: api.ResourceCPU,
						TargetAverageUtilization: &targetUtilization,
					},
				},
			}
		},
		func(s *autoscaling.HorizontalPodAutoscalerStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			randomQuantity := func() resource.Quantity {
				var q resource.Quantity
				c.Fuzz(&q)
				// precalc the string for benchmarking purposes
				_ = q.String()
				return q
			}
			currentUtilization := int32(c.RandUint64())
			s.CurrentMetrics = []autoscaling.MetricStatus{
				{
					Type: autoscaling.PodsMetricSourceType,
					Pods: &autoscaling.PodsMetricStatus{
						MetricName:          c.RandString(),
						CurrentAverageValue: randomQuantity(),
					},
				},
				{
					Type: autoscaling.ResourceMetricSourceType,
					Resource: &autoscaling.ResourceMetricStatus{
						Name: api.ResourceCPU,
						CurrentAverageUtilization: &currentUtilization,
					},
				},
			}
		},
	}
}

func rbacFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(r *rbac.RoleRef, c fuzz.Continue) {
			c.FuzzNoCustom(r) // fuzz self without calling this function again

			// match defaulter
			if len(r.APIGroup) == 0 {
				r.APIGroup = rbac.GroupName
			}
		},
		func(r *rbac.Subject, c fuzz.Continue) {
			switch c.Int31n(3) {
			case 0:
				r.Kind = rbac.ServiceAccountKind
				r.APIGroup = ""
				c.FuzzNoCustom(&r.Name)
				c.FuzzNoCustom(&r.Namespace)
			case 1:
				r.Kind = rbac.UserKind
				r.APIGroup = rbac.GroupName
				c.FuzzNoCustom(&r.Name)
				// user "*" won't round trip because we convert it to the system:authenticated group. try again.
				for r.Name == "*" {
					c.FuzzNoCustom(&r.Name)
				}
			case 2:
				r.Kind = rbac.GroupKind
				r.APIGroup = rbac.GroupName
				c.FuzzNoCustom(&r.Name)
			}
		},
	}
}

func appsFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(s *apps.StatefulSet, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again

			// match defaulter
			if len(s.Spec.PodManagementPolicy) == 0 {
				s.Spec.PodManagementPolicy = apps.OrderedReadyPodManagement
			}
			if len(s.Spec.UpdateStrategy.Type) == 0 {
				s.Spec.UpdateStrategy.Type = apps.RollingUpdateStatefulSetStrategyType
			}
			if s.Spec.RevisionHistoryLimit == nil {
				s.Spec.RevisionHistoryLimit = new(int32)
				*s.Spec.RevisionHistoryLimit = 10
			}
		},
	}
}
func policyFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(s *policy.PodDisruptionBudgetStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			s.PodDisruptionsAllowed = int32(c.Rand.Intn(2))
		},
	}
}

func certificateFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(obj *certificates.CertificateSigningRequestSpec, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again
			obj.Usages = []certificates.KeyUsage{certificates.UsageKeyEncipherment}
		},
	}
}

func admissionregistrationFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(obj *admissionregistration.ExternalAdmissionHook, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again
			p := admissionregistration.FailurePolicyType("Fail")
			obj.FailurePolicy = &p
		},
		func(obj *admissionregistration.Initializer, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again
			p := admissionregistration.FailurePolicyType("Fail")
			obj.FailurePolicy = &p
		},
	}
}

func FuzzerFuncs(t apitesting.TestingCommon, codecs runtimeserializer.CodecFactory) []interface{} {
	return apitesting.MergeFuzzerFuncs(t,
		apitesting.GenericFuzzerFuncs(t, codecs),
		overrideGenericFuncs(t, codecs),
		coreFuncs(t),
		extensionFuncs(t),
		appsFuncs(t),
		batchFuncs(t),
		autoscalingFuncs(t),
		rbacFuncs(t),
		kubeadmfuzzer.KubeadmFuzzerFuncs(t),
		policyFuncs(t),
		certificateFuncs(t),
		admissionregistrationFuncs(t),
	)
}

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}
