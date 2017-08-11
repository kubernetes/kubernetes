/*
Copyright 2017 The Kubernetes Authors.

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

package fuzzer

import (
	"reflect"
	"strconv"

	fuzz "github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api"
)

// Funcs returns the fuzzer functions for the core api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
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

				versions := []schema.GroupVersion{
					{Group: "admission.k8s.io", Version: "v1alpha1"},
					{Group: "apps", Version: "v1beta1"},
					{Group: "apps", Version: "v1beta2"},
					{Group: "foo", Version: "v42"},
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
