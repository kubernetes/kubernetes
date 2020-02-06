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
	"time"

	fuzz "github.com/google/gofuzz"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/apis/core"
)

// Funcs returns the fuzzer functions for the core group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(q *resource.Quantity, c fuzz.Continue) {
			*q = *resource.NewQuantity(c.Int63n(1000), resource.DecimalExponent)
		},
		func(j *core.ObjectReference, c fuzz.Continue) {
			// We have to customize the randomization of TypeMetas because their
			// APIVersion and Kind must remain blank in memory.
			j.APIVersion = c.RandString()
			j.Kind = c.RandString()
			j.Namespace = c.RandString()
			j.Name = c.RandString()
			j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
			j.FieldPath = c.RandString()
		},
		func(j *core.PodExecOptions, c fuzz.Continue) {
			j.Stdout = true
			j.Stderr = true
		},
		func(j *core.PodAttachOptions, c fuzz.Continue) {
			j.Stdout = true
			j.Stderr = true
		},
		func(j *core.PodPortForwardOptions, c fuzz.Continue) {
			if c.RandBool() {
				j.Ports = make([]int32, c.Intn(10))
				for i := range j.Ports {
					j.Ports[i] = c.Int31n(65535)
				}
			}
		},
		func(s *core.PodSpec, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			// has a default value
			ttl := int64(30)
			if c.RandBool() {
				ttl = int64(c.Uint32())
			}
			s.TerminationGracePeriodSeconds = &ttl

			c.Fuzz(s.SecurityContext)

			if s.SecurityContext == nil {
				s.SecurityContext = new(core.PodSecurityContext)
			}
			if s.Affinity == nil {
				s.Affinity = new(core.Affinity)
			}
			if s.SchedulerName == "" {
				s.SchedulerName = core.DefaultSchedulerName
			}
			if s.EnableServiceLinks == nil {
				enableServiceLinks := corev1.DefaultEnableServiceLinks
				s.EnableServiceLinks = &enableServiceLinks
			}
		},
		func(j *core.PodPhase, c fuzz.Continue) {
			statuses := []core.PodPhase{core.PodPending, core.PodRunning, core.PodFailed, core.PodUnknown}
			*j = statuses[c.Rand.Intn(len(statuses))]
		},
		func(j *core.Binding, c fuzz.Continue) {
			c.Fuzz(&j.ObjectMeta)
			j.Target.Name = c.RandString()
		},
		func(j *core.ReplicationController, c fuzz.Continue) {
			c.FuzzNoCustom(j)

			// match defaulting
			if j.Spec.Template != nil {
				if len(j.Labels) == 0 {
					j.Labels = j.Spec.Template.Labels
				}
				if len(j.Spec.Selector) == 0 {
					j.Spec.Selector = j.Spec.Template.Labels
				}
			}
		},
		func(j *core.ReplicationControllerSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			//j.TemplateRef = nil // this is required for round trip
		},
		func(j *core.List, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// TODO: uncomment when round trip starts from a versioned object
			if false { //j.Items == nil {
				j.Items = []runtime.Object{}
			}
		},
		func(q *core.ResourceRequirements, c fuzz.Continue) {
			randomQuantity := func() resource.Quantity {
				var q resource.Quantity
				c.Fuzz(&q)
				// precalc the string for benchmarking purposes
				_ = q.String()
				return q
			}
			q.Limits = make(core.ResourceList)
			q.Requests = make(core.ResourceList)
			cpuLimit := randomQuantity()
			q.Limits[core.ResourceCPU] = cpuLimit.DeepCopy()
			q.Requests[core.ResourceCPU] = cpuLimit.DeepCopy()
			memoryLimit := randomQuantity()
			q.Limits[core.ResourceMemory] = memoryLimit.DeepCopy()
			q.Requests[core.ResourceMemory] = memoryLimit.DeepCopy()
			storageLimit := randomQuantity()
			q.Limits[core.ResourceStorage] = storageLimit.DeepCopy()
			q.Requests[core.ResourceStorage] = storageLimit.DeepCopy()
		},
		func(q *core.LimitRangeItem, c fuzz.Continue) {
			var cpuLimit resource.Quantity
			c.Fuzz(&cpuLimit)

			q.Type = core.LimitTypeContainer
			q.Default = make(core.ResourceList)
			q.Default[core.ResourceCPU] = cpuLimit.DeepCopy()

			q.DefaultRequest = make(core.ResourceList)
			q.DefaultRequest[core.ResourceCPU] = cpuLimit.DeepCopy()

			q.Max = make(core.ResourceList)
			q.Max[core.ResourceCPU] = cpuLimit.DeepCopy()

			q.Min = make(core.ResourceList)
			q.Min[core.ResourceCPU] = cpuLimit.DeepCopy()

			q.MaxLimitRequestRatio = make(core.ResourceList)
			q.MaxLimitRequestRatio[core.ResourceCPU] = resource.MustParse("10")
		},
		func(p *core.PullPolicy, c fuzz.Continue) {
			policies := []core.PullPolicy{core.PullAlways, core.PullNever, core.PullIfNotPresent}
			*p = policies[c.Rand.Intn(len(policies))]
		},
		func(rp *core.RestartPolicy, c fuzz.Continue) {
			policies := []core.RestartPolicy{core.RestartPolicyAlways, core.RestartPolicyNever, core.RestartPolicyOnFailure}
			*rp = policies[c.Rand.Intn(len(policies))]
		},
		// core.DownwardAPIVolumeFile needs to have a specific func since FieldRef has to be
		// defaulted to a version otherwise roundtrip will fail
		func(m *core.DownwardAPIVolumeFile, c fuzz.Continue) {
			m.Path = c.RandString()
			versions := []string{"v1"}
			m.FieldRef = &core.ObjectFieldSelector{}
			m.FieldRef.APIVersion = versions[c.Rand.Intn(len(versions))]
			m.FieldRef.FieldPath = c.RandString()
			c.Fuzz(m.Mode)
			if m.Mode != nil {
				*m.Mode &= 0777
			}
		},
		func(s *core.SecretVolumeSource, c fuzz.Continue) {
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
		func(cm *core.ConfigMapVolumeSource, c fuzz.Continue) {
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
		func(d *core.DownwardAPIVolumeSource, c fuzz.Continue) {
			c.FuzzNoCustom(d) // fuzz self without calling this function again

			// DefaultMode should always be set, it has a default
			// value and it is expected to be between 0 and 0777
			var mode int32
			c.Fuzz(&mode)
			mode &= 0777
			d.DefaultMode = &mode
		},
		func(s *core.ProjectedVolumeSource, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again

			// DefaultMode should always be set, it has a default
			// value and it is expected to be between 0 and 0777
			var mode int32
			c.Fuzz(&mode)
			mode &= 0777
			s.DefaultMode = &mode
		},
		func(k *core.KeyToPath, c fuzz.Continue) {
			c.FuzzNoCustom(k) // fuzz self without calling this function again
			k.Key = c.RandString()
			k.Path = c.RandString()

			// Mode is not mandatory, but if it is set, it should be
			// a value between 0 and 0777
			if k.Mode != nil {
				*k.Mode &= 0777
			}
		},
		func(vs *core.VolumeSource, c fuzz.Continue) {
			// Exactly one of the fields must be set.
			v := reflect.ValueOf(vs).Elem()
			i := int(c.RandUint64() % uint64(v.NumField()))
			t := v.Field(i).Addr()
			for v.Field(i).IsNil() {
				c.Fuzz(t.Interface())
			}
		},
		func(i *core.ISCSIVolumeSource, c fuzz.Continue) {
			i.ISCSIInterface = c.RandString()
			if i.ISCSIInterface == "" {
				i.ISCSIInterface = "default"
			}
		},
		func(i *core.ISCSIPersistentVolumeSource, c fuzz.Continue) {
			i.ISCSIInterface = c.RandString()
			if i.ISCSIInterface == "" {
				i.ISCSIInterface = "default"
			}
		},
		func(d *core.DNSPolicy, c fuzz.Continue) {
			policies := []core.DNSPolicy{core.DNSClusterFirst, core.DNSDefault}
			*d = policies[c.Rand.Intn(len(policies))]
		},
		func(p *core.Protocol, c fuzz.Continue) {
			protocols := []core.Protocol{core.ProtocolTCP, core.ProtocolUDP, core.ProtocolSCTP}
			*p = protocols[c.Rand.Intn(len(protocols))]
		},
		func(p *core.ServiceAffinity, c fuzz.Continue) {
			types := []core.ServiceAffinity{core.ServiceAffinityClientIP, core.ServiceAffinityNone}
			*p = types[c.Rand.Intn(len(types))]
		},
		func(p *core.ServiceType, c fuzz.Continue) {
			types := []core.ServiceType{core.ServiceTypeClusterIP, core.ServiceTypeNodePort, core.ServiceTypeLoadBalancer}
			*p = types[c.Rand.Intn(len(types))]
		},
		func(p *core.IPFamily, c fuzz.Continue) {
			types := []core.IPFamily{core.IPv4Protocol, core.IPv6Protocol}
			selected := types[c.Rand.Intn(len(types))]
			*p = selected
		},
		func(p *core.ServiceExternalTrafficPolicyType, c fuzz.Continue) {
			types := []core.ServiceExternalTrafficPolicyType{core.ServiceExternalTrafficPolicyTypeCluster, core.ServiceExternalTrafficPolicyTypeLocal}
			*p = types[c.Rand.Intn(len(types))]
		},
		func(ct *core.Container, c fuzz.Continue) {
			c.FuzzNoCustom(ct)                                          // fuzz self without calling this function again
			ct.TerminationMessagePath = "/" + ct.TerminationMessagePath // Must be non-empty
			ct.TerminationMessagePolicy = "File"
		},
		func(p *core.Probe, c fuzz.Continue) {
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
		func(ev *core.EnvVar, c fuzz.Continue) {
			ev.Name = c.RandString()
			if c.RandBool() {
				ev.Value = c.RandString()
			} else {
				ev.ValueFrom = &core.EnvVarSource{}
				ev.ValueFrom.FieldRef = &core.ObjectFieldSelector{}

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
		func(ev *core.EnvFromSource, c fuzz.Continue) {
			if c.RandBool() {
				ev.Prefix = "p_"
			}
			if c.RandBool() {
				c.Fuzz(&ev.ConfigMapRef)
			} else {
				c.Fuzz(&ev.SecretRef)
			}
		},
		func(cm *core.ConfigMapEnvSource, c fuzz.Continue) {
			c.FuzzNoCustom(cm) // fuzz self without calling this function again
			if c.RandBool() {
				opt := c.RandBool()
				cm.Optional = &opt
			}
		},
		func(s *core.SecretEnvSource, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
		},
		func(sc *core.SecurityContext, c fuzz.Continue) {
			c.FuzzNoCustom(sc) // fuzz self without calling this function again
			if c.RandBool() {
				priv := c.RandBool()
				sc.Privileged = &priv
			}

			if c.RandBool() {
				sc.Capabilities = &core.Capabilities{
					Add:  make([]core.Capability, 0),
					Drop: make([]core.Capability, 0),
				}
				c.Fuzz(&sc.Capabilities.Add)
				c.Fuzz(&sc.Capabilities.Drop)
			}
		},
		func(s *core.Secret, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			s.Type = core.SecretTypeOpaque
		},
		func(r *core.RBDVolumeSource, c fuzz.Continue) {
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
		func(r *core.RBDPersistentVolumeSource, c fuzz.Continue) {
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
		func(obj *core.HostPathVolumeSource, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			types := []core.HostPathType{core.HostPathUnset, core.HostPathDirectoryOrCreate, core.HostPathDirectory,
				core.HostPathFileOrCreate, core.HostPathFile, core.HostPathSocket, core.HostPathCharDev, core.HostPathBlockDev}
			typeVol := types[c.Rand.Intn(len(types))]
			if obj.Type == nil {
				obj.Type = &typeVol
			}
		},
		func(pv *core.PersistentVolume, c fuzz.Continue) {
			c.FuzzNoCustom(pv) // fuzz self without calling this function again
			types := []core.PersistentVolumePhase{core.VolumeAvailable, core.VolumePending, core.VolumeBound, core.VolumeReleased, core.VolumeFailed}
			pv.Status.Phase = types[c.Rand.Intn(len(types))]
			pv.Status.Message = c.RandString()
			reclamationPolicies := []core.PersistentVolumeReclaimPolicy{core.PersistentVolumeReclaimRecycle, core.PersistentVolumeReclaimRetain}
			pv.Spec.PersistentVolumeReclaimPolicy = reclamationPolicies[c.Rand.Intn(len(reclamationPolicies))]
			volumeModes := []core.PersistentVolumeMode{core.PersistentVolumeFilesystem, core.PersistentVolumeBlock}
			pv.Spec.VolumeMode = &volumeModes[c.Rand.Intn(len(volumeModes))]
		},
		func(pvc *core.PersistentVolumeClaim, c fuzz.Continue) {
			c.FuzzNoCustom(pvc) // fuzz self without calling this function again
			types := []core.PersistentVolumeClaimPhase{core.ClaimBound, core.ClaimPending, core.ClaimLost}
			pvc.Status.Phase = types[c.Rand.Intn(len(types))]
			volumeModes := []core.PersistentVolumeMode{core.PersistentVolumeFilesystem, core.PersistentVolumeBlock}
			pvc.Spec.VolumeMode = &volumeModes[c.Rand.Intn(len(volumeModes))]
		},
		func(obj *core.AzureDiskVolumeSource, c fuzz.Continue) {
			if obj.CachingMode == nil {
				obj.CachingMode = new(core.AzureDataDiskCachingMode)
				*obj.CachingMode = core.AzureDataDiskCachingReadWrite
			}
			if obj.Kind == nil {
				obj.Kind = new(core.AzureDataDiskKind)
				*obj.Kind = core.AzureSharedBlobDisk
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
		func(sio *core.ScaleIOVolumeSource, c fuzz.Continue) {
			sio.StorageMode = c.RandString()
			if sio.StorageMode == "" {
				sio.StorageMode = "ThinProvisioned"
			}
			sio.FSType = c.RandString()
			if sio.FSType == "" {
				sio.FSType = "xfs"
			}
		},
		func(sio *core.ScaleIOPersistentVolumeSource, c fuzz.Continue) {
			sio.StorageMode = c.RandString()
			if sio.StorageMode == "" {
				sio.StorageMode = "ThinProvisioned"
			}
			sio.FSType = c.RandString()
			if sio.FSType == "" {
				sio.FSType = "xfs"
			}
		},
		func(s *core.NamespaceSpec, c fuzz.Continue) {
			s.Finalizers = []core.FinalizerName{core.FinalizerKubernetes}
		},
		func(s *core.NamespaceStatus, c fuzz.Continue) {
			s.Phase = core.NamespaceActive
		},
		func(http *core.HTTPGetAction, c fuzz.Continue) {
			c.FuzzNoCustom(http)            // fuzz self without calling this function again
			http.Path = "/" + http.Path     // can't be blank
			http.Scheme = "x" + http.Scheme // can't be blank
		},
		func(ss *core.ServiceSpec, c fuzz.Continue) {
			c.FuzzNoCustom(ss) // fuzz self without calling this function again
			if len(ss.Ports) == 0 {
				// There must be at least 1 port.
				ss.Ports = append(ss.Ports, core.ServicePort{})
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
			types := []core.ServiceAffinity{core.ServiceAffinityNone, core.ServiceAffinityClientIP}
			ss.SessionAffinity = types[c.Rand.Intn(len(types))]
			switch ss.SessionAffinity {
			case core.ServiceAffinityClientIP:
				timeoutSeconds := int32(c.Rand.Intn(int(core.MaxClientIPServiceAffinitySeconds)))
				ss.SessionAffinityConfig = &core.SessionAffinityConfig{
					ClientIP: &core.ClientIPConfig{
						TimeoutSeconds: &timeoutSeconds,
					},
				}
			case core.ServiceAffinityNone:
				ss.SessionAffinityConfig = nil
			}
		},
		func(s *core.NodeStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.Allocatable = s.Capacity
		},
		func(e *core.Event, c fuzz.Continue) {
			c.FuzzNoCustom(e)
			e.EventTime = metav1.MicroTime{Time: time.Unix(1, 1000)}
			if e.Series != nil {
				e.Series.LastObservedTime = metav1.MicroTime{Time: time.Unix(3, 3000)}
			}
		},
	}
}
