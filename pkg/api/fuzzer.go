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

package api

import (
	"reflect"
	"strconv"

	"github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/resource"
	apitesting "k8s.io/apimachinery/pkg/api/testing"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func CoreFuzzerFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(q *resource.Quantity, c fuzz.Continue) {
			*q = *resource.NewQuantity(c.Int63n(1000), resource.DecimalExponent)
		},
		func(j *ObjectReference, c fuzz.Continue) {
			// We have to customize the randomization of TypeMetas because their
			// APIVersion and Kind must remain blank in memory.
			j.APIVersion = c.RandString()
			j.Kind = c.RandString()
			j.Namespace = c.RandString()
			j.Name = c.RandString()
			j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
			j.FieldPath = c.RandString()
		},
		func(j *ListOptions, c fuzz.Continue) {
			label, _ := labels.Parse("a=b")
			j.LabelSelector = label
			field, _ := fields.ParseSelector("a=b")
			j.FieldSelector = field
		},
		func(j *PodExecOptions, c fuzz.Continue) {
			j.Stdout = true
			j.Stderr = true
		},
		func(j *PodAttachOptions, c fuzz.Continue) {
			j.Stdout = true
			j.Stderr = true
		},
		func(j *PodPortForwardOptions, c fuzz.Continue) {
			if c.RandBool() {
				j.Ports = make([]int32, c.Intn(10))
				for i := range j.Ports {
					j.Ports[i] = c.Int31n(65535)
				}
			}
		},
		func(s *PodSpec, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			// has a default value
			ttl := int64(30)
			if c.RandBool() {
				ttl = int64(c.Uint32())
			}
			s.TerminationGracePeriodSeconds = &ttl

			c.Fuzz(s.SecurityContext)

			if s.SecurityContext == nil {
				s.SecurityContext = new(PodSecurityContext)
			}
			if s.Affinity == nil {
				s.Affinity = new(Affinity)
			}
			if s.SchedulerName == "" {
				s.SchedulerName = DefaultSchedulerName
			}
		},
		func(j *PodPhase, c fuzz.Continue) {
			statuses := []PodPhase{PodPending, PodRunning, PodFailed, PodUnknown}
			*j = statuses[c.Rand.Intn(len(statuses))]
		},
		func(j *Binding, c fuzz.Continue) {
			c.Fuzz(&j.ObjectMeta)
			j.Target.Name = c.RandString()
		},
		func(j *ReplicationControllerSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			//j.TemplateRef = nil // this is required for round trip
		},
		func(j *List, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// TODO: uncomment when round trip starts from a versioned object
			if false { //j.Items == nil {
				j.Items = []runtime.Object{}
			}
		},
		func(q *ResourceRequirements, c fuzz.Continue) {
			randomQuantity := func() resource.Quantity {
				var q resource.Quantity
				c.Fuzz(&q)
				// precalc the string for benchmarking purposes
				_ = q.String()
				return q
			}
			q.Limits = make(ResourceList)
			q.Requests = make(ResourceList)
			cpuLimit := randomQuantity()
			q.Limits[ResourceCPU] = *cpuLimit.Copy()
			q.Requests[ResourceCPU] = *cpuLimit.Copy()
			memoryLimit := randomQuantity()
			q.Limits[ResourceMemory] = *memoryLimit.Copy()
			q.Requests[ResourceMemory] = *memoryLimit.Copy()
			storageLimit := randomQuantity()
			q.Limits[ResourceStorage] = *storageLimit.Copy()
			q.Requests[ResourceStorage] = *storageLimit.Copy()
		},
		func(q *LimitRangeItem, c fuzz.Continue) {
			var cpuLimit resource.Quantity
			c.Fuzz(&cpuLimit)

			q.Type = LimitTypeContainer
			q.Default = make(ResourceList)
			q.Default[ResourceCPU] = *(cpuLimit.Copy())

			q.DefaultRequest = make(ResourceList)
			q.DefaultRequest[ResourceCPU] = *(cpuLimit.Copy())

			q.Max = make(ResourceList)
			q.Max[ResourceCPU] = *(cpuLimit.Copy())

			q.Min = make(ResourceList)
			q.Min[ResourceCPU] = *(cpuLimit.Copy())

			q.MaxLimitRequestRatio = make(ResourceList)
			q.MaxLimitRequestRatio[ResourceCPU] = resource.MustParse("10")
		},
		func(p *PullPolicy, c fuzz.Continue) {
			policies := []PullPolicy{PullAlways, PullNever, PullIfNotPresent}
			*p = policies[c.Rand.Intn(len(policies))]
		},
		func(rp *RestartPolicy, c fuzz.Continue) {
			policies := []RestartPolicy{RestartPolicyAlways, RestartPolicyNever, RestartPolicyOnFailure}
			*rp = policies[c.Rand.Intn(len(policies))]
		},
		// DownwardAPIVolumeFile needs to have a specific func since FieldRef has to be
		// defaulted to a version otherwise roundtrip will fail
		func(m *DownwardAPIVolumeFile, c fuzz.Continue) {
			m.Path = c.RandString()
			versions := []string{"v1"}
			m.FieldRef = &ObjectFieldSelector{}
			m.FieldRef.APIVersion = versions[c.Rand.Intn(len(versions))]
			m.FieldRef.FieldPath = c.RandString()
			c.Fuzz(m.Mode)
			if m.Mode != nil {
				*m.Mode &= 0777
			}
		},
		func(s *SecretVolumeSource, c fuzz.Continue) {
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
		func(cm *ConfigMapVolumeSource, c fuzz.Continue) {
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
		func(d *DownwardAPIVolumeSource, c fuzz.Continue) {
			c.FuzzNoCustom(d) // fuzz self without calling this function again

			// DefaultMode should always be set, it has a default
			// value and it is expected to be between 0 and 0777
			var mode int32
			c.Fuzz(&mode)
			mode &= 0777
			d.DefaultMode = &mode
		},
		func(k *KeyToPath, c fuzz.Continue) {
			c.FuzzNoCustom(k) // fuzz self without calling this function again
			k.Key = c.RandString()
			k.Path = c.RandString()

			// Mode is not mandatory, but if it is set, it should be
			// a value between 0 and 0777
			if k.Mode != nil {
				*k.Mode &= 0777
			}
		},
		func(vs *VolumeSource, c fuzz.Continue) {
			// Exactly one of the fields must be set.
			v := reflect.ValueOf(vs).Elem()
			i := int(c.RandUint64() % uint64(v.NumField()))
			t := v.Field(i).Addr()
			for v.Field(i).IsNil() {
				c.Fuzz(t.Interface())
			}
		},
		func(i *ISCSIVolumeSource, c fuzz.Continue) {
			i.ISCSIInterface = c.RandString()
			if i.ISCSIInterface == "" {
				i.ISCSIInterface = "default"
			}
		},
		func(d *DNSPolicy, c fuzz.Continue) {
			policies := []DNSPolicy{DNSClusterFirst, DNSDefault}
			*d = policies[c.Rand.Intn(len(policies))]
		},
		func(p *Protocol, c fuzz.Continue) {
			protocols := []Protocol{ProtocolTCP, ProtocolUDP}
			*p = protocols[c.Rand.Intn(len(protocols))]
		},
		func(p *ServiceAffinity, c fuzz.Continue) {
			types := []ServiceAffinity{ServiceAffinityClientIP, ServiceAffinityNone}
			*p = types[c.Rand.Intn(len(types))]
		},
		func(p *ServiceType, c fuzz.Continue) {
			types := []ServiceType{ServiceTypeClusterIP, ServiceTypeNodePort, ServiceTypeLoadBalancer}
			*p = types[c.Rand.Intn(len(types))]
		},
		func(ct *Container, c fuzz.Continue) {
			c.FuzzNoCustom(ct)                                          // fuzz self without calling this function again
			ct.TerminationMessagePath = "/" + ct.TerminationMessagePath // Must be non-empty
			ct.TerminationMessagePolicy = "File"
		},
		func(p *Probe, c fuzz.Continue) {
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
		func(ev *EnvVar, c fuzz.Continue) {
			ev.Name = c.RandString()
			if c.RandBool() {
				ev.Value = c.RandString()
			} else {
				c.FuzzNoCustom(ev) // fuzz self without calling this function again
			}
		},
		func(ev *EnvFromSource, c fuzz.Continue) {
			if c.RandBool() {
				ev.Prefix = "p_"
			}
			if c.RandBool() {
				c.Fuzz(&ev.ConfigMapRef)
			} else {
				c.Fuzz(&ev.SecretRef)
			}
		},
		func(cm *ConfigMapEnvSource, c fuzz.Continue) {
			c.FuzzNoCustom(cm) // fuzz self without calling this function again
			if c.RandBool() {
				opt := c.RandBool()
				cm.Optional = &opt
			}
		},
		func(s *SecretEnvSource, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
		},
		func(sc *SecurityContext, c fuzz.Continue) {
			c.FuzzNoCustom(sc) // fuzz self without calling this function again
			if c.RandBool() {
				priv := c.RandBool()
				sc.Privileged = &priv
			}

			if c.RandBool() {
				sc.Capabilities = &Capabilities{
					Add:  make([]Capability, 0),
					Drop: make([]Capability, 0),
				}
				c.Fuzz(&sc.Capabilities.Add)
				c.Fuzz(&sc.Capabilities.Drop)
			}
		},
		func(s *Secret, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			s.Type = SecretTypeOpaque
		},
		func(r *RBDVolumeSource, c fuzz.Continue) {
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
		func(pv *PersistentVolume, c fuzz.Continue) {
			c.FuzzNoCustom(pv) // fuzz self without calling this function again
			types := []PersistentVolumePhase{VolumeAvailable, VolumePending, VolumeBound, VolumeReleased, VolumeFailed}
			pv.Status.Phase = types[c.Rand.Intn(len(types))]
			pv.Status.Message = c.RandString()
			reclamationPolicies := []PersistentVolumeReclaimPolicy{PersistentVolumeReclaimRecycle, PersistentVolumeReclaimRetain}
			pv.Spec.PersistentVolumeReclaimPolicy = reclamationPolicies[c.Rand.Intn(len(reclamationPolicies))]
		},
		func(pvc *PersistentVolumeClaim, c fuzz.Continue) {
			c.FuzzNoCustom(pvc) // fuzz self without calling this function again
			types := []PersistentVolumeClaimPhase{ClaimBound, ClaimPending, ClaimLost}
			pvc.Status.Phase = types[c.Rand.Intn(len(types))]
		},
		func(obj *AzureDiskVolumeSource, c fuzz.Continue) {
			if obj.CachingMode == nil {
				obj.CachingMode = new(AzureDataDiskCachingMode)
				*obj.CachingMode = AzureDataDiskCachingNone
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
		func(s *NamespaceSpec, c fuzz.Continue) {
			s.Finalizers = []FinalizerName{FinalizerKubernetes}
		},
		func(s *NamespaceStatus, c fuzz.Continue) {
			s.Phase = NamespaceActive
		},
		func(http *HTTPGetAction, c fuzz.Continue) {
			c.FuzzNoCustom(http)            // fuzz self without calling this function again
			http.Path = "/" + http.Path     // can't be blank
			http.Scheme = "x" + http.Scheme // can't be blank
		},
		func(ss *ServiceSpec, c fuzz.Continue) {
			c.FuzzNoCustom(ss) // fuzz self without calling this function again
			if len(ss.Ports) == 0 {
				// There must be at least 1 port.
				ss.Ports = append(ss.Ports, ServicePort{})
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
		func(n *Node, c fuzz.Continue) {
			c.FuzzNoCustom(n)
			n.Spec.ExternalID = "external"
		},
		func(s *NodeStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.Allocatable = s.Capacity
		},
	}
}
