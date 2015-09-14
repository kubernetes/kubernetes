/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"math/rand"
	"reflect"
	"strconv"
	"testing"

	docker "github.com/fsouza/go-dockerclient"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/registered"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"

	"github.com/google/gofuzz"
	"speter.net/go/exp/math/dec/inf"
)

// FuzzerFor can randomly populate api objects that are destined for version.
func FuzzerFor(t *testing.T, version string, src rand.Source) *fuzz.Fuzzer {
	f := fuzz.New().NilChance(.5).NumElements(1, 1)
	if src != nil {
		f.RandSource(src)
	}
	f.Funcs(
		func(j *runtime.PluginBase, c fuzz.Continue) {
			// Do nothing; this struct has only a Kind field and it must stay blank in memory.
		},
		func(j *runtime.TypeMeta, c fuzz.Continue) {
			// We have to customize the randomization of TypeMetas because their
			// APIVersion and Kind must remain blank in memory.
			j.APIVersion = ""
			j.Kind = ""
		},
		func(j *unversioned.TypeMeta, c fuzz.Continue) {
			// We have to customize the randomization of TypeMetas because their
			// APIVersion and Kind must remain blank in memory.
			j.APIVersion = ""
			j.Kind = ""
		},
		func(j *api.ObjectMeta, c fuzz.Continue) {
			j.Name = c.RandString()
			j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
			j.SelfLink = c.RandString()
			j.UID = types.UID(c.RandString())
			j.GenerateName = c.RandString()

			var sec, nsec int64
			c.Fuzz(&sec)
			c.Fuzz(&nsec)
			j.CreationTimestamp = unversioned.Unix(sec, nsec).Rfc3339Copy()
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
		func(j *unversioned.ListMeta, c fuzz.Continue) {
			j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
			j.SelfLink = c.RandString()
		},
		func(j *api.ListOptions, c fuzz.Continue) {
			// TODO: add some parsing
			j.LabelSelector, _ = labels.Parse("a=b")
			j.FieldSelector, _ = fields.ParseSelector("a=b")
		},
		func(s *api.PodSpec, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			// has a default value
			ttl := int64(30)
			if c.RandBool() {
				ttl = int64(c.Uint32())
			}
			s.TerminationGracePeriodSeconds = &ttl

			if s.SecurityContext == nil {
				s.SecurityContext = &api.PodSecurityContext{}
			}
		},
		func(j *api.PodPhase, c fuzz.Continue) {
			statuses := []api.PodPhase{api.PodPending, api.PodRunning, api.PodFailed, api.PodUnknown}
			*j = statuses[c.Rand.Intn(len(statuses))]
		},
		func(j *api.PodTemplateSpec, c fuzz.Continue) {
			// TODO: v1beta1/2 can't round trip a nil template correctly, fix by having v1beta1/2
			// conversion compare converted object to nil via DeepEqual
			j.ObjectMeta = api.ObjectMeta{}
			c.Fuzz(&j.ObjectMeta)
			j.ObjectMeta = api.ObjectMeta{Labels: j.ObjectMeta.Labels}
			j.Spec = api.PodSpec{}
			c.Fuzz(&j.Spec)
		},
		func(j *api.Binding, c fuzz.Continue) {
			c.Fuzz(&j.ObjectMeta)
			j.Target.Name = c.RandString()
		},
		func(j *api.ReplicationControllerSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			//j.TemplateRef = nil // this is required for round trip
		},
		func(j *experimental.DeploymentStrategy, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// Ensure that strategyType is one of valid values.
			strategyTypes := []experimental.DeploymentStrategyType{experimental.RecreateDeploymentStrategyType, experimental.RollingUpdateDeploymentStrategyType}
			j.Type = strategyTypes[c.Rand.Intn(len(strategyTypes))]
			if j.Type != experimental.RollingUpdateDeploymentStrategyType {
				j.RollingUpdate = nil
			} else {
				rollingUpdate := experimental.RollingUpdateDeployment{}
				if c.RandBool() {
					rollingUpdate.MaxUnavailable = util.NewIntOrStringFromInt(int(c.RandUint64()))
					rollingUpdate.MaxSurge = util.NewIntOrStringFromInt(int(c.RandUint64()))
				} else {
					rollingUpdate.MaxSurge = util.NewIntOrStringFromString(fmt.Sprintf("%d%%", c.RandUint64()))
				}
				j.RollingUpdate = &rollingUpdate
			}
		},
		func(j *experimental.JobSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			completions := c.Rand.Int()
			parallelism := c.Rand.Int()
			j.Completions = &completions
			j.Parallelism = &parallelism
		},
		func(j *api.List, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// TODO: uncomment when round trip starts from a versioned object
			if false { //j.Items == nil {
				j.Items = []runtime.Object{}
			}
		},
		func(j *runtime.Object, c fuzz.Continue) {
			// TODO: uncomment when round trip starts from a versioned object
			if true { //c.RandBool() {
				*j = &runtime.Unknown{
					TypeMeta: runtime.TypeMeta{Kind: "Something", APIVersion: "unknown"},
					RawJSON:  []byte(`{"apiVersion":"unknown","kind":"Something","someKey":"someValue"}`),
				}
			} else {
				types := []runtime.Object{&api.Pod{}, &api.ReplicationController{}}
				t := types[c.Rand.Intn(len(types))]
				c.Fuzz(t)
				*j = t
			}
		},
		func(pb map[docker.Port][]docker.PortBinding, c fuzz.Continue) {
			// This is necessary because keys with nil values get omitted.
			// TODO: Is this a bug?
			pb[docker.Port(c.RandString())] = []docker.PortBinding{
				{c.RandString(), c.RandString()},
				{c.RandString(), c.RandString()},
			}
		},
		func(pm map[string]docker.PortMapping, c fuzz.Continue) {
			// This is necessary because keys with nil values get omitted.
			// TODO: Is this a bug?
			pm[c.RandString()] = docker.PortMapping{
				c.RandString(): c.RandString(),
			}
		},
		func(q *resource.Quantity, c fuzz.Continue) {
			// Real Quantity fuzz testing is done elsewhere;
			// this limited subset of functionality survives
			// round-tripping to v1beta1/2.
			q.Amount = &inf.Dec{}
			q.Format = resource.DecimalExponent
			//q.Amount.SetScale(inf.Scale(-c.Intn(12)))
			q.Amount.SetUnscaled(c.Int63n(1000))
		},
		func(q *api.ResourceRequirements, c fuzz.Continue) {
			randomQuantity := func() resource.Quantity {
				return *resource.NewQuantity(c.Int63n(1000), resource.DecimalExponent)
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
			randomQuantity := func() resource.Quantity {
				return *resource.NewQuantity(c.Int63n(1000), resource.DecimalExponent)
			}
			cpuLimit := randomQuantity()

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
		func(vs *api.VolumeSource, c fuzz.Continue) {
			// Exactly one of the fields must be set.
			v := reflect.ValueOf(vs).Elem()
			i := int(c.RandUint64() % uint64(v.NumField()))
			v = v.Field(i).Addr()
			// Use a new fuzzer which cannot populate nil to ensure one field will be set.
			f := fuzz.New().NilChance(0).NumElements(1, 1)
			f.Funcs(
				// Only api.DownwardAPIVolumeFile needs to have a specific func since FieldRef has to be
				// defaulted to a version otherwise roundtrip will fail
				// For the remaining volume plugins the default fuzzer is enough.
				func(m *api.DownwardAPIVolumeFile, c fuzz.Continue) {
					m.Path = c.RandString()
					versions := []string{"v1"}
					m.FieldRef.APIVersion = versions[c.Rand.Intn(len(versions))]
					m.FieldRef.FieldPath = c.RandString()
				},
			).Fuzz(v.Interface())
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
		func(ct *api.Container, c fuzz.Continue) {
			c.FuzzNoCustom(ct)                                          // fuzz self without calling this function again
			ct.TerminationMessagePath = "/" + ct.TerminationMessagePath // Must be non-empty
		},
		func(ev *api.EnvVar, c fuzz.Continue) {
			ev.Name = c.RandString()
			if c.RandBool() {
				ev.Value = c.RandString()
			} else {
				ev.ValueFrom = &api.EnvVarSource{}
				ev.ValueFrom.FieldRef = &api.ObjectFieldSelector{}

				versions := registered.RegisteredVersions

				ev.ValueFrom.FieldRef.APIVersion = versions[c.Rand.Intn(len(versions))]
				ev.ValueFrom.FieldRef.FieldPath = c.RandString()
			}
		},
		func(sc *api.SecurityContext, c fuzz.Continue) {
			c.FuzzNoCustom(sc) // fuzz self without calling this function again
			priv := c.RandBool()
			sc.Privileged = &priv
			sc.Capabilities = &api.Capabilities{
				Add:  make([]api.Capability, 0),
				Drop: make([]api.Capability, 0),
			}
			c.Fuzz(&sc.Capabilities.Add)
			c.Fuzz(&sc.Capabilities.Drop)
		},
		func(e *api.Event, c fuzz.Continue) {
			c.FuzzNoCustom(e) // fuzz self without calling this function again
			// Fix event count to 1, otherwise, if a v1beta1 or v1beta2 event has a count set arbitrarily, it's count is ignored
			if e.FirstTimestamp.IsZero() {
				e.Count = 1
			} else {
				c.Fuzz(&e.Count)
			}
		},
		func(s *api.Secret, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			s.Type = api.SecretTypeOpaque
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
			types := []api.PersistentVolumeClaimPhase{api.ClaimBound, api.ClaimPending}
			pvc.Status.Phase = types[c.Rand.Intn(len(types))]
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
				switch ss.Ports[i].TargetPort.Kind {
				case util.IntstrInt:
					ss.Ports[i].TargetPort.IntVal = 1 + ss.Ports[i].TargetPort.IntVal%65535 // non-zero
				case util.IntstrString:
					ss.Ports[i].TargetPort.StrVal = "x" + ss.Ports[i].TargetPort.StrVal // non-empty
				}
			}
		},
		func(n *api.Node, c fuzz.Continue) {
			c.FuzzNoCustom(n)
			n.Spec.ExternalID = "external"
		},
		func(s *experimental.APIVersion, c fuzz.Continue) {
			// We can't use c.RandString() here because it may generate empty
			// string, which will cause tests failure.
			s.APIGroup = "something"
		},
	)
	return f
}
