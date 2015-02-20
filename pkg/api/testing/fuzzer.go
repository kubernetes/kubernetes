/*
Copyright 2015 Google Inc. All rights reserved.

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
	"strconv"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/fsouza/go-dockerclient"
	"github.com/google/gofuzz"
	"speter.net/go/exp/math/dec/inf"
)

func fuzzOneOf(c fuzz.Continue, objs ...interface{}) {
	// Use a new fuzzer which cannot populate nil to ensure one obj will be set.
	// FIXME: would be nicer to use FuzzOnePtr() and reflect.
	f := fuzz.New().NilChance(0).NumElements(1, 1)
	i := c.RandUint64() % uint64(len(objs))
	f.Fuzz(objs[i])
}

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
		func(j *api.TypeMeta, c fuzz.Continue) {
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
			j.CreationTimestamp = util.Unix(sec, nsec).Rfc3339Copy()
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
		func(j *api.ListMeta, c fuzz.Continue) {
			j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
			j.SelfLink = c.RandString()
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
		func(j *api.ReplicationControllerSpec, c fuzz.Continue) {
			// TemplateRef is set to nil by omission; this is required for round trip
			c.Fuzz(&j.Template)
			c.Fuzz(&j.Selector)
			j.Replicas = int(c.RandUint64())
		},
		func(j *api.ReplicationControllerStatus, c fuzz.Continue) {
			// only replicas round trips
			j.Replicas = int(c.RandUint64())
		},
		func(j *api.List, c fuzz.Continue) {
			c.Fuzz(&j.ListMeta)
			c.Fuzz(&j.Items)
			if j.Items == nil {
				j.Items = []runtime.Object{}
			}
		},
		func(j *runtime.Object, c fuzz.Continue) {
			if c.RandBool() {
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
		func(intstr *util.IntOrString, c fuzz.Continue) {
			// util.IntOrString will panic if its kind is set wrong.
			if c.RandBool() {
				intstr.Kind = util.IntstrInt
				intstr.IntVal = int(c.RandUint64())
				intstr.StrVal = ""
			} else {
				intstr.Kind = util.IntstrString
				intstr.IntVal = 0
				intstr.StrVal = c.RandString()
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
		func(p *api.PullPolicy, c fuzz.Continue) {
			policies := []api.PullPolicy{api.PullAlways, api.PullNever, api.PullIfNotPresent}
			*p = policies[c.Rand.Intn(len(policies))]
		},
		func(rp *api.RestartPolicy, c fuzz.Continue) {
			// Exactly one of the fields should be set.
			fuzzOneOf(c, &rp.Always, &rp.OnFailure, &rp.Never)
		},
		func(vs *api.VolumeSource, c fuzz.Continue) {
			// Exactly one of the fields should be set.
			//FIXME: the fuzz can still end up nil.  What if fuzz allowed me to say that?
			fuzzOneOf(c, &vs.HostPath, &vs.EmptyDir, &vs.GCEPersistentDisk, &vs.GitRepo, &vs.Secret)
		},
		func(d *api.DNSPolicy, c fuzz.Continue) {
			policies := []api.DNSPolicy{api.DNSClusterFirst, api.DNSDefault}
			*d = policies[c.Rand.Intn(len(policies))]
		},
		func(p *api.Protocol, c fuzz.Continue) {
			protocols := []api.Protocol{api.ProtocolTCP, api.ProtocolUDP}
			*p = protocols[c.Rand.Intn(len(protocols))]
		},
		func(p *api.AffinityType, c fuzz.Continue) {
			types := []api.AffinityType{api.AffinityTypeClientIP, api.AffinityTypeNone}
			*p = types[c.Rand.Intn(len(types))]
		},
		func(ct *api.Container, c fuzz.Continue) {
			// This function exists soley to set TerminationMessagePath to a
			// non-empty string. TODO: consider making TerminationMessagePath a
			// new type to simplify fuzzing.
			ct.TerminationMessagePath = api.TerminationMessagePathDefault
			// Let fuzzer handle the rest of the fileds.
			c.Fuzz(&ct.Name)
			c.Fuzz(&ct.Image)
			c.Fuzz(&ct.Command)
			c.Fuzz(&ct.Ports)
			c.Fuzz(&ct.WorkingDir)
			c.Fuzz(&ct.Env)
			c.Fuzz(&ct.VolumeMounts)
			c.Fuzz(&ct.LivenessProbe)
			c.Fuzz(&ct.Lifecycle)
			c.Fuzz(&ct.ImagePullPolicy)
			c.Fuzz(&ct.Privileged)
			c.Fuzz(&ct.Capabilities)
		},
		func(e *api.Event, c fuzz.Continue) {
			// Fix event count to 1, otherwise, if a v1beta1 or v1beta2 event has a count set arbitrarily, it's count is ignored
			c.Fuzz(&e.TypeMeta)
			c.Fuzz(&e.ObjectMeta)
			c.Fuzz(&e.InvolvedObject)
			c.Fuzz(&e.Reason)
			c.Fuzz(&e.Message)
			c.Fuzz(&e.Source)
			c.Fuzz(&e.FirstTimestamp)
			c.Fuzz(&e.LastTimestamp)
			if e.FirstTimestamp.IsZero() {
				e.Count = 1
			} else {
				c.Fuzz(&e.Count)
			}
		},
		func(s *api.Secret, c fuzz.Continue) {
			c.Fuzz(&s.TypeMeta)
			c.Fuzz(&s.ObjectMeta)

			s.Type = api.SecretTypeOpaque
			c.Fuzz(&s.Data)
		},
		func(ep *api.Endpoint, c fuzz.Continue) {
			// TODO: If our API used a particular type for IP fields we could just catch that here.
			ep.IP = fmt.Sprintf("%d.%d.%d.%d", c.Rand.Intn(256), c.Rand.Intn(256), c.Rand.Intn(256), c.Rand.Intn(256))
			ep.Port = c.Rand.Intn(65536)
		},
	)
	return f
}
