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
		func(j *api.ReplicationControllerSpec, c fuzz.Continue) {
			// TemplateRef must be nil for round trip
			c.Fuzz(&j.Template)
			if j.Template == nil {
				// TODO: v1beta1/2 can't round trip a nil template correctly, fix by having v1beta1/2
				// conversion compare converted object to nil via DeepEqual
				j.Template = &api.PodTemplateSpec{}
			}
			j.Template.ObjectMeta = api.ObjectMeta{Labels: j.Template.ObjectMeta.Labels}
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
	)
	return f
}
