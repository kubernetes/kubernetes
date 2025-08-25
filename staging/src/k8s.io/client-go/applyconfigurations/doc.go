/*
Copyright 2021 The Kubernetes Authors.

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

/*
Package applyconfigurations provides typesafe go representations of the apply
configurations that are used to constructs Server-side Apply requests.

# Basics

The Apply functions in the typed client (see the k8s.io/client-go/kubernetes/typed packages) offer
a direct and typesafe way of calling Server-side Apply. Each Apply function takes an "apply
configuration" type as an argument, which is a structured representation of an Apply request. For
example:

	import (
	     ...
	     v1ac "k8s.io/client-go/applyconfigurations/autoscaling/v1"
	)
	hpaApplyConfig := v1ac.HorizontalPodAutoscaler(autoscalerName, ns).
	     WithSpec(v1ac.HorizontalPodAutoscalerSpec().
	              WithMinReplicas(0)
	     )
	return hpav1client.Apply(ctx, hpaApplyConfig, metav1.ApplyOptions{FieldManager: "mycontroller", Force: true})

Note in this example that HorizontalPodAutoscaler is imported from an "applyconfigurations"
package. Each "apply configuration" type represents the same Kubernetes object kind as the
corresponding go struct, but where all fields are pointers to make them optional, allowing apply
requests to be accurately represented. For example, this when the apply configuration in the above
example is marshalled to YAML, it produces:

	apiVersion: autoscaling/v1
	kind: HorizontalPodAutoscaler
	metadata:
	    name: myHPA
	    namespace: myNamespace
	spec:
	    minReplicas: 0

To understand why this is needed, the above YAML cannot be produced by the
v1.HorizontalPodAutoscaler go struct. Take for example:

	hpa := v1.HorizontalPodAutoscaler{
	     TypeMeta: metav1.TypeMeta{
	              APIVersion: "autoscaling/v1",
	              Kind:       "HorizontalPodAutoscaler",
	     },
	     ObjectMeta: ObjectMeta{
	              Namespace: ns,
	              Name:      autoscalerName,
	     },
	     Spec: v1.HorizontalPodAutoscalerSpec{
	              MinReplicas: pointer.Int32Ptr(0),
	     },
	}

The above code attempts to declare the same apply configuration as shown in the previous examples,
but when marshalled to YAML, produces:

	kind: HorizontalPodAutoscaler
	apiVersion: autoscaling/v1
	metadata:
	  name: myHPA
	  namespace: myNamespace
	spec:
	  scaleTargetRef:
	    kind: ""
	    name: ""
	  minReplicas: 0
	  maxReplicas: 0

Which, among other things, contains spec.maxReplicas set to 0. This is almost certainly not what
the caller intended (the intended apply configuration says nothing about the maxReplicas field),
and could have serious consequences on a production system: it directs the autoscaler to downscale
to zero pods. The problem here originates from the fact that the go structs contain required fields
that are zero valued if not set explicitly. The go structs work as intended for create and update
operations, but are fundamentally incompatible with apply, which is why we have introduced the
generated "apply configuration" types.

The "apply configurations" also have convenience With<FieldName> functions that make it easier to
build apply requests. This allows developers to set fields without having to deal with the fact that
all the fields in the "apply configuration" types are pointers, and are inconvenient to set using
go. For example "MinReplicas: &0" is not legal go code, so without the With functions, developers
would work around this problem by using a library, .e.g. "MinReplicas: pointer.Int32Ptr(0)", but
string enumerations like corev1.Protocol are still a problem since they cannot be supported by a
general purpose library. In addition to the convenience, the With functions also isolate
developers from the underlying representation, which makes it safer for the underlying
representation to be changed to support additional features in the future.

# Controller Support

The new client-go support makes it much easier to use Server-side Apply in controllers, by either of
two mechanisms.

Mechanism 1:

When authoring new controllers to use Server-side Apply, a good approach is to have the controller
recreate the apply configuration for an object each time it reconciles that object.  This ensures
that the controller fully reconciles all the fields that it is responsible for. Controllers
typically should unconditionally set all the fields they own by setting "Force: true" in the
ApplyOptions. Controllers must also provide a FieldManager name that is unique to the
reconciliation loop that apply is called from.

When upgrading existing controllers to use Server-side Apply the same approach often works
well--migrate the controllers to recreate the apply configuration each time it reconciles any
object. For cases where this does not work well, see Mechanism 2.

Mechanism 2:

When upgrading existing controllers to use Server-side Apply, the controller might have multiple
code paths that update different parts of an object depending on various conditions. Migrating a
controller like this to Server-side Apply can be risky because if the controller forgets to include
any fields in an apply configuration that is included in a previous apply request, a field can be
accidentally deleted. For such cases, an alternative to mechanism 1 is to replace any controller
reconciliation code that performs a "read/modify-in-place/update" (or patch) workflow with a
"extract/modify-in-place/apply" workflow. Here's an example of the new workflow:

	    fieldMgr := "my-field-manager"
	    deploymentClient := clientset.AppsV1().Deployments("default")
	    // read, could also be read from a shared informer
	    deployment, err := deploymentClient.Get(ctx, "example-deployment", metav1.GetOptions{})
	    if err != nil {
	      // handle error
	    }
	    // extract
	    deploymentApplyConfig, err := appsv1ac.ExtractDeployment(deployment, fieldMgr)
	    if err != nil {
	      // handle error
	    }
	    // modify-in-place
	    deploymentApplyConfig.Spec.Template.Spec.WithContainers(corev1ac.Container().
		WithName("modify-slice").
		WithImage("nginx:1.14.2"),
	    )
	    // apply
	    applied, err := deploymentClient.Apply(ctx, extractedDeployment, metav1.ApplyOptions{FieldManager: fieldMgr})
*/
package applyconfigurations
