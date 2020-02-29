# Behaviors Defining Conformance

The conformance program is intended to answer the question "What is
Kubernetes?". That is, what features, functions, and APIs are needed in order to
call something "Kubernetes". Since v1.9, this has been defined as passing a
specific set of e2e tests.  As part of the [conformance behavior KEP](https://git.k8s.io/enhancements/keps/sig-architecture/20190412-conformance-behaviors.md),
this instead is moved to an explicit list of "behaviors", which are captured in
this directory tree. The e2e tests are used to validate whether specific
behaviors are met, but the definition of conformance is based upon that list of
approved behaviors. This allows separate reviewers for behaviors and tests,
provides a description of conformance separate from long, complex tests with
code, and enables the definition of conformance to encompass behaviors for which
tests have not yet been written.  All of this begs the question, though, "what is
a behavior?".

In behavior driven development, it is sometimes defined as the "business logic"
expected from the software. That is, it is a sequence of reactions to some
stimulus: an API call, a component failure, a network request on the data plane,
or some other action. We can classify these reactions into a few different types
in Kubernetes:

1. Transient runtime / communication state
1. Cluster state changes observable via the control plane
1. Cluster state changes observable via the data plane

Another way to think about this is that a behavior is the combination of the
question and answer to "What happens when...".

A behavior will consist of:

* A description of the initial conditions and stimulus
* A description of the resulting reactions
* If necessary, a sequencing (ordering) of those reactions

All this is still pretty vague, so it is helpful to enumerate some of the
characteristics expected of behavior descriptions for conformance in particular.

 - Behaviors should be defined at the user-visible level. Things happening
   behind the scenes that are not visible to the user via the API or actual data
   plane execution do not need to be part of conformance.
 - Behaviors should describe portable features. That is, they should be
   expected to work as described across vendor systems.
 - Behaviors should be defined so they are minimally constraining; if a detail is
   not needed for portability, it should be left out.
 - Behaviors should not be tied to specific implementations of components, or even
   to the existence of specific components. If a vendor chooses to rewrite a
   component or replace it with something else, they should still pass conformance
   simply by meeting the expected behaviors.
 - Ordered sequencing of the reactions should be avoided unless absolutely
   necessary. For example, it is core to the `initContainers` functionality that
   they run before the main containers, so in that case ordering is required.

<!--
 TODO Examples: include example of a good behavior and a poorly written behavior
-->
