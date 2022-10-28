package ginkgo

import (
	"github.com/onsi/ginkgo/v2/internal"
)

/*
Offset(uint) is a decorator that allows you to change the stack-frame offset used when computing the line number of the node in question.

You can learn more here: https://onsi.github.io/ginkgo/#the-offset-decorator
You can learn more about decorators here: https://onsi.github.io/ginkgo/#decorator-reference
*/
type Offset = internal.Offset

/*
FlakeAttempts(uint N) is a decorator that allows you to mark individual specs or spec containers as flaky. Ginkgo will run them up to `N` times until they pass.

You can learn more here: https://onsi.github.io/ginkgo/#the-flakeattempts-decorator
You can learn more about decorators here: https://onsi.github.io/ginkgo/#decorator-reference
*/
type FlakeAttempts = internal.FlakeAttempts

/*
MustPassRepeatedly(uint N) is a decorator that allows you to repeat the execution of individual specs or spec containers. Ginkgo will run them up to `N` times until they fail.

You can learn more here: https://onsi.github.io/ginkgo/#the-mustpassrepeatedly-decorator
You can learn more about decorators here: https://onsi.github.io/ginkgo/#decorator-reference
*/
type MustPassRepeatedly = internal.MustPassRepeatedly

/*
Focus is a decorator that allows you to mark a spec or container as focused.  Identical to FIt and FDescribe.

You can learn more here: https://onsi.github.io/ginkgo/#filtering-specs
You can learn more about decorators here: https://onsi.github.io/ginkgo/#decorator-reference
*/
const Focus = internal.Focus

/*
Pending is a decorator that allows you to mark a spec or container as pending.  Identical to PIt and PDescribe.

You can learn more here: https://onsi.github.io/ginkgo/#filtering-specs
You can learn more about decorators here: https://onsi.github.io/ginkgo/#decorator-reference
*/
const Pending = internal.Pending

/*
Serial is a decorator that allows you to mark a spec or container as serial.  These specs will never run in parallel with other specs.
Tests in ordered containers cannot be marked as serial - mark the ordered container instead.

You can learn more here: https://onsi.github.io/ginkgo/#serial-specs
You can learn more about decorators here: https://onsi.github.io/ginkgo/#decorator-reference
*/
const Serial = internal.Serial

/*
Ordered is a decorator that allows you to mark a container as ordered.  Tests in the container will always run in the order they appear.
They will never be randomized and they will never run in parallel with one another, though they may run in parallel with other specs.

You can learn more here: https://onsi.github.io/ginkgo/#ordered-containers
You can learn more about decorators here: https://onsi.github.io/ginkgo/#decorator-reference
*/
const Ordered = internal.Ordered

/*
OncePerOrdered is a decorator that allows you to mark outer BeforeEach, AfterEach, JustBeforeEach, and JustAfterEach setup nodes to run once
per ordered context.  Normally these setup nodes run around each individual spec, with OncePerOrdered they will run once around the set of specs in an ordered container.
The behavior for non-Ordered containers/specs is unchanged.

You can learn more here: https://onsi.github.io/ginkgo/#setup-around-ordered-containers-the-onceperordered-decorator
You can learn more about decorators here: https://onsi.github.io/ginkgo/#decorator-reference
*/
const OncePerOrdered = internal.OncePerOrdered

/*
Label decorates specs with Labels.  Multiple labels can be passed to Label and these can be arbitrary strings but must not include the following characters: "&|!,()/".
Labels can be applied to container and subject nodes, but not setup nodes.  You can provide multiple Labels to a given node and a spec's labels is the union of all labels in its node hierarchy.

You can learn more here: https://onsi.github.io/ginkgo/#spec-labels
You can learn more about decorators here: https://onsi.github.io/ginkgo/#decorator-reference
*/
func Label(labels ...string) Labels {
	return Labels(labels)
}

/*
Labels are the type for spec Label decorators.  Use Label(...) to construct Labels.
You can learn more here: https://onsi.github.io/ginkgo/#spec-labels
*/
type Labels = internal.Labels

/*
PollProgressAfter allows you to override the configured value for --poll-progress-after for a particular node.

Ginkgo will start emitting node progress if the node is still running after a duration of PollProgressAfter.  This allows you to get quicker feedback about the state of a long-running spec.
*/
type PollProgressAfter = internal.PollProgressAfter

/*
PollProgressInterval allows you to override the configured value for --poll-progress-interval for a particular node.

Once a node has been running for longer than PollProgressAfter Ginkgo will emit node progress periodically at an interval of PollProgresInterval.
*/
type PollProgressInterval = internal.PollProgressInterval

/*
NodeTimeout allows you to specify a timeout for an indivdiual node.  The node cannot be a container and must be interruptible (i.e. it must be passed a function that accepts a SpecContext or context.Context).

If the node does not exit within the specified NodeTimeout its context will be cancelled.  The node wil then have a period of time controlled by the GracePeriod decorator (or global --grace-period command-line argument) to exit.  If the node does not exit within GracePeriod Ginkgo will leak the node and proceed to any clean-up nodes associated with the current spec.
*/
type NodeTimeout = internal.NodeTimeout

/*
SpecTimeout allows you to specify a timeout for an indivdiual spec.  SpecTimeout can only decorate interruptible It nodes.

All nodes associated with the It node will need to complete before the SpecTimeout has elapsed.  Individual nodes (e.g. BeforeEach) may be decorated with different NodeTimeouts - but these can only serve to provide a more stringent deadline for the node in question; they cannot extend the deadline past the SpecTimeout.

If the spec does not complete within the specified SpecTimeout the currently running node will have its context cancelled.  The node wil then have a period of time controlled by that node's GracePeriod decorator (or global --grace-period command-line argument) to exit.  If the node does not exit within GracePeriod Ginkgo will leak the node and proceed to any clean-up nodes associated with the current spec.
*/
type SpecTimeout = internal.SpecTimeout

/*
GracePeriod denotes the period of time Ginkgo will wait for an interruptible node to exit once an interruption (whether due to a timeout or a user-invoked signal) has occurred.  If both the global --grace-period cli flag and a GracePeriod decorator are specified the value in the decorator will take precedence.

Nodes that do not finish within a GracePeriod will be leaked and Ginkgo will proceed to run subsequent nodes.  In the event of a timeout, such leaks will be reported to the user.
*/
type GracePeriod = internal.GracePeriod

/*
SuppressProgressReporting is a decorator that allows you to disable progress reporting of a particular node.  This is useful if `ginkgo -v -progress` is generating too much noise; particularly
if you have a `ReportAfterEach` node that is running for every skipped spec and is generating lots of progress reports.
*/
const SuppressProgressReporting = internal.SuppressProgressReporting
