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
FlakeAttempts(uint N) is a decorator that allows you to mark individual specs or spec containers as flaky.  Ginkgo will run them up to `N` times until they pass.

You can learn more here: https://onsi.github.io/ginkgo/#repeating-spec-runs-and-managing-flaky-specs
You can learn more about decorators here: https://onsi.github.io/ginkgo/#decorator-reference
*/
type FlakeAttempts = internal.FlakeAttempts

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

You can learh more here: https://onsi.github.io/ginkgo/#setup-around-ordered-containers-the-onceperordered-decorator
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
