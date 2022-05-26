A linter to enforce importing certain packages consistently.

## What is this for?

Ideally, go imports should avoid aliasing. Sometimes though, especially with
Kubernetes API code, it becomes unavoidable, because many packages are imported
as e.g. "[package]/v1alpha1" and you end up with lots of collisions if you use
"v1alpha1". 

This linter lets you enforce that whenever (for example)
"pkg/apis/serving/v1alpha1" is aliased, it is aliased as "servingv1alpha1".

## Usage

~~~~
importas \
  -alias knative.dev/serving/pkg/apis/autoscaling/v1alpha1:autoscalingv1alpha1 \
  -alias knative.dev/serving/pkg/apis/serving/v1:servingv1 \
  ./...
~~~~

### `-no-unaliased` option

By default, importas allows non-aliased imports, even when the package is specified by `-alias` flag.
With `-no-unaliased` option, importas does not allow this.

~~~~
importas -no-unaliased \
  -alias knative.dev/serving/pkg/apis/autoscaling/v1alpha1:autoscalingv1alpha1 \
  -alias knative.dev/serving/pkg/apis/serving/v1:servingv1 \
  ./...
~~~~

### `-no-extra-aliases` option

By default, importas allows aliases which are not specified by `-alias` flags.
With `-no-extra-aliases` option, importas does not allow any unspecified aliases.

~~~~
importas -no-extra-aliases \
  -alias knative.dev/serving/pkg/apis/autoscaling/v1alpha1:autoscalingv1alpha1 \
  -alias knative.dev/serving/pkg/apis/serving/v1:servingv1 \
  ./...
~~~~

### Use regular expression

You can specify the package path by regular expression, and alias by regular expression replacement syntax like following snippet.

~~~~
importas -alias 'knative.dev/serving/pkg/apis/(\w+)/(v[\w\d]+):$1$2'
~~~~

`$1` represents the text of the first submatch. See [detail](https://golang.org/pkg/regexp/#Regexp.Expand).

So it will enforce that

"knative.dev/serving/pkg/apis/autoscaling/v1alpha1" is aliased by "autoscalingv1alpha1", and
"knative.dev/serving/pkg/apis/serving/v1" is aliased by "servingv1"
