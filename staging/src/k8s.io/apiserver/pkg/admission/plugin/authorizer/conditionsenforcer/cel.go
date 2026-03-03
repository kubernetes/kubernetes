package conditionsenforcer

import (
	"context"

	celgo "github.com/google/cel-go/cel"
	celtypes "github.com/google/cel-go/common/types"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/admission"
	plugincel "k8s.io/apiserver/pkg/admission/plugin/cel"
	namespacematcher "k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/namespace"
	apiscel "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
)

const (
	ConditionTypeAuthorizationCEL authorizer.ConditionType = "k8s.io/authorization-cel"
)

func NewCELBuiltinConditionSetEvaluator(authorizer authorizer.Authorizer, informerFactory informers.SharedInformerFactory, client kubernetes.Interface) authorizer.BuiltinConditionSetEvaluator {
	compositionEnvTemplateWithStrictCost := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion())

	// TODO(luxas): Right now allowing the nsmatcher to be nil, as we don't have the informerFactory and client in withAuthorization
	var nsmatcher *namespacematcher.Matcher
	if informerFactory != nil && client != nil {
		nsmatcher = &namespacematcher.Matcher{
			NamespaceLister: informerFactory.Core().V1().Namespaces().Lister(),
			Client:          client,
		}
	}

	compiler := plugincel.NewConditionCompiler(compositionEnvTemplateWithStrictCost)
	return &celConditionsEnforcer{
		compiler: compiler,
		// TODO: Add unit test for using the authorizer
		optionalVarsDecls: plugincel.OptionalVariableDeclarations{HasParams: false, HasAuthorizer: authorizer != nil},
		optionalVars:      plugincel.OptionalVariableBindings{VersionedParams: nil, Authorizer: authorizer},
		nsmatcher:         nsmatcher,
	}
}

var _ authorizer.BuiltinConditionSetEvaluator = &celConditionsEnforcer{}

type celConditionsEnforcer struct {
	compiler          plugincel.ConditionCompiler
	optionalVarsDecls plugincel.OptionalVariableDeclarations
	optionalVars      plugincel.OptionalVariableBindings
	nsmatcher         *namespacematcher.Matcher
}

func (e *celConditionsEnforcer) BuiltinEvaluateConditions(ctx context.Context, conditionSet *authorizer.ConditionSet, data authorizer.ConditionData) (*authorizer.Decision, error) {
	wr := data.WriteRequest()
	if wr == nil {
		// Can only evaluate write requests for now, skip everything else
		return nil, nil
	}
	return e.evaluateWriteRequest(ctx, conditionSet, wr, apiscel.RuntimeCELCostBudget)
}

// runtimeCELCostBudget was added for testing purpose only. Callers should always use const RuntimeCELCostBudget from k8s.io/apiserver/pkg/apis/cel/config.go as input.
func (e *celConditionsEnforcer) evaluateWriteRequest(ctx context.Context, conditionSet *authorizer.ConditionSet, wr authorizer.WriteRequestConditionData, runtimeCELCostBudget int64) (*authorizer.Decision, error) {

	if conditionSet.Type() != ConditionTypeAuthorizationCEL {
		// Cannot handle this condition type, skip
		return nil, nil
	}

	attrsShim, ok := wr.(*attrsShim)
	if !ok {
		// TODO(luxas): Make more generic when we have a decision on how much data to include here.
		return nil, nil
	}

	admissionRequest := plugincel.CreateAdmissionRequest(attrsShim.Attributes, metav1.GroupVersionResource(attrsShim.GetResource()), metav1.GroupVersionKind(attrsShim.VersionedKind))
	// Decide which fields are exposed
	ns, err := e.getNamespace(ctx, attrsShim.Attributes)
	if err != nil {
		// Critical error, defer to main authorizer
		return nil, err
	}

	evaluatedDecision, warnings, err := authorizer.EvaluateConditionSet(conditionSet, ConditionTypeAuthorizationCEL, func(condStr string) (bool, error) {
		// TODO: Compile all deny conditions at a time, then all no opinion, etc?
		evaluator := e.compiler.CompileCondition([]plugincel.ExpressionAccessor{validationCondition(condStr)}, e.optionalVarsDecls, environment.StoredExpressions)

		// TODO(luxas): Do we trust the authorizer to give reasonable-length conditions?
		// Can a webhook authorizer today block a request for e.g. 60s?
		// TODO(luxas): Optimize this codepath further.
		evalResults, _, err := evaluator.ForInput(ctx, attrsShim.VersionedAttributes, admissionRequest, e.optionalVars, ns, runtimeCELCostBudget)
		if err != nil {
			return false, err
		}
		evalResult := &evalResults[0]
		if evalResult.Error != nil {
			return false, evalResult.Error
		}
		return evalResult.EvalResult == celtypes.True, nil
	})
	// If we got a "critical" evaluation error, skip builtin evaluation completely, defer to the authorizer that authored the condition to evaluate.
	// This could happen, for example, if the authorizer has a newer CEL environment than that is built-in, or due to similar version skew.
	if err != nil {
		return nil, err
	}
	// If we managed to evaluate without any critical error, return the decision we got, along with any warnings.
	return &evaluatedDecision, utilerrors.NewAggregate(warnings)
}

// getNamespace exactly mirrors the similar logic in VAP
func (e *celConditionsEnforcer) getNamespace(ctx context.Context, a admission.Attributes) (*v1.Namespace, error) {
	// TODO(luxas): Wire things up so that we always have this available.
	if e.nsmatcher == nil {
		return nil, nil
	}

	var namespace *v1.Namespace
	var err error
	namespaceName := a.GetNamespace()

	// Special case, the namespace object has the namespace of itself (maybe a bug).
	// unset it if the incoming object is a namespace
	if gvk := a.GetKind(); gvk.Kind == "Namespace" && gvk.Version == "v1" && gvk.Group == "" {
		namespaceName = ""
	}

	// if it is cluster scoped, namespaceName will be empty
	// Otherwise, get the Namespace resource.
	if namespaceName != "" {
		namespace, err = e.nsmatcher.GetNamespace(ctx, namespaceName)
		if err != nil {
			return nil, err
		}
	}

	return plugincel.CreateNamespaceObject(namespace), nil
}

type validationCondition string

func (v validationCondition) GetExpression() string {
	return string(v)
}

func (v validationCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.BoolType}
}
