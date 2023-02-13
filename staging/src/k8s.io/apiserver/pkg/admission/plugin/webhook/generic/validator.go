package generic

import (
	"fmt"
	"github.com/google/cel-go/cel"
	celtypes "github.com/google/cel-go/common/types"
	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/admissionregistration/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/library"
	"k8s.io/klog/v2"
	"reflect"
	"sync"
	"time"
)

const (
	ObjectVarName    = "object"
	OldObjectVarName = "oldObject"
	ParamsVarName    = "params"
	RequestVarName   = "request"

	checkFrequency = 100
)

var (
	initEnvsOnce sync.Once
	initEnvs     *envs
	initEnvsErr  error
)

type envs struct {
	noParams   *cel.Env
	withParams *cel.Env
}

func getEnvs() (*envs, error) {
	initEnvsOnce.Do(func() {
		base, err := buildBaseEnv()
		if err != nil {
			initEnvsErr = err
			return
		}
		noParams, err := buildNoParamsEnv(base)
		if err != nil {
			initEnvsErr = err
			return
		}
		withParams, err := buildWithParamsEnv(noParams)
		if err != nil {
			initEnvsErr = err
			return
		}
		initEnvs = &envs{noParams: noParams, withParams: withParams}
	})
	return initEnvs, initEnvsErr
}

// This is a similar code as in k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/compilation.go
// If any changes are made here, consider to make the same changes there as well.
func buildBaseEnv() (*cel.Env, error) {
	var opts []cel.EnvOption
	opts = append(opts, cel.HomogeneousAggregateLiterals())
	// Validate function declarations once during base env initialization,
	// so they don't need to be evaluated each time a CEL rule is compiled.
	// This is a relatively expensive operation.
	opts = append(opts, cel.EagerlyValidateDeclarations(true), cel.DefaultUTCTimeZone(true))
	opts = append(opts, library.ExtensionLibs...)

	return cel.NewEnv(opts...)
}

func buildNoParamsEnv(baseEnv *cel.Env) (*cel.Env, error) {
	var propDecls []cel.EnvOption
	reg := apiservercel.NewRegistry(baseEnv)

	requestType := buildRequestType()
	rt, err := apiservercel.NewRuleTypes(requestType.TypeName(), requestType, reg)
	if err != nil {
		return nil, err
	}
	if rt == nil {
		return nil, nil
	}
	opts, err := rt.EnvOptions(baseEnv.TypeProvider())
	if err != nil {
		return nil, err
	}
	propDecls = append(propDecls, cel.Variable(ObjectVarName, cel.DynType))
	propDecls = append(propDecls, cel.Variable(OldObjectVarName, cel.DynType))
	propDecls = append(propDecls, cel.Variable(RequestVarName, requestType.CelType()))

	opts = append(opts, propDecls...)
	env, err := baseEnv.Extend(opts...)
	if err != nil {
		return nil, err
	}
	return env, nil
}

func buildWithParamsEnv(noParams *cel.Env) (*cel.Env, error) {
	return noParams.Extend(cel.Variable(ParamsVarName, cel.DynType))
}

// buildRequestType generates a DeclType for AdmissionRequest. This may be replaced with a utility that
// converts the native type definition to apiservercel.DeclType once such a utility becomes available.
// The 'uid' field is omitted since it is not needed for in-process admission review.
// The 'object' and 'oldObject' fields are omitted since they are exposed as root level CEL variables.
func buildRequestType() *apiservercel.DeclType {
	field := func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField {
		return apiservercel.NewDeclField(name, declType, required, nil, nil)
	}
	fields := func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField {
		result := make(map[string]*apiservercel.DeclField, len(fields))
		for _, f := range fields {
			result[f.Name] = f
		}
		return result
	}
	gvkType := apiservercel.NewObjectType("kubernetes.GroupVersionKind", fields(
		field("group", apiservercel.StringType, true),
		field("version", apiservercel.StringType, true),
		field("kind", apiservercel.StringType, true),
	))
	gvrType := apiservercel.NewObjectType("kubernetes.GroupVersionResource", fields(
		field("group", apiservercel.StringType, true),
		field("version", apiservercel.StringType, true),
		field("resource", apiservercel.StringType, true),
	))
	userInfoType := apiservercel.NewObjectType("kubernetes.UserInfo", fields(
		field("username", apiservercel.StringType, false),
		field("uid", apiservercel.StringType, false),
		field("groups", apiservercel.NewListType(apiservercel.StringType, -1), false),
		field("extra", apiservercel.NewMapType(apiservercel.StringType, apiservercel.NewListType(apiservercel.StringType, -1), -1), false),
	))
	return apiservercel.NewObjectType("kubernetes.AdmissionRequest", fields(
		field("kind", gvkType, true),
		field("resource", gvrType, true),
		field("subResource", apiservercel.StringType, false),
		field("requestKind", gvkType, true),
		field("requestResource", gvrType, true),
		field("requestSubResource", apiservercel.StringType, false),
		field("name", apiservercel.StringType, true),
		field("namespace", apiservercel.StringType, false),
		field("operation", apiservercel.StringType, true),
		field("userInfo", userInfoType, true),
		field("dryRun", apiservercel.BoolType, false),
		field("options", apiservercel.DynType, false),
	))
}

// CELValidatorCompiler implement the interface ValidatorCompiler.
type CELValidatorCompiler struct {
}

// CELValidator implements the Validator interface
type CELValidator struct {
	webhook            *v1.ValidatingWebhook
	compilationResults []CompilationResult
}

// CompilationResult represents a compiled ValidatingAdmissionPolicy validation expression.
type CompilationResult struct {
	Program cel.Program
	Error   *apiservercel.Error
}

// Compile compiles the cel expression defined in ValidatingAdmissionPolicy
func (c *CELValidatorCompiler) Compile(h webhook.WebhookAccessor) *CELValidator {
	//TODO: support validating and mutating and error handle
	webhook, _ := h.GetValidatingWebhook()
	klog.Infof("ivelichkovich webhook %s has %v CEL constraints", h.GetName(), len(webhook.MatchConditions))
	if len(webhook.MatchConditions) == 0 {
		return nil
	}
	compilationResults := make([]CompilationResult, len(webhook.MatchConditions))
	for i, celExpression := range webhook.MatchConditions {
		compilationResults[i] = CompileValidatingPolicyExpression(celExpression.Expression)
	}
	return &CELValidator{webhook: webhook, compilationResults: compilationResults}
}

// CompileValidatingPolicyExpression returns a compiled webhook CEL expression.
func CompileValidatingPolicyExpression(validationExpression string) CompilationResult {
	var env *cel.Env
	envs, err := getEnvs()
	if err != nil {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInternal,
				Detail: "compiler initialization failed: " + err.Error(),
			},
		}
	}
	//if hasParams {
	//	env = envs.withParams
	//} else {
	env = envs.noParams
	//}

	ast, issues := env.Compile(validationExpression)
	if issues != nil {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInvalid,
				Detail: "compilation failed: " + issues.String(),
			},
		}
	}
	if ast.OutputType() != cel.BoolType {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInvalid,
				Detail: "cel expression must evaluate to a bool",
			},
		}
	}

	_, err = cel.AstToCheckedExpr(ast)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInternal,
				Detail: "unexpected compilation error: " + err.Error(),
			},
		}
	}
	prog, err := env.Program(ast,
		cel.EvalOptions(cel.OptOptimize),
		cel.OptimizeRegex(library.ExtensionLibRegexOptimizations...),
		cel.InterruptCheckFrequency(checkFrequency),
	)
	if err != nil {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   apiservercel.ErrorTypeInvalid,
				Detail: "program instantiation failed: " + err.Error(),
			},
		}
	}
	return CompilationResult{
		Program: prog,
	}
}

type PolicyDecisionAction string

const (
	ActionAdmit PolicyDecisionAction = "admit"
	ActionDeny  PolicyDecisionAction = "deny"
)

type PolicyDecisionEvaluation string

const (
	EvalAdmit PolicyDecisionEvaluation = "admit"
	EvalError PolicyDecisionEvaluation = "error"
	EvalDeny  PolicyDecisionEvaluation = "deny"
)

type PolicyDecision struct {
	Action     PolicyDecisionAction
	Evaluation PolicyDecisionEvaluation
	Message    string
	Reason     metav1.StatusReason
	Elapsed    time.Duration
}

type validationActivation struct {
	object, oldObject, params, request interface{}
}

// Validate validates all cel expressions in Validator and returns a PolicyDecision for each CEL expression or returns an error.
// An error will be returned if failed to convert the object/oldObject/params/request to unstructured.
// Each PolicyDecision will have a decision and a message.
// policyDecision.message will be empty if the decision is allowed and no error met.
func (v *CELValidator) Validate(versionedAttr VersionedAttributes) ([]PolicyDecision, error) {
	// TODO igorveli: use VersionedAttributes rather than attributes
	decisions := make([]PolicyDecision, len(v.compilationResults))
	var err error

	//klog.Infof("ivelichkovich old object: %v", versionedAttr.VersionedOldObject)
	oldObjectVal, err := objectToResolveVal(versionedAttr.VersionedOldObject)
	if err != nil {
		return nil, err
	}
	//klog.Infof("ivelichkovich new object: %v", versionedAttr.VersionedObject)
	objectVal, err := objectToResolveVal(versionedAttr.VersionedObject)
	if err != nil {
		return nil, err
	}
	//TODO igorveli: what are params here?
	//paramsVal, err := objectToResolveVal(versionedParams)
	//if err != nil {
	//	return nil, err
	//}

	request := createAdmissionRequest(versionedAttr.Attributes)
	requestVal, err := convertObjectToUnstructured(request)
	if err != nil {
		return nil, err
	}
	//TODO: what do params do?
	va := &validationActivation{
		object:    objectVal,
		oldObject: oldObjectVal,
		//params:    paramsVal,
		request: requestVal.Object,
	}

	//TODO: igorveli what to do for failure policy
	var f v1.FailurePolicyType
	if v.webhook.FailurePolicy == nil {
		f = v1.Fail
	} else {
		f = *v.webhook.FailurePolicy
	}

	for i, compilationResult := range v.compilationResults {
		//validation := v.webhook.MatchConditions[i]

		var policyDecision = &decisions[i]

		if compilationResult.Error != nil {
			klog.Infof("ivelichkovich compilation result is error %v", compilationResult.Error)
			policyDecision.Action = policyDecisionActionForError(f)
			policyDecision.Evaluation = EvalError
			policyDecision.Message = fmt.Sprintf("compilation error: %v", compilationResult.Error)
			continue
		}
		if compilationResult.Program == nil {
			klog.Infof("ivelichkovich compilation program is nil")
			policyDecision.Action = policyDecisionActionForError(f)
			policyDecision.Evaluation = EvalError
			policyDecision.Message = "unexpected internal error compiling expression"
			continue
		}
		t1 := time.Now()
		evalResult, _, err := compilationResult.Program.Eval(va)
		elapsed := time.Since(t1)
		policyDecision.Elapsed = elapsed
		if err != nil {
			klog.Infof("ivelichkovich successfully ran cel")
			policyDecision.Action = policyDecisionActionForError(f)
			policyDecision.Evaluation = EvalError
			policyDecision.Message = fmt.Sprintf("expression '%v' resulted in error: %v", v.webhook.MatchConditions[i].Expression, err)
		} else if evalResult != celtypes.True {
			klog.Infof("ivelichkovich action deny")
			policyDecision.Action = ActionDeny
		} else {
			klog.Infof("ivelichkovich action admit")
			policyDecision.Action = ActionAdmit
			policyDecision.Evaluation = EvalAdmit
		}
	}
	klog.Infof("ivelichkovich count of decisions %v", len(decisions))
	return decisions, nil
}

func policyDecisionActionForError(f v1.FailurePolicyType) PolicyDecisionAction {
	if f == v1.Ignore {
		return ActionAdmit
	}
	return ActionDeny
}

func createAdmissionRequest(attr admission.Attributes) *admissionv1.AdmissionRequest {
	// FIXME: how to get resource GVK, GVR and subresource?
	gvk := attr.GetKind()
	gvr := attr.GetResource()
	subresource := attr.GetSubresource()

	requestGVK := attr.GetKind()
	requestGVR := attr.GetResource()
	requestSubResource := attr.GetSubresource()

	aUserInfo := attr.GetUserInfo()
	var userInfo authenticationv1.UserInfo
	if aUserInfo != nil {
		userInfo = authenticationv1.UserInfo{
			Extra:    make(map[string]authenticationv1.ExtraValue),
			Groups:   aUserInfo.GetGroups(),
			UID:      aUserInfo.GetUID(),
			Username: aUserInfo.GetName(),
		}
		// Convert the extra information in the user object
		for key, val := range aUserInfo.GetExtra() {
			userInfo.Extra[key] = authenticationv1.ExtraValue(val)
		}
	}

	dryRun := attr.IsDryRun()

	return &admissionv1.AdmissionRequest{
		Kind: metav1.GroupVersionKind{
			Group:   gvk.Group,
			Kind:    gvk.Kind,
			Version: gvk.Version,
		},
		Resource: metav1.GroupVersionResource{
			Group:    gvr.Group,
			Resource: gvr.Resource,
			Version:  gvr.Version,
		},
		SubResource: subresource,
		RequestKind: &metav1.GroupVersionKind{
			Group:   requestGVK.Group,
			Kind:    requestGVK.Kind,
			Version: requestGVK.Version,
		},
		RequestResource: &metav1.GroupVersionResource{
			Group:    requestGVR.Group,
			Resource: requestGVR.Resource,
			Version:  requestGVR.Version,
		},
		RequestSubResource: requestSubResource,
		Name:               attr.GetName(),
		Namespace:          attr.GetNamespace(),
		Operation:          admissionv1.Operation(attr.GetOperation()),
		UserInfo:           userInfo,
		// Leave Object and OldObject unset since we don't provide access to them via request
		DryRun: &dryRun,
		Options: runtime.RawExtension{
			Object: attr.GetOperationOptions(),
		},
	}
}

func objectToResolveVal(r runtime.Object) (interface{}, error) {
	if r == nil || reflect.ValueOf(r).IsNil() {
		return nil, nil
	}
	v, err := convertObjectToUnstructured(r)
	if err != nil {
		return nil, err
	}
	return v.Object, nil
}

func convertObjectToUnstructured(obj interface{}) (*unstructured.Unstructured, error) {
	if obj == nil || reflect.ValueOf(obj).IsNil() {
		return &unstructured.Unstructured{Object: nil}, nil
	}
	ret, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	return &unstructured.Unstructured{Object: ret}, nil
}
