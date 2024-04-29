package declarative_test

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp/cmpopts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	"k8s.io/kubernetes/pkg/apis/batch"
	installbatch "k8s.io/kubernetes/pkg/apis/batch/install"
	"k8s.io/kubernetes/pkg/apis/batch/validation"
	"k8s.io/kubernetes/pkg/apis/core"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/generated/openapi"
	"k8s.io/kubernetes/test/utils/apivalidation"
	"k8s.io/utils/pointer"
	"k8s.io/utils/ptr"
)

// maxParallelismForIndexJob is the maximum parallelism that an Indexed Job
// is allowed to have. This threshold allows to cap the length of
// .status.completedIndexes.
const maxParallelismForIndexedJob = 100000

// maxFailedIndexesForIndexedJob is the maximum number of failed indexes that
// an Indexed Job is allowed to have. This threshold allows to cap the length of
// .status.completedIndexes and .status.failedIndexes.
const maxFailedIndexesForIndexedJob = 100_000

const (
	completionsSoftLimit                    = 100_000
	parallelismLimitForHighCompletions      = 10_000
	maxFailedIndexesLimitForHighCompletions = 10_000

	// maximum number of rules in pod failure policy
	maxPodFailurePolicyRules = 20

	// maximum number of values for a OnExitCodes requirement in pod failure policy
	maxPodFailurePolicyOnExitCodesValues = 255

	// maximum number of patterns for a OnPodConditions requirement in pod failure policy
	maxPodFailurePolicyOnPodConditionsPatterns = 20

	// maximum length of the value of the managedBy field
	maxManagedByLength = 63

	// maximum length of succeededIndexes in JobSuccessPolicy.
	maxJobSuccessPolicySucceededIndexesLimit = 64 * 1024
	// maximum number of rules in successPolicy.
	maxSuccessPolicyRule = 20
)

var (
	batchScheme *runtime.Scheme = func() *runtime.Scheme {
		sch := runtime.NewScheme()
		_ = core.AddToScheme(sch)
		_ = corev1.AddToScheme(sch)
		installbatch.Install(sch)
		return sch
	}()
	batchDefs *resolver.DefinitionsSchemaResolver = resolver.NewDefinitionsSchemaResolver(openapi.GetOpenAPIDefinitions, batchScheme)
)

var (
	timeZoneEmpty      = ""
	timeZoneLocal      = "LOCAL"
	timeZoneUTC        = "UTC"
	timeZoneCorrect    = "Europe/Rome"
	timeZoneBadPrefix  = " Europe/Rome"
	timeZoneBadSuffix  = "Europe/Rome "
	timeZoneBadName    = "Europe/InvalidRome"
	timeZoneEmptySpace = " "
)

var ignoreErrValueDetail = cmpopts.IgnoreFields(field.Error{}, "BadValue", "Detail")

func getValidManualSelector() *metav1.LabelSelector {
	return &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
}

func getValidPodTemplateSpecForManual(selector *metav1.LabelSelector) api.PodTemplateSpec {
	return api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: selector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
}

func getValidGeneratedSelector() *metav1.LabelSelector {
	return &metav1.LabelSelector{
		MatchLabels: map[string]string{batch.ControllerUidLabel: "1a2b3c", batch.LegacyControllerUidLabel: "1a2b3c", batch.JobNameLabel: "myjob", batch.LegacyJobNameLabel: "myjob"},
	}
}

func getValidPodTemplateSpecForGenerated(selector *metav1.LabelSelector) api.PodTemplateSpec {
	return api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: selector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy:  api.RestartPolicyOnFailure,
			DNSPolicy:      api.DNSClusterFirst,
			Containers:     []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
			InitContainers: []api.Container{{Name: "def", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
}

func TestValidateJob(t *testing.T) {
	validJobObjectMeta := metav1.ObjectMeta{
		Name:      "myjob",
		Namespace: metav1.NamespaceDefault,
		UID:       types.UID("1a2b3c"),
	}
	validManualSelector := getValidManualSelector()
	failedPodReplacement := batch.Failed
	terminatingOrFailedPodReplacement := batch.TerminatingOrFailed
	validPodTemplateSpecForManual := getValidPodTemplateSpecForManual(validManualSelector)
	validGeneratedSelector := getValidGeneratedSelector()
	validPodTemplateSpecForGenerated := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	validPodTemplateSpecForGeneratedRestartPolicyNever := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	validPodTemplateSpecForGeneratedRestartPolicyNever.Spec.RestartPolicy = api.RestartPolicyNever
	validHostNetPodTemplateSpec := func() api.PodTemplateSpec {
		spec := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
		spec.Spec.SecurityContext = &api.PodSecurityContext{
			HostNetwork: true,
		}
		spec.Spec.Containers[0].Ports = []api.ContainerPort{{
			ContainerPort: 12345,
			Protocol:      api.ProtocolTCP,
		}}
		return spec
	}()

	successCases := map[string]struct {
		opts validation.JobValidationOptions
		job  batch.Job
	}{
		"valid success policy": {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{
							{
								SucceededCount:   ptr.To[int32](1),
								SucceededIndexes: ptr.To("0,2,4"),
							},
							{
								SucceededIndexes: ptr.To("1,3,5-9"),
							},
						},
					},
				},
			},
		},
		"valid pod failure policy": {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.DisruptionTarget,
								Status: api.ConditionTrue,
							}},
						}, {
							Action: batch.PodFailurePolicyActionFailJob,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.PodConditionType("CustomConditionType"),
								Status: api.ConditionFalse,
							}},
						}, {
							Action: batch.PodFailurePolicyActionCount,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
						}, {
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("def"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{4},
							},
						}, {
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpNotIn,
								Values:   []int32{5, 6, 7},
							},
						}},
					},
				},
			},
		},
		"valid pod failure policy with FailIndex": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode:       ptr.To(batch.IndexedCompletion),
					Completions:          pointer.Int32(2),
					BackoffLimitPerIndex: pointer.Int32(1),
					Selector:             validGeneratedSelector,
					ManualSelector:       pointer.Bool(true),
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailIndex,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{10},
							},
						}},
					},
				},
			},
		},
		"valid manual selector": {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "myjob",
					Namespace:   metav1.NamespaceDefault,
					UID:         types.UID("1a2b3c"),
					Annotations: map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.Bool(true),
					Template:       validPodTemplateSpecForManual,
				},
			},
		},
		"valid generated selector": {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
		},
		"valid pod replacement": {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
					PodReplacementPolicy: &terminatingOrFailedPodReplacement,
				},
			},
		},
		"valid pod replacement with failed": {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
					PodReplacementPolicy: &failedPodReplacement,
				},
			},
		},
		"valid hostnet": {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validHostNetPodTemplateSpec,
				},
			},
		},
		"valid NonIndexed completion mode": {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					CompletionMode: ptr.To(batch.NonIndexedCompletion),
				},
			},
		},
		"valid Indexed completion mode": {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    pointer.Int32(2),
					Parallelism:    pointer.Int32(100000),
				},
			},
		},
		"valid parallelism and maxFailedIndexes for high completions when backoffLimitPerIndex is used": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(100_000),
					Parallelism:          pointer.Int32(100_000),
					MaxFailedIndexes:     pointer.Int32(100_000),
					BackoffLimitPerIndex: pointer.Int32(1),
					CompletionMode:       ptr.To(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
		},
		"valid parallelism and maxFailedIndexes for unlimited completions when backoffLimitPerIndex is used": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(1_000_000_000),
					Parallelism:          pointer.Int32(10_000),
					MaxFailedIndexes:     pointer.Int32(10_000),
					BackoffLimitPerIndex: pointer.Int32(1),
					CompletionMode:       ptr.To(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
		},
		"valid job tracking annotation": {
			opts: validation.JobValidationOptions{
				RequirePrefixedLabels: true,
			},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
		},
		"valid batch labels": {
			opts: validation.JobValidationOptions{
				RequirePrefixedLabels: true,
			},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
		},
		"do not allow new batch labels": {
			opts: validation.JobValidationOptions{
				RequirePrefixedLabels: false,
			},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.LegacyControllerUidLabel: "1a2b3c"},
					},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.LegacyControllerUidLabel: "1a2b3c", batch.LegacyJobNameLabel: "myjob"},
						},
						Spec: api.PodSpec{
							RestartPolicy:  api.RestartPolicyOnFailure,
							DNSPolicy:      api.DNSClusterFirst,
							Containers:     []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
							InitContainers: []api.Container{{Name: "def", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
				},
			},
		},
		"valid managedBy field": {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:  validGeneratedSelector,
					Template:  validPodTemplateSpecForGenerated,
					ManagedBy: ptr.To("example.com/foo"),
				},
			},
		},
	}

	negative := int32(-1)
	negative64 := int64(-1)
	errorCases := map[string]struct {
		opts   validation.JobValidationOptions
		job    batch.Job
		errors apivalidation.ExpectedErrorList
	}{
		`spec.managedBy: Too long: may not be longer than 63`: {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:  validGeneratedSelector,
					Template:  validPodTemplateSpecForGenerated,
					ManagedBy: ptr.To("example.com/" + strings.Repeat("x", 60)),
				},
			},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.managedBy", Type: field.ErrorTypeTooLong, Detail: "may not be longer than 63", BadValue: "example.com/" + strings.Repeat("x", 60)},
			},
		},
		`spec.managedBy: Invalid value: "invalid custom controller name": must be a domain-prefixed path (such as "acme.io/foo")`: {
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:  validGeneratedSelector,
					Template:  validPodTemplateSpecForGenerated,
					ManagedBy: ptr.To("invalid custom controller name"),
				},
			},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.managedBy", Type: field.ErrorTypeInvalid, Detail: "must be a domain-prefixed path (such as \"acme.io/foo\")", BadValue: "invalid custom controller name"},
			},
		},
		`spec.successPolicy: Invalid value: batch.SuccessPolicy{Rules:[]batch.SuccessPolicyRule{}}: requires indexed completion mode`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.successPolicy", Type: field.ErrorTypeInvalid, Detail: "requires indexed completion mode", BadValue: batch.SuccessPolicy{Rules: []batch.SuccessPolicyRule{}}},
				{Field: "spec.successPolicy.rules", Type: field.ErrorTypeInvalid, Detail: "at least", NativeSkipReason: "skipped when forbidden"},
			},
		},
		`spec.successPolicy.rules: Required value: at least one rules must be specified when the successPolicy is specified`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy:  &batch.SuccessPolicy{},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.successPolicy.rules", Type: field.ErrorTypeRequired},
			},
		},
		`spec.successPolicy.rules[0]: Required value: at least one of succeededCount or succeededIndexes must be specified`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededCount:   nil,
							SucceededIndexes: nil,
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.successPolicy.rules[0]", Type: field.ErrorTypeRequired, Detail: "at least one of succeededCount or succeededIndexes must be specified"},
			},
		},
		`spec.successPolicy.rules[0].succeededIndexes: Invalid value: "invalid-format": error parsing succeededIndexes: cannot convert string to integer for index: "invalid"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededIndexes: ptr.To("invalid-format"),
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.successPolicy.rules[0].succeededIndexes", Type: field.ErrorTypeInvalid, Detail: "error parsing succeededIndexes: cannot convert string to integer for index: \"invalid\"", BadValue: "invalid-format", SchemaDetail: `should match`},
			},
		},
		`spec.successPolicy.rules[0].succeededIndexes: Too long: must have at most 65536 bytes`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededIndexes: ptr.To(strings.Repeat("1", maxJobSuccessPolicySucceededIndexesLimit+1)),
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.successPolicy.rules[0].succeededIndexes", Type: field.ErrorTypeTooLong, Detail: "must have at most 65536 bytes", BadValue: strings.Repeat("1", maxJobSuccessPolicySucceededIndexesLimit+1), SchemaType: field.ErrorTypeTooLong, SchemaDetail: `may not be longer than 65536`},
				{Field: "spec.successPolicy.rules[0].succeededIndexes", Type: field.ErrorTypeInvalid, Detail: "error parsing succeededIndexes: cannot convert string to integer for index", BadValue: strings.Repeat("1", maxJobSuccessPolicySucceededIndexesLimit+1), SchemaSkipReason: `no parsing of indices in Schema`},
			},
		},
		`spec.successPolicy.rules[0].succeededCount: must be greater than or equal to 0`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededCount: ptr.To[int32](-1),
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.successPolicy.rules[0].succeededCount", Type: field.ErrorTypeInvalid, Detail: "be greater than or equal to 0", BadValue: int64(-1)},
			},
		},
		`spec.successPolicy.rules[0].succeededCount: Invalid value: 6: must be less than or equal to 5 (the number of specified completions)`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededCount: ptr.To[int32](6),
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.successPolicy.rules[0].succeededCount", Type: field.ErrorTypeInvalid, Detail: "must be less than or equal to 5 (the number of specified completions)", BadValue: int32(6), SchemaField: `spec.successPolicy.rules`, SchemaDetail: `must be less than or equal to`},
			},
		},
		`spec.successPolicy.rules[0].succeededCount: Invalid value: 4: must be less than or equal to 3 (the number of indexes in the specified succeededIndexes field)`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededCount:   ptr.To[int32](4),
							SucceededIndexes: ptr.To("0-2"),
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.successPolicy.rules[0].succeededCount", Type: field.ErrorTypeInvalid, Detail: "must be less than or equal to 3 (the number of indexes in the specified succeededIndexes field)", BadValue: int32(4), SchemaSkipReason: `Blocked by lack of advanced list comprehension`},
			},
		},
		`spec.successPolicy.rules: Too many: 21: must have at most 20 items`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: func() []batch.SuccessPolicyRule {
							var rules []batch.SuccessPolicyRule
							for i := 0; i < 21; i++ {
								rules = append(rules, batch.SuccessPolicyRule{
									SucceededCount: ptr.To[int32](5),
								})
							}
							return rules
						}(),
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.successPolicy.rules", Type: field.ErrorTypeTooMany, Detail: "must have at most 20 items", BadValue: 21},
			},
		},
		`spec.podFailurePolicy.rules[0]: Invalid value: specifying one of OnExitCodes and OnPodConditions is required`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0]", Type: field.ErrorTypeInvalid, Detail: "specifying one of OnExitCodes and OnPodConditions is required", BadValue: field.OmitValueType{}},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values[1]: Duplicate value: 11`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{11, 11},
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.values[1]", Type: field.ErrorTypeDuplicate, BadValue: int32(11)},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values: Too many: 256: must have at most 255 items`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values: func() (values []int32) {
									tooManyValues := make([]int32, maxPodFailurePolicyOnExitCodesValues+1)
									for i := range tooManyValues {
										tooManyValues[i] = int32(i)
									}
									return tooManyValues
								}(),
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.values", Type: field.ErrorTypeTooMany, Detail: "must have at most 255 items", BadValue: 256},
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.values[0]", Type: field.ErrorTypeInvalid, Detail: "must not be 0 for the In operator", BadValue: int32(0), SchemaField: `spec.podFailurePolicy.rules[0].onExitCodes.values`},
			},
		},
		`spec.podFailurePolicy.rules: Too many: 21: must have at most 20 items`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: func() []batch.PodFailurePolicyRule {
							tooManyRules := make([]batch.PodFailurePolicyRule, maxPodFailurePolicyRules+1)
							for i := range tooManyRules {
								tooManyRules[i] = batch.PodFailurePolicyRule{
									Action: batch.PodFailurePolicyActionFailJob,
									OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
										Operator: batch.PodFailurePolicyOnExitCodesOpIn,
										Values:   []int32{int32(i + 1)},
									},
								}
							}
							return tooManyRules
						}(),
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules", Type: field.ErrorTypeTooMany, Detail: "must have at most 20 items", BadValue: 21},
			},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions: Too many: 21: must have at most 20 items`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnPodConditions: func() []batch.PodFailurePolicyOnPodConditionsPattern {
								tooManyPatterns := make([]batch.PodFailurePolicyOnPodConditionsPattern, maxPodFailurePolicyOnPodConditionsPatterns+1)
								for i := range tooManyPatterns {
									tooManyPatterns[i] = batch.PodFailurePolicyOnPodConditionsPattern{
										Type:   api.PodConditionType(fmt.Sprintf("CustomType_%d", i)),
										Status: api.ConditionTrue,
									}
								}
								return tooManyPatterns
							}(),
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onPodConditions", Type: field.ErrorTypeTooMany, Detail: "must have at most 20 items", BadValue: 21},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values[2]: Duplicate value: 13`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{12, 13, 13, 13},
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.values[2]", Type: field.ErrorTypeDuplicate, BadValue: int32(13)},
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.values[3]", Type: field.ErrorTypeDuplicate, BadValue: int32(13), SchemaSkipReason: `Schema only reports single duplicate?`},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values: Invalid value: []int32{19, 11}: must be ordered`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{19, 11},
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.values", Type: field.ErrorTypeInvalid, Detail: "must be ordered", BadValue: []int32{19, 11}},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values: Invalid value: []int32{}: at least one value is required`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{},
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.values", Type: field.ErrorTypeInvalid, Detail: "at least one value is required", BadValue: []int32{}, SchemaDetail: `at least 1 items`},
			},
		},
		`spec.podFailurePolicy.rules[0].action: Required value: valid values: ["Count" "FailIndex" "FailJob" "Ignore"]`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: "",
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{1, 2, 3},
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].action", Type: field.ErrorTypeRequired, Detail: "valid values: [\"Count\" \"FailIndex\" \"FailJob\" \"Ignore\"]", SchemaType: field.ErrorTypeNotSupported, SchemaDetail: `supported values: "Count", "FailIndex", "FailJob", "Ignore"`},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.operator: Required value: valid values: ["In" "NotIn"]`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: "",
								Values:   []int32{1, 2, 3},
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.operator", Type: field.ErrorTypeRequired, Detail: "valid values: [\"In\" \"NotIn\"]", BadValue: "", SchemaType: field.ErrorTypeNotSupported, SchemaDetail: `supported values: "In", "NotIn`},
			},
		},
		`spec.podFailurePolicy.rules[0]: Invalid value: specifying both OnExitCodes and OnPodConditions is not supported`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.DisruptionTarget,
								Status: api.ConditionTrue,
							}},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0]", Type: field.ErrorTypeInvalid, Detail: "specifying both OnExitCodes and OnPodConditions is not supported", BadValue: field.OmitValueType{}},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values[1]: Invalid value: 0: must not be 0 for the In operator`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{1, 0, 2},
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.values[1]", Type: field.ErrorTypeInvalid, Detail: "must not be 0 for the In operator", BadValue: int32(0), SchemaField: `spec.podFailurePolicy.rules[0].onExitCodes.values`},
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.values", Type: field.ErrorTypeInvalid, Detail: "must be ordered", BadValue: []int32{1, 0, 2}},
			},
		},
		`spec.podFailurePolicy.rules[1].onExitCodes.containerName: Invalid value: "xyz": must be one of the container or initContainer names in the pod template`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
						}, {
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("xyz"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{5, 6, 7},
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[1].onExitCodes.containerName", Type: field.ErrorTypeInvalid, Detail: "must be one of the container or initContainer names in the pod template", BadValue: "xyz", SchemaField: `spec.podFailurePolicy.rules`},
			},
		},
		`spec.podFailurePolicy.rules[0].action: Unsupported value: "UnknownAction": supported values: "Count", "FailIndex", "FailJob", "Ignore"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: "UnknownAction",
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].action", Type: field.ErrorTypeNotSupported, Detail: "supported values: \"Count\", \"FailIndex\", \"FailJob\", \"Ignore\"", BadValue: batch.PodFailurePolicyAction("UnknownAction")},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.operator: Unsupported value: "UnknownOperator": supported values: "In", "NotIn"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: "UnknownOperator",
								Values:   []int32{1, 2, 3},
							},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onExitCodes.operator", Type: field.ErrorTypeNotSupported, Detail: "supported values: \"In\", \"NotIn\"", BadValue: batch.PodFailurePolicyOnExitCodesOperator("UnknownOperator"), SchemaType: field.ErrorTypeNotSupported},
			},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].status: Required value: valid values: ["False" "True" "Unknown"]`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type: api.DisruptionTarget,
							}},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onPodConditions[0].status", Type: field.ErrorTypeRequired, Detail: "valid values: [\"False\" \"True\" \"Unknown\"]", SchemaType: field.ErrorTypeNotSupported, SchemaDetail: `supported values: "False", "True", "Unknown"`},
			},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].status: Unsupported value: "UnknownStatus": supported values: "False", "True", "Unknown"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.DisruptionTarget,
								Status: "UnknownStatus",
							}},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onPodConditions[0].status", Type: field.ErrorTypeNotSupported, Detail: "supported values: \"False\", \"True\", \"Unknown\"", BadValue: core.ConditionStatus("UnknownStatus")},
			},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].type: Invalid value: "": name part must be non-empty`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Status: api.ConditionTrue,
							}},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onPodConditions[0].type", Type: field.ErrorTypeInvalid, Detail: "name part must be non-empty", BadValue: ""},
				{Field: "spec.podFailurePolicy.rules[0].onPodConditions[0].type", Type: field.ErrorTypeInvalid, Detail: "name part must consist of alphanumeric characters", BadValue: ""},
			},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].type: Invalid value: "Invalid Condition Type": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.PodConditionType("Invalid Condition Type"),
								Status: api.ConditionTrue,
							}},
						}},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podFailurePolicy.rules[0].onPodConditions[0].type", Type: field.ErrorTypeInvalid, Detail: "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')", BadValue: "Invalid Condition Type"},
			},
		},
		`spec.podReplacementPolicy: Unsupported value: "TerminatingOrFailed": supported values: "Failed"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					PodReplacementPolicy: &terminatingOrFailedPodReplacement,
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.DisruptionTarget,
								Status: api.ConditionTrue,
							}},
						},
						},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podReplacementPolicy", Type: field.ErrorTypeNotSupported, Detail: "supported values: \"Failed\"", BadValue: batch.PodReplacementPolicy("TerminatingOrFailed"), SchemaType: field.ErrorTypeInvalid, SchemaDetail: `must be "Failed" when podFailurePolicy is used`},
			},
		},
		`spec.podReplacementPolicy: Unsupported value: "": supported values: "Failed", "TerminatingOrFailed"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					PodReplacementPolicy: (*batch.PodReplacementPolicy)(pointer.String("")),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.podReplacementPolicy", Type: field.ErrorTypeNotSupported, Detail: "\"Failed\", \"TerminatingOrFailed\"", BadValue: batch.PodReplacementPolicy("")},
			},
		},
		`spec.template.spec.restartPolicy: Invalid value: "OnFailure": only "Never" is supported when podFailurePolicy is specified`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validGeneratedSelector.MatchLabels,
						},
						Spec: api.PodSpec{
							RestartPolicy: api.RestartPolicyOnFailure,
							DNSPolicy:     api.DNSClusterFirst,
							Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.template.spec.restartPolicy", Type: field.ErrorTypeInvalid, Detail: "only \"Never\" is supported when podFailurePolicy is specified", BadValue: core.RestartPolicyOnFailure},
			},
		},
		"spec.parallelism:must be greater than or equal to 0": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Parallelism: &negative,
					Selector:    validGeneratedSelector,
					Template:    validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.parallelism", Type: field.ErrorTypeInvalid, Detail: "be greater than or equal to 0", BadValue: int64(-1)},
			},
		},
		"spec.backoffLimit:must be greater than or equal to 0": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					BackoffLimit: pointer.Int32(-1),
					Selector:     validGeneratedSelector,
					Template:     validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.backoffLimit", Type: field.ErrorTypeInvalid, Detail: "be greater than or equal to 0", BadValue: int64(-1)},
			},
		},
		"spec.backoffLimitPerIndex: Invalid value: 1: requires indexed completion mode": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					BackoffLimitPerIndex: pointer.Int32(1),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.backoffLimitPerIndex", Type: field.ErrorTypeInvalid, Detail: "requires indexed completion mode", BadValue: int32(1)},
			},
		},
		"spec.backoffLimitPerIndex:must be greater than or equal to 0": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					BackoffLimitPerIndex: pointer.Int32(-1),
					CompletionMode:       ptr.To(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.backoffLimitPerIndex", Type: field.ErrorTypeInvalid, Detail: "be greater than or equal to 0", BadValue: int64(-1)},
				{Field: "spec.completions", Type: field.ErrorTypeRequired, Detail: "when completion mode is Indexed"},
			},
		},
		"spec.maxFailedIndexes: Invalid value: 11: must be less than or equal to completions": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(10),
					MaxFailedIndexes:     pointer.Int32(11),
					BackoffLimitPerIndex: pointer.Int32(1),
					CompletionMode:       ptr.To(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.maxFailedIndexes", Type: field.ErrorTypeInvalid, Detail: "must be less than or equal to completions", BadValue: int32(11)},
			},
		},
		"spec.maxFailedIndexes: Required value: must be specified when completions is above 100000": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(100_001),
					BackoffLimitPerIndex: pointer.Int32(1),
					CompletionMode:       ptr.To(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.maxFailedIndexes", Type: field.ErrorTypeRequired, Detail: "must be specified when completions is above 100000"},
			},
		},
		"spec.parallelism: Invalid value: 50000: must be less than or equal to 10000 when completions are above 100000 and used with backoff limit per index": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(100_001),
					Parallelism:          pointer.Int32(50_000),
					BackoffLimitPerIndex: pointer.Int32(1),
					MaxFailedIndexes:     pointer.Int32(1),
					CompletionMode:       ptr.To(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.parallelism", Type: field.ErrorTypeInvalid, Detail: "must be less than or equal to 10000 when completions are above 100000 and used with backoff limit per index", BadValue: int32(50_000)},
			},
		},
		"spec.maxFailedIndexes: Invalid value: 100001: must be less than or equal to 100000": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(100_001),
					BackoffLimitPerIndex: pointer.Int32(1),
					MaxFailedIndexes:     pointer.Int32(100_001),
					CompletionMode:       ptr.To(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.maxFailedIndexes", Type: field.ErrorTypeInvalid, Detail: "must be less than or equal to 100000", BadValue: int32(100_001)},
				{Field: "spec.maxFailedIndexes", Type: field.ErrorTypeInvalid, Detail: "must be less than or equal to 10000 when completions are above 100000 and used with backoff limit per index", BadValue: int32(100_001)},
			},
		},
		"spec.maxFailedIndexes: Invalid value: 50000: must be less than or equal to 10000 when completions are above 100000 and used with backoff limit per index": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(100_001),
					BackoffLimitPerIndex: pointer.Int32(1),
					MaxFailedIndexes:     pointer.Int32(50_000),
					CompletionMode:       ptr.To(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.maxFailedIndexes", Type: field.ErrorTypeInvalid, Detail: "must be less than or equal to 10000 when completions are above 100000 and used with backoff limit per index", BadValue: int32(50_000)},
			},
		},
		"spec.maxFailedIndexes:must be greater than or equal to 0": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					BackoffLimitPerIndex: pointer.Int32(1),
					MaxFailedIndexes:     pointer.Int32(-1),
					CompletionMode:       ptr.To(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
					Completions:          pointer.Int32(10),
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.maxFailedIndexes", Type: field.ErrorTypeInvalid, Detail: "be greater than or equal to 0", BadValue: int64(-1)},
			},
		},
		"spec.backoffLimitPerIndex: Required value: when maxFailedIndexes is specified": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					MaxFailedIndexes: pointer.Int32(1),
					CompletionMode:   ptr.To(batch.IndexedCompletion),
					Selector:         validGeneratedSelector,
					Template:         validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.backoffLimitPerIndex", Type: field.ErrorTypeRequired, Detail: "when maxFailedIndexes is specified"},
				{Field: "spec.completions", Type: field.ErrorTypeRequired, Detail: "when completion mode is Indexed"},
			},
		},
		"spec.completions:must be greater than or equal to 0": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Completions: &negative,
					Selector:    validGeneratedSelector,
					Template:    validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.completions", Type: field.ErrorTypeInvalid, Detail: "be greater than or equal to 0", BadValue: int64(-1)},
			},
		},
		"spec.activeDeadlineSeconds:must be greater than or equal to 0": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					ActiveDeadlineSeconds: &negative64,
					Selector:              validGeneratedSelector,
					Template:              validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.activeDeadlineSeconds", Type: field.ErrorTypeInvalid, Detail: "be greater than or equal to 0", BadValue: int64(-1)},
			},
		},
		"spec.selector:Required value": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Template: validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.selector", Type: field.ErrorTypeRequired, SchemaSkipReason: `Not required to user, generated by syste,`},
				{Field: "spec.template.metadata.labels", Type: field.ErrorTypeInvalid, Detail: "`selector` does not match template `labels`", BadValue: getValidGeneratedSelector().MatchLabels, SchemaSkipReason: `No need to validate generated selector in schema`},
			},
		},
		"spec.template.metadata.labels: Invalid value: map[string]string{\"y\":\"z\"}: `selector` does not match template `labels`": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.Bool(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"y": "z"},
						},
						Spec: api.PodSpec{
							RestartPolicy: api.RestartPolicyOnFailure,
							DNSPolicy:     api.DNSClusterFirst,
							Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.template.metadata.labels", Type: field.ErrorTypeInvalid, Detail: "`selector` does not match template `labels`", BadValue: map[string]string{"y": "z"}, SchemaSkipReason: `Blocked by lack of CEL selector library`},
			},
		},
		"spec.template.metadata.labels: Invalid value: map[string]string{\"controller-uid\":\"4d5e6f\"}: `selector` does not match template `labels`": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.Bool(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"controller-uid": "4d5e6f"},
						},
						Spec: api.PodSpec{
							RestartPolicy: api.RestartPolicyOnFailure,
							DNSPolicy:     api.DNSClusterFirst,
							Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.template.metadata.labels", Type: field.ErrorTypeInvalid, Detail: "`selector` does not match template `labels`", BadValue: map[string]string{"controller-uid": "4d5e6f"}, SchemaSkipReason: `Blocked by lack of CEL selector library`},
			},
		},
		"spec.template.spec.restartPolicy: Required value": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.Bool(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validManualSelector.MatchLabels,
						},
						Spec: api.PodSpec{
							RestartPolicy: api.RestartPolicyAlways,
							DNSPolicy:     api.DNSClusterFirst,
							Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.template.spec.restartPolicy", Type: field.ErrorTypeRequired, Detail: `"OnFailure", "Never"`, SchemaType: field.ErrorTypeNotSupported},
			},
		},
		"spec.template.spec.restartPolicy: Unsupported value": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.Bool(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validManualSelector.MatchLabels,
						},
						Spec: api.PodSpec{
							RestartPolicy: "Invalid",
							DNSPolicy:     api.DNSClusterFirst,
							Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.template.spec.restartPolicy", Type: field.ErrorTypeNotSupported, Detail: "supported values: \"Always\", \"OnFailure\", \"Never\"", BadValue: core.RestartPolicy("Invalid"), SchemaDetail: "supported values: \"Always\", \"Never\", \"OnFailure\""},
				{Field: "spec.template.spec.restartPolicy", Type: field.ErrorTypeNotSupported, Detail: "supported values: \"OnFailure\", \"Never\"", BadValue: core.RestartPolicy("Invalid")},
			},
		},
		"spec.ttlSecondsAfterFinished: must be greater than or equal to 0": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					TTLSecondsAfterFinished: &negative,
					Selector:                validGeneratedSelector,
					Template:                validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.ttlSecondsAfterFinished", Type: field.ErrorTypeInvalid, Detail: "be greater than or equal to 0", BadValue: int64(-1)},
			},
		},
		"spec.completions: Required value: when completion mode is Indexed": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					CompletionMode: ptr.To(batch.IndexedCompletion),
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.completions", Type: field.ErrorTypeRequired, Detail: "when completion mode is Indexed"},
			},
		},
		"spec.parallelism: must be less than or equal to 100000 when completion mode is Indexed": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					CompletionMode: ptr.To(batch.IndexedCompletion),
					Completions:    pointer.Int32(2),
					Parallelism:    pointer.Int32(100001),
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.parallelism", Type: field.ErrorTypeInvalid, Detail: "must be less than or equal to 100000 when completion mode is Indexed", BadValue: int32(100001)},
			},
		},
		"spec.template.metadata.labels[controller-uid]: Required value: must be '1a2b3c'": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.LegacyControllerUidLabel: "1a2b3c"},
					},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.LegacyJobNameLabel: "myjob"},
						},
						Spec: api.PodSpec{
							RestartPolicy:  api.RestartPolicyOnFailure,
							DNSPolicy:      api.DNSClusterFirst,
							Containers:     []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
							InitContainers: []api.Container{{Name: "def", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
				},
			},
			opts: validation.JobValidationOptions{},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.template.metadata.labels[controller-uid]", Type: field.ErrorTypeRequired, Detail: "must be '1a2b3c'", SchemaSkipReason: `Skipped for now until we validate the selectors`},
				{Field: "spec.template.metadata.labels", Type: field.ErrorTypeInvalid, Detail: "does not match template `labels`", BadValue: map[string]string{batch.LegacyJobNameLabel: "myjob"}, SchemaSkipReason: `Blocked by lack of CEL label selector library`},
			},
		},
		"metadata.uid: Required value": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.LegacyControllerUidLabel: "test"},
					},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.LegacyJobNameLabel: "myjob"},
						},
						Spec: api.PodSpec{
							RestartPolicy:  api.RestartPolicyOnFailure,
							DNSPolicy:      api.DNSClusterFirst,
							Containers:     []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
							InitContainers: []api.Container{{Name: "def", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
				},
			},
			opts: validation.JobValidationOptions{},
			errors: apivalidation.ExpectedErrorList{
				{Field: "metadata.uid", Type: field.ErrorTypeRequired, SchemaSkipReason: `filled in server-side`},
				{Field: "spec.selector", Type: field.ErrorTypeInvalid, Detail: "`selector` not auto-generated", BadValue: &metav1.LabelSelector{MatchLabels: map[string]string{batch.LegacyControllerUidLabel: "test"}}, SchemaSkipReason: `Blocked by lack of CEL label selector library`},
				{Field: "spec.template.metadata.labels", Type: field.ErrorTypeInvalid, Detail: "`selector` does not match template `labels`", BadValue: map[string]string{batch.LegacyJobNameLabel: "myjob"}, SchemaSkipReason: `Blocked by lack of CEL label selector library`},
				{Field: "spec.template.metadata.labels[controller-uid]", Type: field.ErrorTypeRequired, Detail: "must be ''", SchemaSkipReason: `Skipped for now until we validate the selectors`},
			},
		},
		"spec.selector: Invalid value: v1.LabelSelector{MatchLabels:map[string]string{\"a\":\"b\"}, MatchExpressions:[]v1.LabelSelectorRequirement(nil)}: `selector` not auto-generated": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Template: validPodTemplateSpecForGenerated,
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.selector", Type: field.ErrorTypeInvalid, Detail: "`selector` not auto-generated", BadValue: &metav1.LabelSelector{MatchLabels: map[string]string{"a": "b"}, MatchExpressions: []metav1.LabelSelectorRequirement(nil)}, SchemaSkipReason: `Blocked by lack of CEL label selector library`},
				{Field: "spec.template.metadata.labels", Type: field.ErrorTypeInvalid, Detail: "`selector` does not match template `labels`", BadValue: validGeneratedSelector.MatchLabels, SchemaSkipReason: `Blocked by lack of CEL label selector library`},
			},
		},
		"spec.template.metadata.labels[batch.kubernetes.io/controller-uid]: Required value: must be '1a2b3c'": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.ControllerUidLabel: "1a2b3c"},
					},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.JobNameLabel: "myjob", batch.LegacyControllerUidLabel: "1a2b3c", batch.LegacyJobNameLabel: "myjob"},
						},
						Spec: api.PodSpec{
							RestartPolicy:  api.RestartPolicyOnFailure,
							DNSPolicy:      api.DNSClusterFirst,
							Containers:     []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
							InitContainers: []api.Container{{Name: "def", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
						},
					},
				},
			},
			opts: validation.JobValidationOptions{RequirePrefixedLabels: true},
			errors: apivalidation.ExpectedErrorList{
				{Field: "spec.template.metadata.labels[batch.kubernetes.io/controller-uid]", Type: field.ErrorTypeRequired, Detail: "must be '1a2b3c'", SchemaSkipReason: `Skipped for now until we validate the selectors`},
				{Field: "spec.template.metadata.labels", Type: field.ErrorTypeInvalid, Detail: "does not match template `labels`", BadValue: map[string]string{batch.JobNameLabel: "myjob", batch.LegacyControllerUidLabel: "1a2b3c", batch.LegacyJobNameLabel: "myjob"}, SchemaSkipReason: `Blocked by lack of CEL label selector library`},
			},
		},
	}

	cases := []apivalidation.TestCase[*batch.Job, validation.JobValidationOptions]{}
	for name, test := range successCases {
		cases = append(cases, apivalidation.TestCase[*batch.Job, validation.JobValidationOptions]{
			Name:    name,
			Object:  &test.job,
			Options: test.opts,
		})
	}

	for name, test := range errorCases {
		cases = append(cases, apivalidation.TestCase[*batch.Job, validation.JobValidationOptions]{
			Name:           name,
			Object:         &test.job,
			Options:        test.opts,
			ExpectedErrors: test.errors,
		})
	}
	sort.SliceStable(cases, func(i, j int) bool {
		return cases[i].Name < cases[j].Name
	})

	apivalidation.TestValidate[batch.Job, validation.JobValidationOptions](t, batchScheme, batchDefs, func(j *batch.Job, jvo validation.JobValidationOptions) field.ErrorList {
		return validation.ValidateJob(j, jvo)
	}, cases...)

}
