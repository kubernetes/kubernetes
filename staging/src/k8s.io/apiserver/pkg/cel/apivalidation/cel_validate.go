package apivalidation

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"regexp"
	"strings"

	celgo "github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
	"k8s.io/apimachinery/pkg/util/validation/field"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/common"
)

// This file is mostly a copy & paste of k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel/validation.go
// With structural.Schema replaced with common.Schema, and unused functions
// removed
func validateExpressions(ctx context.Context, result *ValidationContext, sts common.Schema, compiledRules []CompilationResult, remainingCostBudget int64) (remainingBudget int64) {
	remainingBudget = remainingCostBudget
	fldPath := result.Path
	if result.Value.IsNull() {
		// We only validate non-null values. Rules that need to check for the state of a nullable value or the presence of an optional
		// field must do so from the surrounding schema. E.g. if an array has nullable string items, a rule on the array
		// schema can check if items are null, but a rule on the nullable string schema only validates the non-null strings.
		return remainingBudget
	} else if len(compiledRules) == 0 {
		// nothing to do
		return remainingBudget
	} else if remainingBudget <= 0 {
		result.AddErrors(field.Invalid(fldPath, sts.Type(), fmt.Sprintf("validation failed due to running out of cost budget, no further validation rules will be run")))
		return -1
	} else if result.IsRoot() || sts.IsXEmbeddedResource() {
		sts = sts.WithTypeAndObjectMeta()
	}

	usesOldSelf := false
	for _, compiled := range compiledRules {
		if compiled.TransitionRule {
			usesOldSelf = true
			break
		}
	}

	var activation interpreter.Activation
	if usesOldSelf && result.OldValue != nil {
		activation = validationActivationWithOldSelf(sts, result.Value.Unstructured(), result.OldValue.Unstructured())
	} else {
		activation = validationActivationWithoutOldSelf(sts, result.Value.Unstructured(), nil)
	}

	for i, compiled := range compiledRules {
		rule := sts.XValidations()[i]
		if compiled.Error != nil {
			result.AddErrors(field.Invalid(fldPath, sts.Type(), fmt.Sprintf("rule compile error: %v", compiled.Error)))
			continue
		}
		if compiled.Program == nil {
			// rule is empty
			continue
		}
		//!TODO: classify transition rule errors as UPDATE ERRORS
		if compiled.TransitionRule && result.OldValue == nil {
			// transition rules are evaluated only if there is a comparable existing value
			continue
		}
		evalResult, evalDetails, err := compiled.Program.ContextEval(ctx, activation)
		if evalDetails == nil {
			result.AddErrors(field.InternalError(fldPath, fmt.Errorf("runtime cost could not be calculated for validation rule: %v, no further validation rules will be run", ruleErrorString(rule))))
			return -1
		} else {
			rtCost := evalDetails.ActualCost()
			if rtCost == nil {
				result.AddErrors(field.Invalid(fldPath, sts.Type(), fmt.Sprintf("runtime cost could not be calculated for validation rule: %v, no further validation rules will be run", ruleErrorString(rule))))
				return -1
			} else {
				if *rtCost > math.MaxInt64 || int64(*rtCost) > remainingBudget {
					result.AddErrors(field.Invalid(fldPath, sts.Type(), fmt.Sprintf("validation failed due to running out of cost budget, no further validation rules will be run")))
					return -1
				}
				remainingBudget -= int64(*rtCost)
			}
		}
		if err != nil {
			// see types.Err for list of well defined error types
			if strings.HasPrefix(err.Error(), "no such overload") {
				// Most overload errors are caught by the compiler, which provides details on where exactly in the rule
				// error was found. Here, an overload error has occurred at runtime no details are provided, so we
				// append a more descriptive error message. This error can only occur when static type checking has
				// been bypassed. int-or-string is typed as dynamic and so bypasses compiler type checking.
				result.AddErrors(field.Invalid(fldPath, sts.Type(), fmt.Sprintf("'%v': call arguments did not match a supported operator, function or macro signature for rule: %v", err, ruleErrorString(rule))))
			} else if strings.HasPrefix(err.Error(), "operation cancelled: actual cost limit exceeded") {
				result.AddErrors(field.Invalid(fldPath, sts.Type(), fmt.Sprintf("'%v': no further validation rules will be run due to call cost exceeds limit for rule: %v", err, ruleErrorString(rule))))
				return -1
			} else {
				// no such key: {key}, index out of bounds: {index}, integer overflow, division by zero, ...
				result.AddErrors(field.Invalid(fldPath, sts.Type(), fmt.Sprintf("%v evaluating rule: %v", err, ruleErrorString(rule))))
			}
			continue
		}
		if evalResult != types.True {
			if compiled.MessageExpression != nil {
				messageExpression, newRemainingBudget, msgErr := evalMessageExpression(ctx, compiled.MessageExpression, rule.MessageExpression(), activation, remainingBudget)
				if msgErr != nil {
					if msgErr.Type == cel.ErrorTypeInternal {
						result.AddErrors(field.InternalError(fldPath, msgErr))
						return -1
					} else if msgErr.Type == cel.ErrorTypeInvalid {
						result.AddErrors(field.Invalid(fldPath, sts.Type(), msgErr.Error()))
						return -1
					} else {
						result.AddErrors(field.Invalid(fldPath, sts.Type(), ruleMessageOrDefault(rule)))
						remainingBudget = newRemainingBudget
					}
				} else {
					result.AddErrors(field.Invalid(fldPath, sts.Type(), messageExpression))
					remainingBudget = newRemainingBudget
				}
			} else {
				result.AddErrors(field.Invalid(fldPath, sts.Type(), ruleMessageOrDefault(rule)))
			}
		}
	}
	return remainingBudget
}

// ValidFieldPath validates that jsonPath is a valid JSON Path containing only field and map accessors
// that are valid for the given schema, and returns a field.Path representation of the validated jsonPath or an error.
func ValidFieldPath(jsonPath string, schema common.Schema) (validFieldPath *field.Path, err error) {
	appendToPath := func(name string, isNamed bool) error {
		if !isNamed {
			validFieldPath = validFieldPath.Key(name)
			schema = schema.AdditionalProperties().Schema()
		} else {
			validFieldPath = validFieldPath.Child(name)
			val, ok := schema.Properties()[name]
			if !ok {
				return fmt.Errorf("does not refer to a valid field")
			}
			schema = val
		}
		return nil
	}

	validFieldPath = nil

	scanner := bufio.NewScanner(strings.NewReader(jsonPath))

	// configure the scanner to split the string into tokens.
	// The three delimiters ('.', '[', ']') will be returned as single char tokens.
	// All other text between delimiters is returned as string tokens.
	scanner.Split(func(data []byte, atEOF bool) (advance int, token []byte, err error) {
		if len(data) > 0 {
			for i := 0; i < len(data); i++ {
				// If in a single quoted string, look for the end of string
				// ignoring delimiters.
				if data[0] == '\'' {
					if i > 0 && data[i] == '\'' && data[i-1] != '\\' {
						// Return quoted string
						return i + 1, data[:i+1], nil
					}
					continue
				}
				switch data[i] {
				case '.', '[', ']': // delimiters
					if i == 0 {
						// Return the delimiter.
						return 1, data[:1], nil
					} else {
						// Return identifier leading up to the delimiter.
						// The next call to split will return the delimiter.
						return i, data[:i], nil
					}
				}
			}
			if atEOF {
				// Return the string.
				return len(data), data, nil
			}
		}
		return 0, nil, nil
	})

	var tok string
	var isNamed bool
	for scanner.Scan() {
		tok = scanner.Text()
		switch tok {
		case "[":
			if !scanner.Scan() {
				return nil, fmt.Errorf("unexpected end of JSON path")
			}
			tok = scanner.Text()
			if len(tok) < 2 || tok[0] != '\'' || tok[len(tok)-1] != '\'' {
				return nil, fmt.Errorf("expected single quoted string but got %s", tok)
			}
			unescaped, err := unescapeSingleQuote(tok[1 : len(tok)-1])
			if err != nil {
				return nil, fmt.Errorf("invalid string literal: %v", err)
			}

			if schema.Properties() != nil {
				isNamed = true
			} else if schema.AdditionalProperties() != nil {
				isNamed = false
			} else {
				return nil, fmt.Errorf("does not refer to a valid field")
			}
			if err := appendToPath(unescaped, isNamed); err != nil {
				return nil, err
			}
			if !scanner.Scan() {
				return nil, fmt.Errorf("unexpected end of JSON path")
			}
			tok = scanner.Text()
			if tok != "]" {
				return nil, fmt.Errorf("expected ] but got %s", tok)
			}
		case ".":
			if !scanner.Scan() {
				return nil, fmt.Errorf("unexpected end of JSON path")
			}
			tok = scanner.Text()
			if schema.Properties() != nil {
				isNamed = true
			} else if schema.AdditionalProperties() != nil {
				isNamed = false
			} else {
				return nil, fmt.Errorf("does not refer to a valid field")
			}
			if err := appendToPath(tok, isNamed); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("expected [ or . but got: %s", tok)
		}
	}
	return validFieldPath, nil
}

// evalMessageExpression evaluates the given message expression and returns the evaluated string form and the remaining budget, or an error if one
// occurred during evaluation.
func evalMessageExpression(ctx context.Context, expr celgo.Program, exprSrc string, activation interpreter.Activation, remainingBudget int64) (string, int64, *cel.Error) {
	evalResult, evalDetails, err := expr.ContextEval(ctx, activation)
	if evalDetails == nil {
		return "", -1, &cel.Error{
			Type:   cel.ErrorTypeInternal,
			Detail: fmt.Sprintf("runtime cost could not be calculated for messageExpression: %q", exprSrc),
		}
	}
	rtCost := evalDetails.ActualCost()
	if rtCost == nil {
		return "", -1, &cel.Error{
			Type:   cel.ErrorTypeInternal,
			Detail: fmt.Sprintf("runtime cost could not be calculated for messageExpression: %q", exprSrc),
		}
	} else if *rtCost > math.MaxInt64 || int64(*rtCost) > remainingBudget {
		return "", -1, &cel.Error{
			Type:   cel.ErrorTypeInvalid,
			Detail: "messageExpression evaluation failed due to running out of cost budget, no further validation rules will be run",
		}
	}
	if err != nil {
		if strings.HasPrefix(err.Error(), "operation cancelled: actual cost limit exceeded") {
			return "", -1, &cel.Error{
				Type:   cel.ErrorTypeInvalid,
				Detail: fmt.Sprintf("no further validation rules will be run due to call cost exceeds limit for messageExpression: %q", exprSrc),
			}
		}
		return "", remainingBudget - int64(*rtCost), &cel.Error{
			Detail: fmt.Sprintf("messageExpression evaluation failed due to: %v", err.Error()),
		}
	}
	messageStr, ok := evalResult.Value().(string)
	if !ok {
		return "", remainingBudget - int64(*rtCost), &cel.Error{
			Detail: "messageExpression failed to convert to string",
		}
	}
	trimmedMsgStr := strings.TrimSpace(messageStr)
	if len(trimmedMsgStr) > celconfig.MaxEvaluatedMessageExpressionSizeBytes {
		return "", remainingBudget - int64(*rtCost), &cel.Error{
			Detail: fmt.Sprintf("messageExpression beyond allowable length of %d", celconfig.MaxEvaluatedMessageExpressionSizeBytes),
		}
	} else if hasNewlines(trimmedMsgStr) {
		return "", remainingBudget - int64(*rtCost), &cel.Error{
			Detail: "messageExpression should not contain line breaks",
		}
	} else if len(trimmedMsgStr) == 0 {
		return "", remainingBudget - int64(*rtCost), &cel.Error{
			Detail: "messageExpression should evaluate to a non-empty string",
		}
	}
	return trimmedMsgStr, remainingBudget - int64(*rtCost), nil
}

var newlineMatcher = regexp.MustCompile(`[\n]+`)

func hasNewlines(s string) bool {
	return newlineMatcher.MatchString(s)
}

func ruleMessageOrDefault(rule common.ValidationRule) string {
	if len(rule.Message()) == 0 {
		return fmt.Sprintf("failed rule: %s", ruleErrorString(rule))
	} else {
		return strings.TrimSpace(rule.Message())
	}
}

func ruleErrorString(rule common.ValidationRule) string {
	if len(rule.Message()) > 0 {
		return strings.TrimSpace(rule.Message())
	}
	return strings.TrimSpace(rule.Rule())
}

type validationActivation struct {
	self, oldSelf ref.Val
	hasOldSelf    bool
}

func validationActivationWithOldSelf(sts common.Schema, obj, oldObj interface{}) interpreter.Activation {
	va := &validationActivation{
		self: common.UnstructuredToVal(obj, sts),
	}
	if oldObj != nil {
		va.oldSelf = common.UnstructuredToVal(oldObj, sts) // +k8s:verify-mutation:reason=clone
		va.hasOldSelf = true                               // +k8s:verify-mutation:reason=clone
	}
	return va
}

func validationActivationWithoutOldSelf(sts common.Schema, obj, _ interface{}) interpreter.Activation {
	return &validationActivation{
		self: common.UnstructuredToVal(obj, sts),
	}
}

func (a *validationActivation) ResolveName(name string) (interface{}, bool) {
	switch name {
	case ScopedVarName:
		return a.self, true
	case OldScopedVarName:
		return a.oldSelf, a.hasOldSelf
	default:
		return nil, false
	}
}

func (a *validationActivation) Parent() interpreter.Activation {
	return nil
}
