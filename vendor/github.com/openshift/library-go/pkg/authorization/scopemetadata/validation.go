package scopemetadata

import (
	"fmt"

	kutilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"

	oauthv1 "github.com/openshift/api/oauth/v1"
)

func ValidateScopes(scopes []string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(scopes) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, "may not be empty"))
	}

	for i, scope := range scopes {
		illegalCharacter := false
		// https://tools.ietf.org/html/rfc6749#section-3.3 (full list of allowed chars is %x21 / %x23-5B / %x5D-7E)
		// for those without an ascii table, that's `!`, `#-[`, `]-~` inclusive.
		for _, ch := range scope {
			switch {
			case ch == '!':
			case ch >= '#' && ch <= '[':
			case ch >= ']' && ch <= '~':
			default:
				allErrs = append(allErrs, field.Invalid(fldPath.Index(i), scope, fmt.Sprintf("%v not allowed", ch)))
				illegalCharacter = true
			}
		}
		if illegalCharacter {
			continue
		}

		found := false
		for _, evaluator := range ScopeDescribers {
			if !evaluator.Handles(scope) {
				continue
			}

			found = true
			if err := evaluator.Validate(scope); err != nil {
				allErrs = append(allErrs, field.Invalid(fldPath.Index(i), scope, err.Error()))
				break
			}
		}

		if !found {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i), scope, "no scope handler found"))
		}
	}

	return allErrs
}

func ValidateScopeRestrictions(client *oauthv1.OAuthClient, scopes ...string) error {
	if len(scopes) == 0 {
		return fmt.Errorf("%s may not request unscoped tokens", client.Name)
	}

	if len(client.ScopeRestrictions) == 0 {
		return nil
	}

	errs := []error{}
	for _, scope := range scopes {
		if err := validateScopeRestrictions(client, scope); err != nil {
			errs = append(errs, err)
		}
	}

	return kutilerrors.NewAggregate(errs)
}

func validateScopeRestrictions(client *oauthv1.OAuthClient, scope string) error {
	errs := []error{}

	for _, restriction := range client.ScopeRestrictions {
		if len(restriction.ExactValues) > 0 {
			if err := validateLiteralScopeRestrictions(scope, restriction.ExactValues); err != nil {
				errs = append(errs, err)
				continue
			}
			return nil
		}

		if restriction.ClusterRole != nil {
			if !ClusterRoleEvaluatorHandles(scope) {
				continue
			}
			if err := validateClusterRoleScopeRestrictions(scope, *restriction.ClusterRole); err != nil {
				errs = append(errs, err)
				continue
			}
			return nil
		}
	}

	// if we got here, then nothing matched.   If we already have errors, do nothing, otherwise add one to make it report failed.
	if len(errs) == 0 {
		errs = append(errs, fmt.Errorf("%v did not match any scope restriction", scope))
	}

	return kutilerrors.NewAggregate(errs)
}

func validateLiteralScopeRestrictions(scope string, literals []string) error {
	for _, literal := range literals {
		if literal == scope {
			return nil
		}
	}

	return fmt.Errorf("%v not found in %v", scope, literals)
}

func validateClusterRoleScopeRestrictions(scope string, restriction oauthv1.ClusterRoleScopeRestriction) error {
	role, namespace, escalating, err := ClusterRoleEvaluatorParseScope(scope)
	if err != nil {
		return err
	}

	foundName := false
	for _, restrictedRoleName := range restriction.RoleNames {
		if restrictedRoleName == "*" || restrictedRoleName == role {
			foundName = true
			break
		}
	}
	if !foundName {
		return fmt.Errorf("%v does not use an approved name", scope)
	}

	foundNamespace := false
	for _, restrictedNamespace := range restriction.Namespaces {
		if restrictedNamespace == "*" || restrictedNamespace == namespace {
			foundNamespace = true
			break
		}
	}
	if !foundNamespace {
		return fmt.Errorf("%v does not use an approved namespace", scope)
	}

	if escalating && !restriction.AllowEscalation {
		return fmt.Errorf("%v is not allowed to escalate", scope)
	}

	return nil
}
