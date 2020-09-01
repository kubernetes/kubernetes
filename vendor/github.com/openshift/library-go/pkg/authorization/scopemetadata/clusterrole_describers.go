package scopemetadata

import (
	"fmt"
	"strings"
)

// role:<clusterrole name>:<namespace to allow the cluster role, * means all>
type ClusterRoleEvaluator struct{}

var clusterRoleEvaluatorInstance = ClusterRoleEvaluator{}

func (ClusterRoleEvaluator) Handles(scope string) bool {
	return ClusterRoleEvaluatorHandles(scope)
}

func (e ClusterRoleEvaluator) Validate(scope string) error {
	_, _, _, err := ClusterRoleEvaluatorParseScope(scope)
	return err
}

func (e ClusterRoleEvaluator) Describe(scope string) (string, string, error) {
	roleName, scopeNamespace, escalating, err := ClusterRoleEvaluatorParseScope(scope)
	if err != nil {
		return "", "", err
	}

	// Anything you can do [in project "foo" | server-wide] that is also allowed by the "admin" role[, except access escalating resources like secrets]

	scopePhrase := ""
	if scopeNamespace == scopesAllNamespaces {
		scopePhrase = "server-wide"
	} else {
		scopePhrase = fmt.Sprintf("in project %q", scopeNamespace)
	}

	warning := ""
	escalatingPhrase := ""
	if escalating {
		warning = fmt.Sprintf("Includes access to escalating resources like secrets")
	} else {
		escalatingPhrase = ", except access escalating resources like secrets"
	}

	description := fmt.Sprintf("Anything you can do %s that is also allowed by the %q role%s", scopePhrase, roleName, escalatingPhrase)

	return description, warning, nil
}

func ClusterRoleEvaluatorHandles(scope string) bool {
	return strings.HasPrefix(scope, clusterRoleIndicator)
}

// ClusterRoleEvaluatorParseScope parses the requested scope, determining the requested role name, namespace, and if
// access to escalating objects is required.  It will return an error if it doesn't parse cleanly
func ClusterRoleEvaluatorParseScope(scope string) (string /*role name*/, string /*namespace*/, bool /*escalating*/, error) {
	if !ClusterRoleEvaluatorHandles(scope) {
		return "", "", false, fmt.Errorf("bad format for scope %v", scope)
	}
	return parseClusterRoleScope(scope)
}

func parseClusterRoleScope(scope string) (string /*role name*/, string /*namespace*/, bool /*escalating*/, error) {
	if !strings.HasPrefix(scope, clusterRoleIndicator) {
		return "", "", false, fmt.Errorf("bad format for scope %v", scope)
	}
	escalating := false
	if strings.HasSuffix(scope, ":!") {
		escalating = true
		// clip that last segment before parsing the rest
		scope = scope[:strings.LastIndex(scope, ":")]
	}

	tokens := strings.SplitN(scope, ":", 2)
	if len(tokens) != 2 {
		return "", "", false, fmt.Errorf("bad format for scope %v", scope)
	}

	// namespaces can't have colons, but roles can.  pick last.
	lastColonIndex := strings.LastIndex(tokens[1], ":")
	if lastColonIndex <= 0 || lastColonIndex == (len(tokens[1])-1) {
		return "", "", false, fmt.Errorf("bad format for scope %v", scope)
	}

	return tokens[1][0:lastColonIndex], tokens[1][lastColonIndex+1:], escalating, nil
}
