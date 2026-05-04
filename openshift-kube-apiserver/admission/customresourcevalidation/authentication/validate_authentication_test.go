package authentication

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	configv1 "github.com/openshift/api/config/v1"
	"golang.org/x/sync/singleflight"
	"k8s.io/apimachinery/pkg/util/validation/field"
	authenticationcel "k8s.io/apiserver/pkg/authentication/cel"
	"k8s.io/apiserver/pkg/warning"
	"k8s.io/utils/lru"
)

func TestFailValidateAuthenticationSpec(t *testing.T) {
	errorCases := map[string]struct {
		spec       configv1.AuthenticationSpec
		errorType  field.ErrorType
		errorField string
	}{
		"invalid metadata ref": {
			spec: configv1.AuthenticationSpec{
				Type: "",
				OAuthMetadata: configv1.ConfigMapNameReference{
					Name: "../shadow",
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.oauthMetadata.name",
		},
		"invalid webhook ref": {
			spec: configv1.AuthenticationSpec{
				WebhookTokenAuthenticator: &configv1.WebhookTokenAuthenticator{
					KubeConfig: configv1.SecretNameReference{Name: "this+that"},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.webhookTokenAuthenticator.kubeConfig.name",
		},
		"valid webhook ref": {
			spec: configv1.AuthenticationSpec{
				WebhookTokenAuthenticator: &configv1.WebhookTokenAuthenticator{
					KubeConfig: configv1.SecretNameReference{Name: "this"},
				},
			},
		},
		"invalid webhook ref for a Type": {
			spec: configv1.AuthenticationSpec{
				Type: "OIDC",
				WebhookTokenAuthenticator: &configv1.WebhookTokenAuthenticator{
					KubeConfig: configv1.SecretNameReference{Name: "this"},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.webhookTokenAuthenticator",
		},
		"invalid UID CEL expression": {
			spec: configv1.AuthenticationSpec{
				Type: "OIDC",
				OIDCProviders: []configv1.OIDCProvider{
					{
						ClaimMappings: configv1.TokenClaimMappings{
							UID: &configv1.TokenClaimOrExpressionMapping{
								Expression: "!@^#&(!^@(*#&(",
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.oidcProviders[0].claimMappings.uid.expression",
		},
		"invalid Extra CEL expression": {
			spec: configv1.AuthenticationSpec{
				Type: "OIDC",
				OIDCProviders: []configv1.OIDCProvider{
					{
						ClaimMappings: configv1.TokenClaimMappings{
							Extra: []configv1.ExtraMapping{
								{
									Key:             "foo/bar",
									ValueExpression: "!@*(&#^(!@*)&^&",
								},
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.oidcProviders[0].claimMappings.extra[0].valueExpression",
		},
		"invalid username CEL expression": {
			spec: configv1.AuthenticationSpec{
				Type: "OIDC",
				OIDCProviders: []configv1.OIDCProvider{
					{
						ClaimMappings: configv1.TokenClaimMappings{
							Username: configv1.UsernameClaimMapping{
								Expression: "!@^#",
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.oidcProviders[0].claimMappings.username.expression",
		},
		"invalid groups CEL expression": {
			spec: configv1.AuthenticationSpec{
				Type: "OIDC",
				OIDCProviders: []configv1.OIDCProvider{
					{
						ClaimMappings: configv1.TokenClaimMappings{
							Groups: configv1.PrefixedClaimMapping{
								TokenClaimMapping: configv1.TokenClaimMapping{
									Expression: "!@^#",
								},
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.oidcProviders[0].claimMappings.groups.expression",
		},
		"invalid claimValidationRule CEL expression": {
			spec: configv1.AuthenticationSpec{
				Type: "OIDC",
				OIDCProviders: []configv1.OIDCProvider{
					{
						ClaimValidationRules: []configv1.TokenClaimValidationRule{
							{
								Type: configv1.TokenValidationRuleTypeCEL,
								CEL: configv1.TokenClaimValidationCELRule{
									Expression: "user.groups",
									Message:    "invalid",
								},
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.oidcProviders[0].claimValidationRules[0].cel.expression",
		},
		"invalid userValidationRule CEL expression": {
			spec: configv1.AuthenticationSpec{
				Type: "OIDC",
				OIDCProviders: []configv1.OIDCProvider{
					{
						UserValidationRules: []configv1.TokenUserValidationRule{
							{
								Expression: "claims.email",
								Message:    "invalid",
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.oidcProviders[0].userValidationRules[0].expression",
		},
		"claimValidationRule CEL expression does not return bool": {
			spec: configv1.AuthenticationSpec{
				Type: "OIDC",
				OIDCProviders: []configv1.OIDCProvider{
					{
						ClaimValidationRules: []configv1.TokenClaimValidationRule{
							{
								Type: configv1.TokenValidationRuleTypeCEL,
								CEL: configv1.TokenClaimValidationCELRule{
									Expression: "claims.email",
									Message:    "invalid",
								},
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.oidcProviders[0].claimValidationRules[0].cel.expression",
		},
		"userValidationRule CEL expression does not return bool": {
			spec: configv1.AuthenticationSpec{
				Type: "OIDC",
				OIDCProviders: []configv1.OIDCProvider{
					{
						UserValidationRules: []configv1.TokenUserValidationRule{
							{
								Expression: "user.username",
								Message:    "invalid",
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.oidcProviders[0].userValidationRules[0].expression",
		},
		"username expression uses claims.email without claims.email_verified": {
			spec: configv1.AuthenticationSpec{
				Type: "OIDC",
				OIDCProviders: []configv1.OIDCProvider{
					{
						ClaimMappings: configv1.TokenClaimMappings{
							Username: configv1.UsernameClaimMapping{
								Expression: "claims.email",
							},
						},
					},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.oidcProviders[0].claimMappings.username.expression",
		},
	}

	for tcName, tc := range errorCases {
		errs := validateAuthenticationSpec(context.TODO(), tc.spec, &celStore{
			compiler:       authenticationcel.NewDefaultCompiler(),
			compilingGroup: new(singleflight.Group),
			compiledStore:  lru.New(100),
			timerFactory:   &excessiveCompileTimerFactory{},
		})
		if (len(errs) > 0) != (len(tc.errorType) != 0) {
			t.Errorf("'%s': expected failure: %t, got: %t", tcName, len(tc.errorType) != 0, len(errs) > 0)
		}

		for _, e := range errs {
			if e.Type != tc.errorType {
				t.Errorf("'%s': expected errors of type '%s', got %v:", tcName, tc.errorType, e)
			}

			if e.Field != tc.errorField {
				t.Errorf("'%s': expected errors in field '%s', got %v:", tcName, tc.errorField, e)
			}
		}
	}
}

func TestSucceedValidateAuthenticationSpec(t *testing.T) {
	successCases := map[string]configv1.AuthenticationSpec{
		"integrated oauth authn type": {
			Type: "IntegratedOAuth",
		},
		"_none_ authn type": {
			Type: "None",
		},
		"empty authn type": {
			Type: "",
		},
		"integrated oauth + oauth metadata": {
			OAuthMetadata: configv1.ConfigMapNameReference{
				Name: "configmapwithmetadata",
			},
		},
		"webhook set": {
			WebhookTokenAuthenticators: []configv1.DeprecatedWebhookTokenAuthenticator{
				{KubeConfig: configv1.SecretNameReference{Name: "wheniwaslittleiwantedtobecomeawebhook"}},
			},
		},
		"some webhooks": {
			WebhookTokenAuthenticators: []configv1.DeprecatedWebhookTokenAuthenticator{
				{KubeConfig: configv1.SecretNameReference{Name: "whatacoolnameforasecret"}},
				{KubeConfig: configv1.SecretNameReference{Name: "whatacoolnameforasecret2"}},
				{KubeConfig: configv1.SecretNameReference{Name: "thisalsoisacoolname"}},
				{KubeConfig: configv1.SecretNameReference{Name: "letsnotoverdoit"}},
			},
		},
		"all fields set": {
			Type: "IntegratedOAuth",
			OAuthMetadata: configv1.ConfigMapNameReference{
				Name: "suchname",
			},
			WebhookTokenAuthenticators: []configv1.DeprecatedWebhookTokenAuthenticator{
				{KubeConfig: configv1.SecretNameReference{Name: "thisisawebhook"}},
				{KubeConfig: configv1.SecretNameReference{Name: "thisisawebhook2"}},
				{KubeConfig: configv1.SecretNameReference{Name: "thisisawebhook33"}},
			},
		},
		"valid uid CEL expression": {
			Type: "OIDC",
			OIDCProviders: []configv1.OIDCProvider{
				{
					ClaimMappings: configv1.TokenClaimMappings{
						UID: &configv1.TokenClaimOrExpressionMapping{
							Expression: "claims.uid",
						},
					},
				},
			},
		},
		"valid Extra CEL expression": {
			Type: "OIDC",
			OIDCProviders: []configv1.OIDCProvider{
				{
					ClaimMappings: configv1.TokenClaimMappings{
						Extra: []configv1.ExtraMapping{
							{
								Key:             "foo/bar",
								ValueExpression: "claims.roles",
							},
						},
					},
				},
			},
		},
		"valid username CEL expression": {
			Type: "OIDC",
			OIDCProviders: []configv1.OIDCProvider{
				{
					ClaimMappings: configv1.TokenClaimMappings{
						Username: configv1.UsernameClaimMapping{
							Expression: "claims.email",
						},
					},
					ClaimValidationRules: []configv1.TokenClaimValidationRule{
						{
							Type: configv1.TokenValidationRuleTypeCEL,
							CEL: configv1.TokenClaimValidationCELRule{
								Expression: "claims.email_verified == true",
								Message:    "email must be verified",
							},
						},
					},
				},
			},
		},
		"valid groups CEL expression": {
			Type: "OIDC",
			OIDCProviders: []configv1.OIDCProvider{
				{
					ClaimMappings: configv1.TokenClaimMappings{
						Groups: configv1.PrefixedClaimMapping{
							TokenClaimMapping: configv1.TokenClaimMapping{
								Expression: "claims.groups",
							},
						},
					},
				},
			},
		},
		"valid claimValidationRule CEL expression": {
			Type: "OIDC",
			OIDCProviders: []configv1.OIDCProvider{
				{
					ClaimValidationRules: []configv1.TokenClaimValidationRule{
						{
							Type: configv1.TokenValidationRuleTypeCEL,
							CEL: configv1.TokenClaimValidationCELRule{
								Expression: "claims.iss == 'https://example.com'",
								Message:    "issuer must be https://example.com",
							},
						},
					},
				},
			},
		},
		"valid userValidationRule CEL expression": {
			Type: "OIDC",
			OIDCProviders: []configv1.OIDCProvider{
				{
					UserValidationRules: []configv1.TokenUserValidationRule{
						{
							Expression: "user.username != ''",
							Message:    "username must not be empty",
						},
					},
				},
			},
		},
		"RequiredClaim type skips CEL validation": {
			Type: "OIDC",
			OIDCProviders: []configv1.OIDCProvider{
				{
					ClaimValidationRules: []configv1.TokenClaimValidationRule{
						{
							Type: configv1.TokenValidationRuleTypeRequiredClaim,
							RequiredClaim: &configv1.TokenRequiredClaim{
								Claim:         "email_verified",
								RequiredValue: "true",
							},
						},
					},
				},
			},
		},
		"username expression uses claims.email with claims.email_verified in username expression": {
			Type: "OIDC",
			OIDCProviders: []configv1.OIDCProvider{
				{
					ClaimMappings: configv1.TokenClaimMappings{
						Username: configv1.UsernameClaimMapping{
							Expression: "claims.email_verified ? claims.email : 'unverified'",
						},
					},
				},
			},
		},
		"username expression uses claims.email with claims.email_verified in claimValidationRule": {
			Type: "OIDC",
			OIDCProviders: []configv1.OIDCProvider{
				{
					ClaimMappings: configv1.TokenClaimMappings{
						Username: configv1.UsernameClaimMapping{
							Expression: "claims.email",
						},
					},
					ClaimValidationRules: []configv1.TokenClaimValidationRule{
						{
							Type: configv1.TokenValidationRuleTypeCEL,
							CEL: configv1.TokenClaimValidationCELRule{
								Expression: "claims.email_verified == true",
								Message:    "email must be verified",
							},
						},
					},
				},
			},
		},
		"username expression uses claims.email with claims.email_verified in extra": {
			Type: "OIDC",
			OIDCProviders: []configv1.OIDCProvider{
				{
					ClaimMappings: configv1.TokenClaimMappings{
						Username: configv1.UsernameClaimMapping{
							Expression: "claims.email",
						},
						Extra: []configv1.ExtraMapping{
							{
								Key:             "example.com/email-verified",
								ValueExpression: "claims.email_verified ? 'true' : 'false'",
							},
						},
					},
				},
			},
		},
	}

	for tcName, s := range successCases {
		errs := validateAuthenticationSpec(context.TODO(), s, &celStore{
			compiler:       authenticationcel.NewDefaultCompiler(),
			compilingGroup: new(singleflight.Group),
			compiledStore:  lru.New(100),
			timerFactory:   &excessiveCompileTimerFactory{},
		})
		if len(errs) != 0 {
			t.Errorf("'%s': expected success, but failed: %v", tcName, errs.ToAggregate().Error())
		}
	}
}

func TestFailValidateAuthenticationStatus(t *testing.T) {
	errorCases := map[string]struct {
		status     configv1.AuthenticationStatus
		errorType  field.ErrorType
		errorField string
	}{
		"wrong reference name": {
			status: configv1.AuthenticationStatus{
				IntegratedOAuthMetadata: configv1.ConfigMapNameReference{
					Name: "something_wrong",
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "status.integratedOAuthMetadata.name",
		},
	}

	for tcName, tc := range errorCases {
		errs := validateAuthenticationStatus(tc.status)
		if len(errs) == 0 {
			t.Errorf("'%s': should have failed but did not", tcName)
		}

		for _, e := range errs {
			if e.Type != tc.errorType {
				t.Errorf("'%s': expected errors of type '%s', got %v:", tcName, tc.errorType, e)
			}

			if e.Field != tc.errorField {
				t.Errorf("'%s': expected errors in field '%s', got %v:", tcName, tc.errorField, e)
			}
		}
	}
}

func TestSucceedValidateAuthenticationStatus(t *testing.T) {
	successCases := map[string]configv1.AuthenticationStatus{
		"basic case": {
			IntegratedOAuthMetadata: configv1.ConfigMapNameReference{
				Name: "hey-there",
			},
		},
		"empty reference": {
			IntegratedOAuthMetadata: configv1.ConfigMapNameReference{
				Name: "",
			},
		},
		"empty status": {},
	}

	for tcName, s := range successCases {
		errs := validateAuthenticationStatus(s)
		if len(errs) != 0 {
			t.Errorf("'%s': expected success, but failed: %v", tcName, errs.ToAggregate().Error())
		}
	}
}

func TestCompileExpression(t *testing.T) {
	type testcase struct {
		name       string
		cel        func() *celStore
		ctx        func() context.Context
		shouldErr  bool
		shouldWarn bool
		expression authenticationcel.ExpressionAccessor
		compileFn  func(*celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error)
	}

	testcases := []testcase{
		{
			name: "does not return a warning when excessive compilation timer is not triggered",
			cel: func() *celStore {
				return &celStore{
					compiler: &mockCompiler{
						err: nil,
					},
					compilingGroup: new(singleflight.Group),
					compiledStore:  lru.New(1),
					timerFactory: &mockTimerFactory{
						trigger: false,
					},
				}
			},
			ctx:        func() context.Context { return context.TODO() },
			expression: &authenticationcel.ClaimMappingExpression{Expression: `["foo", "bar"].exists(x, x == "foo")`},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				return s.compiler.CompileClaimsExpression
			},
		},
		{
			name: "does not return a warning when excessive compilation timer is not triggered (user expression)",
			cel: func() *celStore {
				return &celStore{
					compiler: &mockCompiler{
						err: nil,
					},
					compilingGroup: new(singleflight.Group),
					compiledStore:  lru.New(1),
					timerFactory: &mockTimerFactory{
						trigger: false,
					},
				}
			},
			ctx:        func() context.Context { return context.TODO() },
			expression: &authenticationcel.UserValidationCondition{Expression: `user.username != ""`},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				return s.compiler.CompileUserExpression
			},
		},
		{
			name: "returns a warning when excessive compilation timer is triggered",
			cel: func() *celStore {
				return &celStore{
					compiler: &mockCompiler{
						err: nil,
					},
					compilingGroup: new(singleflight.Group),
					compiledStore:  lru.New(1),
					timerFactory: &mockTimerFactory{
						trigger: true,
					},
				}
			},
			ctx:        func() context.Context { return context.TODO() },
			shouldWarn: true,
			expression: &authenticationcel.ClaimMappingExpression{Expression: `["foo", "bar"].exists(x, x == "foo")`},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				return s.compiler.CompileClaimsExpression
			},
		},
		{
			name: "returns a warning when excessive compilation timer is triggered (user expression)",
			cel: func() *celStore {
				return &celStore{
					compiler: &mockCompiler{
						err: nil,
					},
					compilingGroup: new(singleflight.Group),
					compiledStore:  lru.New(1),
					timerFactory: &mockTimerFactory{
						trigger: true,
					},
				}
			},
			ctx:        func() context.Context { return context.TODO() },
			shouldWarn: true,
			expression: &authenticationcel.UserValidationCondition{Expression: `user.username != ""`},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				return s.compiler.CompileUserExpression
			},
		},
		{
			name: "still returns error if excessive compilation timer is triggered and errors out",
			cel: func() *celStore {
				return &celStore{
					compiler: &mockCompiler{
						err: errors.New("boom"),
					},
					compilingGroup: new(singleflight.Group),
					compiledStore:  lru.New(1),
					timerFactory: &mockTimerFactory{
						trigger: true,
					},
				}
			},
			ctx:        func() context.Context { return context.TODO() },
			shouldWarn: true,
			shouldErr:  true,
			expression: &authenticationcel.ClaimMappingExpression{Expression: `["foo", "bar"].exists(x, x == "foo")`},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				return s.compiler.CompileClaimsExpression
			},
		},
		{
			name: "still returns error if excessive compilation timer is triggered and errors out (user expression)",
			cel: func() *celStore {
				return &celStore{
					compiler: &mockCompiler{
						err: errors.New("boom"),
					},
					compilingGroup: new(singleflight.Group),
					compiledStore:  lru.New(1),
					timerFactory: &mockTimerFactory{
						trigger: true,
					},
				}
			},
			ctx:        func() context.Context { return context.TODO() },
			shouldWarn: true,
			shouldErr:  true,
			expression: &authenticationcel.UserValidationCondition{Expression: `user.username != ""`},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				return s.compiler.CompileUserExpression
			},
		},
		{
			name: "returns an error if the context has been canceled",
			cel: func() *celStore {
				return &celStore{
					compiler: &mockCompiler{
						err: nil,
					},
					compilingGroup: new(singleflight.Group),
					compiledStore:  lru.New(1),
					timerFactory: &mockTimerFactory{
						trigger: false,
					},
				}
			},
			ctx: func() context.Context {
				ctx, cancel := context.WithCancel(context.TODO())
				cancel()
				return ctx
			},
			shouldErr:  true,
			expression: &authenticationcel.ClaimMappingExpression{Expression: `["foo", "bar"].exists(x, x == "foo")`},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				return s.compiler.CompileClaimsExpression
			},
		},
		{
			name: "returns an error if the context has been canceled (user expression)",
			cel: func() *celStore {
				return &celStore{
					compiler: &mockCompiler{
						err: nil,
					},
					compilingGroup: new(singleflight.Group),
					compiledStore:  lru.New(1),
					timerFactory: &mockTimerFactory{
						trigger: false,
					},
				}
			},
			ctx: func() context.Context {
				ctx, cancel := context.WithCancel(context.TODO())
				cancel()
				return ctx
			},
			shouldErr:  true,
			expression: &authenticationcel.UserValidationCondition{Expression: `user.username != ""`},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				return s.compiler.CompileUserExpression
			},
		},
		{
			name: "returns already compiled expression results if the expression has been compiled before",
			cel: func() *celStore {
				expression := &authenticationcel.ClaimMappingExpression{Expression: `["foo", "bar"].exists(x, x == "foo")`}
				compiledLRU := lru.New(1)
				res := celCompileResult{
					err: errors.New("boom"),
				}
				compiledLRU.Add(fmt.Sprintf("%T:%s", expression, expression.Expression), res)

				return &celStore{
					compiler:       &mockCompiler{},
					compilingGroup: new(singleflight.Group),
					compiledStore:  compiledLRU,
					timerFactory: &mockTimerFactory{
						trigger: false,
					},
				}
			},
			ctx:        func() context.Context { return context.TODO() },
			shouldErr:  true,
			shouldWarn: false,
			expression: &authenticationcel.ClaimMappingExpression{Expression: `["foo", "bar"].exists(x, x == "foo")`},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				return s.compiler.CompileClaimsExpression
			},
		},
		{
			name: "returns already compiled expression results if the expression has been compiled before (user expression)",
			cel: func() *celStore {
				expression := &authenticationcel.UserValidationCondition{Expression: `user.username != ""`}
				compiledLRU := lru.New(1)
				res := celCompileResult{
					err: errors.New("boom"),
				}
				compiledLRU.Add(fmt.Sprintf("%T:%s", expression, expression.Expression), res)

				return &celStore{
					compiler:       &mockCompiler{},
					compilingGroup: new(singleflight.Group),
					compiledStore:  compiledLRU,
					timerFactory: &mockTimerFactory{
						trigger: false,
					},
				}
			},
			ctx:        func() context.Context { return context.TODO() },
			shouldErr:  true,
			shouldWarn: false,
			expression: &authenticationcel.UserValidationCondition{Expression: `user.username != ""`},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				return s.compiler.CompileUserExpression
			},
		},
		{
			name: "failed compilation with one accessor type does not poison cache for same expression with different accessor type",
			cel: func() *celStore {
				return defaultCelStore()
			},
			ctx:        func() context.Context { return context.TODO() },
			shouldErr:  false,
			expression: &authenticationcel.ClaimMappingExpression{Expression: "claims.email"},
			compileFn: func(s *celStore) func(authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
				// First compile as UserValidationCondition to poison the cache with a failure
				// because claims.* is not in scope for user expressions.
				// Then return CompileClaimsExpression which should succeed for the same expression.
				userAccessor := &authenticationcel.UserValidationCondition{Expression: "claims.email"}
				_, _ = compileExpression(context.TODO(), s, &costRecorder{}, field.NewPath("user"), userAccessor, s.compiler.CompileUserExpression)
				return s.compiler.CompileClaimsExpression
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			warningRecorder := &mockWarningRecorder{}
			ctx := warning.WithWarningRecorder(tc.ctx(), warningRecorder)
			celStore := tc.cel()
			_, err := compileExpression(ctx, celStore, &costRecorder{}, field.NewPath("^"), tc.expression, tc.compileFn(celStore))
			if tc.shouldErr != (len(err) > 0) {
				t.Fatalf("error expectation does not match actual. expected? %v . received: %v", tc.shouldErr, err)
			}

			if tc.shouldWarn != (len(warningRecorder.warnings) > 0) {
				t.Fatalf("warning expectation does not match actual. expected? %v . received: %v", tc.shouldWarn, warningRecorder.warnings)
			}
		})
	}
}

type mockCompiler struct {
	receiver    chan error
	err         error
	useDelegate bool
	delegate    authenticationcel.Compiler
	called      int
}

func (mc *mockCompiler) CompileClaimsExpression(expressionAccessor authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
	mc.called += 1
	if mc.receiver != nil {
		err := <-mc.receiver
		return authenticationcel.CompilationResult{}, err
	}
	return authenticationcel.CompilationResult{}, mc.err
}

func (mc *mockCompiler) CompileUserExpression(expressionAccessor authenticationcel.ExpressionAccessor) (authenticationcel.CompilationResult, error) {
	mc.called += 1
	if mc.receiver != nil {
		err := <-mc.receiver
		return authenticationcel.CompilationResult{}, err
	}
	return authenticationcel.CompilationResult{}, mc.err
}

type mockTimerFactory struct {
	trigger bool
}

func (mct *mockTimerFactory) Timer(_ time.Duration, do func()) timer {
	if mct.trigger {
		do()
		return &mockTimer{done: true}
	}

	return &mockTimer{done: false}
}

type mockTimer struct {
	done bool
}

func (mt *mockTimer) Stop() bool {
	return mt.done
}

type mockCompiledExpressionStore struct {
	adds     int
	gets     int
	delegate *lru.Cache
}

func (mces *mockCompiledExpressionStore) Add(key lru.Key, value interface{}) {
	mces.adds += 1

	if mces.delegate != nil {
		mces.delegate.Add(key, value)
	}
}

func (mces *mockCompiledExpressionStore) Get(key lru.Key) (interface{}, bool) {
	mces.gets += 1

	if mces.delegate != nil {
		return mces.delegate.Get(key)
	}

	return nil, false
}

// signallingSingleFlightGroup is an implementation
// of the singleFlightDoer interface that is used to
// exercise the behavior of a singleflight.Group deduplicating
// work when multiple goroutines attempt to compile the same
// CEL expression
type signallingSingleFlightGroup struct {
	singleflight.Group

	// ready is a channel in which the signallingSingleFlightGroup
	// can send a signal that it has started work for a key-func pair
	ready chan struct{}
}

func (ssfg *signallingSingleFlightGroup) Do(key string, fn func() (any, error)) (any, error, bool) {
	c := ssfg.DoChan(key, fn)
	ssfg.ready <- struct{}{}
	res := <-c
	return res.Val, res.Err, res.Shared
}

// TestCompileExpressionDeduplicatesWork ensures
// that we only do work to compile a CEL expression across
// goroutines once.
// We do this by:
// 1. Mocking the compiler such that it blocks
// until it receives a signal on a channel.
// 2. Mocking the singleFlightDoer with a singleFlightDoer that sends
// a signal on a channel when work has been started on the singleflight.Group
// 3. Spinning N goroutines to compile the same expression, where N is an arbitary number of duplicates
// 4. Waiting until we have received a signal from each spun goroutine that it has started compilation
// of the CEL expression
// 5. Sending an error on the channel the mock compiler is blocking on
//
// This ensures that all spun goroutines are actively "compiling" the CEL
// expression before we tell the compiler to complete compilation. This means
// the first goroutine to actually call the compiler.CompileClaimsExpression method
// will hog the singleflight.Group and the rest of the goroutines will wait for
// it's results.
func TestCompileExpressionDeduplicatesWork(t *testing.T) {
	// [1] Mock the compiler and have it block until
	// we send an error on a channel
	receiver := make(chan error)
	mCompiler := &mockCompiler{
		receiver: receiver,
	}

	mCompiledExpressionStore := &mockCompiledExpressionStore{
		delegate: lru.New(1),
	}

	// [2] Mock the singleFlightDoer and send a signal on
	// a channel when work has been started on the singleflight.Group
	ready := make(chan struct{})
	ssfg := &signallingSingleFlightGroup{
		ready: ready,
	}

	cel := &celStore{
		compiler:       mCompiler,
		compilingGroup: ssfg,
		compiledStore:  mCompiledExpressionStore,
		timerFactory: &mockTimerFactory{
			trigger: false,
		},
	}

	expression := &authenticationcel.ClaimMappingExpression{
		Expression: `["foo", "bar"].exists(x, x == "foo")`,
	}

	results := make(chan field.ErrorList)
	fieldPath := field.NewPath("^")

	// [3] Spin N goroutines to compile the same expression
	duplicates := 2
	for range duplicates {
		go func() {
			_, errs := compileExpression(context.TODO(), cel, &costRecorder{}, fieldPath, expression, cel.compiler.CompileClaimsExpression)
			results <- errs
		}()
	}

	// [4] Wait for N goroutines to be reported as having
	// started work via the singleflight.Group
	for range duplicates {
		<-ready
	}

	// [5] Send an error on the channel the mock compiler is blocking on
	expectedErr := errors.New("boom")
	receiver <- expectedErr

	expectedFieldError := field.ErrorList{field.Invalid(fieldPath, expression.Expression, expectedErr.Error())}

	// singleflight.Group will return the results of the first call to all
	// goroutines waiting for the work to be finished.
	// Check to ensure all goroutines reported the same results.
	for range duplicates {
		result := <-results
		if result.ToAggregate().Error() != expectedFieldError.ToAggregate().Error() {
			t.Fatalf("expected all results to have error %v but got a result with a different error of %v", expectedFieldError.ToAggregate(), result.ToAggregate())
		}
	}

	// The mock compiler should have only been called a single time because only the
	// first call to the singleflight.Group for the CEL expression should
	// have resulted in an actual call to the compiler.
	if mCompiler.called == 0 {
		t.Fatal("expected compiler to be called, but it was not")
	}

	if mCompiler.called > 1 {
		t.Fatalf("expected compiler to be called once, but it was called %d times", mCompiler.called)
	}

	// The mock cache should have only been called a single time because only the
	// first call to the singleflight.Group for the CEL expression should
	// have resulted in an actual check to see if the expression has previously been compiled.
	if mCompiledExpressionStore.gets == 0 {
		t.Fatal("expected cache to have been hit one time, but was never hit")
	}

	if mCompiledExpressionStore.gets > 1 {
		t.Fatalf("expected cache to have been hit one time, but was hit %d times", mCompiledExpressionStore.gets)
	}
}

func TestValidAuthenticationSpecWithExcessivelyLongCELExpressionCompileTime(t *testing.T) {
	authn := configv1.AuthenticationSpec{
		Type: "OIDC",
		OIDCProviders: []configv1.OIDCProvider{
			{
				ClaimMappings: configv1.TokenClaimMappings{
					UID: &configv1.TokenClaimOrExpressionMapping{
						Expression: "claims.foo",
					},
				},
			},
		},
	}

	warningRecorder := &mockWarningRecorder{}
	ctx := warning.WithWarningRecorder(context.TODO(), warningRecorder)

	errs := validateAuthenticationSpec(ctx, authn, &celStore{
		compiler:       &mockCompiler{},
		compilingGroup: new(singleflight.Group),
		compiledStore:  lru.New(1),
		timerFactory: &mockTimerFactory{
			trigger: true,
		},
	})

	if len(errs) > 0 {
		t.Fatalf("should not have received any errors, but got: %v", errs.ToAggregate())
	}

	if len(warningRecorder.warnings) != 1 {
		t.Fatalf("expected to receive one warning about excessively long cel compilation time, got: %v", warningRecorder.warnings)
	}

	if !strings.Contains(warningRecorder.warnings[0], "took excessively long to compile") {
		t.Fatalf("expected warning to mention excessively long compile time but instead got: %s", warningRecorder.warnings[0])
	}
}

func TestValidAuthenticationSpecWithExcessiveCELExpressionRuntimeCost(t *testing.T) {
	authn := configv1.AuthenticationSpec{
		Type: "OIDC",
		OIDCProviders: []configv1.OIDCProvider{
			{
				ClaimMappings: configv1.TokenClaimMappings{
					UID: &configv1.TokenClaimOrExpressionMapping{
						Expression: "claims.map(x, x+x)",
					},
				},
			},
		},
	}

	warningRecorder := &mockWarningRecorder{}
	ctx := warning.WithWarningRecorder(context.TODO(), warningRecorder)

	errs := validateAuthenticationSpec(ctx, authn, &celStore{
		compiler:       authenticationcel.NewDefaultCompiler(),
		compilingGroup: new(singleflight.Group),
		compiledStore:  lru.New(1),
		timerFactory:   &excessiveCompileTimerFactory{},
		sizeEstimator: &fixedSizeEstimator{
			size: 100000, // enough to blow the whole resource cost warning threshold
		},
	})

	if len(errs) > 0 {
		t.Fatalf("should not have received any errors, but got: %v", errs.ToAggregate())
	}

	if len(warningRecorder.warnings) != 1 {
		t.Fatalf("expected to receive one warning about excessive runtime cost, got: %v", warningRecorder.warnings)
	}

	if !strings.Contains(warningRecorder.warnings[0], "runtime cost of all CEL expressions exceeds") {
		t.Fatalf("expected warning to mention excessive runtime cost but instead got: %s", warningRecorder.warnings[0])
	}
}

func TestValidAuthenticationSpecNoExcessiveCELExpressionRuntimeCostWithSimpleExpressions(t *testing.T) {
	authn := configv1.AuthenticationSpec{
		Type: "OIDC",
		OIDCProviders: []configv1.OIDCProvider{
			{
				ClaimMappings: configv1.TokenClaimMappings{
					UID: &configv1.TokenClaimOrExpressionMapping{
						Expression: "claims.sub",
					},
					Extra: []configv1.ExtraMapping{
						{
							Key:             "test.io/role",
							ValueExpression: "claims.role",
						},
						{
							Key:             "test.io/country",
							ValueExpression: "claims.country",
						},
						// A bit more complex expression
						{
							Key:             "test.io/org",
							ValueExpression: "claims.email.endsWith('@test.io') ? 'testOrg' : 'acquiredOrg'",
						},
					},
				},
			},
		},
	}

	warningRecorder := &mockWarningRecorder{}
	ctx := warning.WithWarningRecorder(context.TODO(), warningRecorder)

	errs := validateAuthenticationSpec(ctx, authn, defaultCelStore())

	if len(errs) > 0 {
		t.Fatalf("should not have received any errors, but got: %v", errs.ToAggregate())
	}

	if len(warningRecorder.warnings) > 0 {
		t.Fatalf("should not have received any warnings, but got: %v", warningRecorder.warnings)
	}
}

type mockWarningRecorder struct {
	warnings []string
}

func (mwr *mockWarningRecorder) AddWarning(agent, text string) {
	mwr.warnings = append(mwr.warnings, text)
}
