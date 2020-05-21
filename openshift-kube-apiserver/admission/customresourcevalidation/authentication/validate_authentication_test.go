package authentication

import (
	"testing"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestFailValidateAuthenticationSpec(t *testing.T) {
	errorCases := map[string]struct {
		spec       configv1.AuthenticationSpec
		errorType  field.ErrorType
		errorField string
	}{
		"invalid authn type": {
			spec: configv1.AuthenticationSpec{
				Type: "MyCoolOAuthSrv",
			},
			errorType:  field.ErrorTypeNotSupported,
			errorField: "spec.type",
		},
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
				WebhookTokenAuthenticators: []configv1.WebhookTokenAuthenticator{
					{KubeConfig: configv1.SecretNameReference{Name: "this+that"}},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.webhookTokenAuthenticators[0].kubeConfig.name",
		},
		"invalid webhook ref - multiple webhooks": {
			spec: configv1.AuthenticationSpec{
				WebhookTokenAuthenticators: []configv1.WebhookTokenAuthenticator{
					{KubeConfig: configv1.SecretNameReference{Name: "that.now"}},
					{KubeConfig: configv1.SecretNameReference{Name: "this+that"}},
					{KubeConfig: configv1.SecretNameReference{Name: "this.then"}},
				},
			},
			errorType:  field.ErrorTypeInvalid,
			errorField: "spec.webhookTokenAuthenticators[1].kubeConfig.name",
		},
		"empty webhook name": {
			spec: configv1.AuthenticationSpec{
				WebhookTokenAuthenticators: []configv1.WebhookTokenAuthenticator{
					{KubeConfig: configv1.SecretNameReference{Name: ""}},
				},
			},
			errorType:  field.ErrorTypeRequired,
			errorField: "spec.webhookTokenAuthenticators[0].kubeConfig.name",
		},
	}

	for tcName, tc := range errorCases {
		errs := validateAuthenticationSpec(tc.spec)
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
			WebhookTokenAuthenticators: []configv1.WebhookTokenAuthenticator{
				{KubeConfig: configv1.SecretNameReference{Name: "wheniwaslittleiwantedtobecomeawebhook"}},
			},
		},
		"some webhooks": {
			WebhookTokenAuthenticators: []configv1.WebhookTokenAuthenticator{
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
			WebhookTokenAuthenticators: []configv1.WebhookTokenAuthenticator{
				{KubeConfig: configv1.SecretNameReference{Name: "thisisawebhook"}},
				{KubeConfig: configv1.SecretNameReference{Name: "thisisawebhook2"}},
				{KubeConfig: configv1.SecretNameReference{Name: "thisisawebhook33"}},
			},
		},
	}

	for tcName, s := range successCases {
		errs := validateAuthenticationSpec(s)
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
