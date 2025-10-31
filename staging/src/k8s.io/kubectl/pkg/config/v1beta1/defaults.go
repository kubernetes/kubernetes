package v1beta1

import clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

func SetDefaults_Preference(pref *Preference) {
	if pref.CredentialPluginPolicy == clientcmdapi.PluginPolicyUnspecified {
		pref.CredentialPluginPolicy = clientcmdapi.PluginPolicyAllowAll
	}
}
