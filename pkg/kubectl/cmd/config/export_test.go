/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/kubernetes/fake"
)

var configYml = `apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUM1ekNDQWMrZ0F3SUJBZ0lCQVRBTkJna3Foa2lHOXcwQkFRc0ZBREFWTVJNd0VRWURWUVFERXdwdGFXNXAKYTNWaVpVTkJNQjRYRFRFNE1EWXhNREE1TURVeU1Wb1hEVEk0TURZd056QTVNRFV5TVZvd0ZURVRNQkVHQTFVRQpBeE1LYldsdWFXdDFZbVZEUVRDQ0FTSXdEUVlKS29aSWh2Y05BUUVCQlFBRGdnRVBBRENDQVFvQ2dnRUJBTytiCmVwUVB5VjVtZVRaNWMwNzBTSlM4cjhsOEZKY0pKT1NQM01KeFJiZjZHdGhqRzNFU3RUSG5XdDdYeWwyUHdJZ2sKZFg4azdNdHlNemxySys0RTI5VWtMa2drREU2MHU4SVE4VE9YcFhRYU1JUkM1czdmT21VWjY1MnBBcmQ1Uy92TQpCbXJWdEJubFZzOXNpYnB5ZGREWjRVU2ZtY1g1RnR4Z1N3d2V1aWU4REVzR1FUWE9OVVBpM0RqemcyQ2lDZXUwCmVsK3dPa0tBUU10aXdtVEc4ZkZkTWpPT2hDZ0N3MVduS0JoT01FWG5qYm9xRjVnRkx4eWIyQWlNck90QVpxaUIKdWp0RUtQT1MzWUl1NWNveGgzdmZsWHdVcnNMYW8vVktnejFUd0NsVTJLSzZIL3NjME8rSCtSbmc0WEUwRTZNaQp6bEIxL3VGQ01PMUxMdkFjdFdjQ0F3RUFBYU5DTUVBd0RnWURWUjBQQVFIL0JBUURBZ0trTUIwR0ExVWRKUVFXCk1CUUdDQ3NHQVFVRkJ3TUNCZ2dyQmdFRkJRY0RBVEFQQmdOVkhSTUJBZjhFQlRBREFRSC9NQTBHQ1NxR1NJYjMKRFFFQkN3VUFBNElCQVFCR3hFb1FjNXdsUGxXL3d0YnQvalJtWnRGaXRkTjc2MXBkbm1SOUd1Yk93a2hqclBFdgpseVhPdVFKSk1ESVBvOTBlN1VyWS96OXNoeEdnNkVIOTEwV0hya2gvV2ZsVzBITnVnZ05FdUlsZndGQ0w2d1E3CmJOVjVjOVdWTUF4OUlCK2c3bm85M2RVajVBalBQMGVCeW1XS3VCenc2UU5RRUtGSXhETWVsRExLNC9qV1B0eG0Kc0JEYzk2Lzl4T3lYVEF3TUdkRjVmeUNGVEZURHNvd0hVa05McWtQY01jajJFbVpNY3F4MEMxZnl5aDRsNkg4TQpyQkRzNEIrNDRLTjM2ZERUemgycHluWCt0Q3B0dGZpNlBmdFR5Y0VmVXhwdGFIUU4xcjNuWFZEM0pGaFVoL2FOCnRpeWU3U2tCMkNadWluUFRnSmVkOUFEZkZWR2V5ZjVneGtNYQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg==
    server: https://192.168.99.100:8443
  name: minikube
contexts:
- context:
    cluster: minikube
    namespace: default
    user: b.chen-minikube
  name: b.chen-minikube
current-context: b.chen-minikube
kind: Config
preferences: {}
users:
- name: b.chen-minikube
  user:
    token: eyJhbGciOiJSUzI1NiIsImtpZCI6IiJ9.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImIuY2hlbi10b2tlbi03Z3c2ZyIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJiLmNoZW4iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC51aWQiOiIxY2EyNzA1OS03Nzk3LTExZTgtYmYwNC0wODAwMjdlNDNkNTkiLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6ZGVmYXVsdDpiLmNoZW4ifQ.hqthaIxd_M2lBjN1ggNN2q7l4c-vVLmhL-Lz_q2cifi2TIwG8kiXcn5CaBdD5Vc57XAOfYqAreDTl6s0b_-lR7TmvVWfz6hS47oHnYErMKvMUyv530BzkItX6_3atbbA7udVAtWRaeQLXuk2mHHnwXRa8zdXRVMVaYKxXRKz7bQl5p2c0EiCRlFGDBpOifrCbve010eZ-G--FGTblH_jpK40eexx0s99ycsebOYn6XhBcLO2neXAIHq3_i8r9XishSqROsK1J8W_qgab6BD2P5f4yOsRYKkzpUBMpXOkMk6j_ZTKJSF0hQZBY-siYYuxbFVZSgchCy9qI66gJLKgoA
`

var secretJson = `
{
    "apiVersion": "v1",
    "data": {
        "ca.crt": "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUM1ekNDQWMrZ0F3SUJBZ0lCQVRBTkJna3Foa2lHOXcwQkFRc0ZBREFWTVJNd0VRWURWUVFERXdwdGFXNXAKYTNWaVpVTkJNQjRYRFRFNE1EWXhNREE1TURVeU1Wb1hEVEk0TURZd056QTVNRFV5TVZvd0ZURVRNQkVHQTFVRQpBeE1LYldsdWFXdDFZbVZEUVRDQ0FTSXdEUVlKS29aSWh2Y05BUUVCQlFBRGdnRVBBRENDQVFvQ2dnRUJBTytiCmVwUVB5VjVtZVRaNWMwNzBTSlM4cjhsOEZKY0pKT1NQM01KeFJiZjZHdGhqRzNFU3RUSG5XdDdYeWwyUHdJZ2sKZFg4azdNdHlNemxySys0RTI5VWtMa2drREU2MHU4SVE4VE9YcFhRYU1JUkM1czdmT21VWjY1MnBBcmQ1Uy92TQpCbXJWdEJubFZzOXNpYnB5ZGREWjRVU2ZtY1g1RnR4Z1N3d2V1aWU4REVzR1FUWE9OVVBpM0RqemcyQ2lDZXUwCmVsK3dPa0tBUU10aXdtVEc4ZkZkTWpPT2hDZ0N3MVduS0JoT01FWG5qYm9xRjVnRkx4eWIyQWlNck90QVpxaUIKdWp0RUtQT1MzWUl1NWNveGgzdmZsWHdVcnNMYW8vVktnejFUd0NsVTJLSzZIL3NjME8rSCtSbmc0WEUwRTZNaQp6bEIxL3VGQ01PMUxMdkFjdFdjQ0F3RUFBYU5DTUVBd0RnWURWUjBQQVFIL0JBUURBZ0trTUIwR0ExVWRKUVFXCk1CUUdDQ3NHQVFVRkJ3TUNCZ2dyQmdFRkJRY0RBVEFQQmdOVkhSTUJBZjhFQlRBREFRSC9NQTBHQ1NxR1NJYjMKRFFFQkN3VUFBNElCQVFCR3hFb1FjNXdsUGxXL3d0YnQvalJtWnRGaXRkTjc2MXBkbm1SOUd1Yk93a2hqclBFdgpseVhPdVFKSk1ESVBvOTBlN1VyWS96OXNoeEdnNkVIOTEwV0hya2gvV2ZsVzBITnVnZ05FdUlsZndGQ0w2d1E3CmJOVjVjOVdWTUF4OUlCK2c3bm85M2RVajVBalBQMGVCeW1XS3VCenc2UU5RRUtGSXhETWVsRExLNC9qV1B0eG0Kc0JEYzk2Lzl4T3lYVEF3TUdkRjVmeUNGVEZURHNvd0hVa05McWtQY01jajJFbVpNY3F4MEMxZnl5aDRsNkg4TQpyQkRzNEIrNDRLTjM2ZERUemgycHluWCt0Q3B0dGZpNlBmdFR5Y0VmVXhwdGFIUU4xcjNuWFZEM0pGaFVoL2FOCnRpeWU3U2tCMkNadWluUFRnSmVkOUFEZkZWR2V5ZjVneGtNYQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg==",
        "namespace": "ZGVmYXVsdA==",
        "token": "ZXlKaGJHY2lPaUpTVXpJMU5pSXNJbXRwWkNJNklpSjkuZXlKcGMzTWlPaUpyZFdKbGNtNWxkR1Z6TDNObGNuWnBZMlZoWTJOdmRXNTBJaXdpYTNWaVpYSnVaWFJsY3k1cGJ5OXpaWEoyYVdObFlXTmpiM1Z1ZEM5dVlXMWxjM0JoWTJVaU9pSmtaV1poZFd4MElpd2lhM1ZpWlhKdVpYUmxjeTVwYnk5elpYSjJhV05sWVdOamIzVnVkQzl6WldOeVpYUXVibUZ0WlNJNkltSXVZMmhsYmkxMGIydGxiaTAzWjNjMlp5SXNJbXQxWW1WeWJtVjBaWE11YVc4dmMyVnlkbWxqWldGalkyOTFiblF2YzJWeWRtbGpaUzFoWTJOdmRXNTBMbTVoYldVaU9pSmlMbU5vWlc0aUxDSnJkV0psY201bGRHVnpMbWx2TDNObGNuWnBZMlZoWTJOdmRXNTBMM05sY25acFkyVXRZV05qYjNWdWRDNTFhV1FpT2lJeFkyRXlOekExT1MwM056azNMVEV4WlRndFltWXdOQzB3T0RBd01qZGxORE5rTlRraUxDSnpkV0lpT2lKemVYTjBaVzA2YzJWeWRtbGpaV0ZqWTI5MWJuUTZaR1ZtWVhWc2REcGlMbU5vWlc0aWZRLmhxdGhhSXhkX00ybEJqTjFnZ05OMnE3bDRjLXZWTG1oTC1Mel9xMmNpZmkyVEl3RzhraVhjbjVDYUJkRDVWYzU3WEFPZllxQXJlRFRsNnMwYl8tbFI3VG12VldmejZoUzQ3b0huWUVyTUt2TVV5djUzMEJ6a0l0WDZfM2F0YmJBN3VkVkF0V1JhZVFMWHVrMm1ISG53WFJhOHpkWFJWTVZhWUt4WFJLejdiUWw1cDJjMEVpQ1JsRkdEQnBPaWZyQ2J2ZTAxMGVaLUctLUZHVGJsSF9qcEs0MGVleHgwczk5eWNzZWJPWW42WGhCY0xPMm5lWEFJSHEzX2k4cjlYaXNoU3FST3NLMUo4V19xZ2FiNkJEMlA1ZjR5T3NSWUtrenBVQk1wWE9rTWs2al9aVEtKU0YwaFFaQlktc2lZWXV4YkZWWlNnY2hDeTlxSTY2Z0pMS2dvQQ=="
    },
    "kind": "Secret",
    "metadata": {
        "annotations": {
            "kubernetes.io/service-account.name": "b.chen",
            "kubernetes.io/service-account.uid": "1ca27059-7797-11e8-bf04-080027e43d59"
        },
        "creationTimestamp": "2018-06-24T10:12:33Z",
        "name": "b.chen-token-7gw6g",
        "namespace": "default",
        "resourceVersion": "195319",
        "selfLink": "/api/v1/namespaces/default/secrets/b.chen-token-7gw6g",
        "uid": "1ca49865-7797-11e8-bf04-080027e43d59"
    },
    "type": "kubernetes.io/service-account-token"
}
`
var serviceaccountJson = `
{
    "apiVersion": "v1",
    "kind": "ServiceAccount",
    "metadata": {
        "creationTimestamp": "2018-06-24T10:12:33Z",
        "name": "b.chen",
        "namespace": "default",
        "resourceVersion": "195320",
        "selfLink": "/api/v1/namespaces/default/serviceaccounts/b.chen",
        "uid": "1ca27059-7797-11e8-bf04-080027e43d59"
    },
    "secrets": [
        {
            "name": "b.chen-token-7gw6g"
        }
    ]
}
`
var expected = `apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUM1ekNDQWMrZ0F3SUJBZ0lCQVRBTkJna3Foa2lHOXcwQkFRc0ZBREFWTVJNd0VRWURWUVFERXdwdGFXNXAKYTNWaVpVTkJNQjRYRFRFNE1EWXhNREE1TURVeU1Wb1hEVEk0TURZd056QTVNRFV5TVZvd0ZURVRNQkVHQTFVRQpBeE1LYldsdWFXdDFZbVZEUVRDQ0FTSXdEUVlKS29aSWh2Y05BUUVCQlFBRGdnRVBBRENDQVFvQ2dnRUJBTytiCmVwUVB5VjVtZVRaNWMwNzBTSlM4cjhsOEZKY0pKT1NQM01KeFJiZjZHdGhqRzNFU3RUSG5XdDdYeWwyUHdJZ2sKZFg4azdNdHlNemxySys0RTI5VWtMa2drREU2MHU4SVE4VE9YcFhRYU1JUkM1czdmT21VWjY1MnBBcmQ1Uy92TQpCbXJWdEJubFZzOXNpYnB5ZGREWjRVU2ZtY1g1RnR4Z1N3d2V1aWU4REVzR1FUWE9OVVBpM0RqemcyQ2lDZXUwCmVsK3dPa0tBUU10aXdtVEc4ZkZkTWpPT2hDZ0N3MVduS0JoT01FWG5qYm9xRjVnRkx4eWIyQWlNck90QVpxaUIKdWp0RUtQT1MzWUl1NWNveGgzdmZsWHdVcnNMYW8vVktnejFUd0NsVTJLSzZIL3NjME8rSCtSbmc0WEUwRTZNaQp6bEIxL3VGQ01PMUxMdkFjdFdjQ0F3RUFBYU5DTUVBd0RnWURWUjBQQVFIL0JBUURBZ0trTUIwR0ExVWRKUVFXCk1CUUdDQ3NHQVFVRkJ3TUNCZ2dyQmdFRkJRY0RBVEFQQmdOVkhSTUJBZjhFQlRBREFRSC9NQTBHQ1NxR1NJYjMKRFFFQkN3VUFBNElCQVFCR3hFb1FjNXdsUGxXL3d0YnQvalJtWnRGaXRkTjc2MXBkbm1SOUd1Yk93a2hqclBFdgpseVhPdVFKSk1ESVBvOTBlN1VyWS96OXNoeEdnNkVIOTEwV0hya2gvV2ZsVzBITnVnZ05FdUlsZndGQ0w2d1E3CmJOVjVjOVdWTUF4OUlCK2c3bm85M2RVajVBalBQMGVCeW1XS3VCenc2UU5RRUtGSXhETWVsRExLNC9qV1B0eG0Kc0JEYzk2Lzl4T3lYVEF3TUdkRjVmeUNGVEZURHNvd0hVa05McWtQY01jajJFbVpNY3F4MEMxZnl5aDRsNkg4TQpyQkRzNEIrNDRLTjM2ZERUemgycHluWCt0Q3B0dGZpNlBmdFR5Y0VmVXhwdGFIUU4xcjNuWFZEM0pGaFVoL2FOCnRpeWU3U2tCMkNadWluUFRnSmVkOUFEZkZWR2V5ZjVneGtNYQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tCg==
    server: https://192.168.99.100:8443
  name: minikube
contexts:
- context:
    cluster: minikube
    namespace: default
    user: b.chen-minikube
  name: b.chen-minikube
current-context: b.chen-minikube
kind: Config
preferences: {}
users:
- name: b.chen-minikube
  user:
    token: eyJhbGciOiJSUzI1NiIsImtpZCI6IiJ9.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImIuY2hlbi10b2tlbi03Z3c2ZyIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJiLmNoZW4iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC51aWQiOiIxY2EyNzA1OS03Nzk3LTExZTgtYmYwNC0wODAwMjdlNDNkNTkiLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6ZGVmYXVsdDpiLmNoZW4ifQ.hqthaIxd_M2lBjN1ggNN2q7l4c-vVLmhL-Lz_q2cifi2TIwG8kiXcn5CaBdD5Vc57XAOfYqAreDTl6s0b_-lR7TmvVWfz6hS47oHnYErMKvMUyv530BzkItX6_3atbbA7udVAtWRaeQLXuk2mHHnwXRa8zdXRVMVaYKxXRKz7bQl5p2c0EiCRlFGDBpOifrCbve010eZ-G--FGTblH_jpK40eexx0s99ycsebOYn6XhBcLO2neXAIHq3_i8r9XishSqROsK1J8W_qgab6BD2P5f4yOsRYKkzpUBMpXOkMk6j_ZTKJSF0hQZBY-siYYuxbFVZSgchCy9qI66gJLKgoA`

func TestExportKubConfig(t *testing.T) {
	fakeKubeFile, err := ioutil.TempFile(os.TempDir(), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(fakeKubeFile.Name())
	err = ioutil.WriteFile(fakeKubeFile.Name(), []byte(configYml), 0644)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	secret := v1.Secret{}
	err = json.Unmarshal([]byte(secretJson), &secret)
	if err != nil {
		t.Fatalf("could not decode secret: %s", err)
	}

	sa := v1.ServiceAccount{}
	err = json.Unmarshal([]byte(serviceaccountJson), &sa)
	if err != nil {
		t.Fatalf("could not decode ServiceAccount: %s", err)
	}

	clientSet := fake.NewSimpleClientset(&secret, &sa)
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()

	cmd := NewCmdKubeConfigExport(clientSet, streams)

	cmd.Flags().Parse([]string{"--namespace=default", "--serviceaccount=b.chen", "--kubeconf=" + fakeKubeFile.Name()})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("unexpected error executing command: %v, kubectl config export flags: %v", err, fakeKubeFile.Name())
	}

	if err != nil {
		t.Fatalf("error export KubConfig for: %v", err)
	}
	if buf.String() != expected {
		t.Errorf("Failed in TestExportKubConfig expected %v\n but got %v\n", expected, buf.String())
	}

}
