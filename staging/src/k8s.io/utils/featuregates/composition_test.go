package featuregates

import "testing"

func TestAddingSameFeatureGateTwice(t *testing.T) {
	clientGoFeatureGateOne := NewEnvVarFeatureGate("client-go/one").EnableByDefault().Beta().ToFeatureGateOrDie()
	clientGoFeatureGateTwo := NewEnvVarFeatureGate("client-go/two").Alpha().ToFeatureGateOrDie()
	clientGoFeatureSet := NewSimpleFeatureSet()
	clientGoFeatureSet.AddFeatureGatesOrDie(clientGoFeatureGateOne, clientGoFeatureGateTwo)

	genericAPIServerFeatureGateApple := NewFeatureGate("apiserver/apple").EnableByDefault().Beta().ToFeatureGateOrDie()
	genericAPIServerFeatureGateBanana := NewFeatureGate("apiserver/banana").Alpha().ToFeatureGateOrDie()
	genericAPIServerFeatureSet := NewSimpleFeatureSet()
	genericAPIServerFeatureSet.AddFeatureGatesOrDie(genericAPIServerFeatureGateApple, genericAPIServerFeatureGateBanana)
	genericAPIServerFeatureSet.AddFeatureSetsOrDie(clientGoFeatureSet)

	oneReference := clientGoFeatureGateOne
	oneDuplicate := NewEnvVarFeatureGate("client-go/one").EnableByDefault().Beta().ToFeatureGateOrDie()
	kubeControllerManagerFeatureGateAlpha := NewFeatureGate("kube-controller-manager/alpha").EnableByDefault().Beta().ToFeatureGateOrDie()
	kubeControllerManagerFeatureGateBravo := NewFeatureGate("kube-controller-manager/bravo").Alpha().ToFeatureGateOrDie()
	kubeControllerManagerFeatureSet := NewSimpleFeatureSet()
	kubeControllerManagerFeatureSet.AddFeatureGatesOrDie(kubeControllerManagerFeatureGateAlpha, kubeControllerManagerFeatureGateBravo)
	kubeControllerManagerFeatureSet.AddFeatureSetsOrDie(genericAPIServerFeatureSet)

	if err := kubeControllerManagerFeatureSet.AddFeatureGates(oneReference); err != nil {
		t.Errorf("failed to add reference to original gate, but this is ok: %v", err)
	}
	if err := kubeControllerManagerFeatureSet.AddFeatureGates(oneDuplicate); err == nil {
		t.Errorf("must fail to add duplicate of featuregate, but didn't")
	}
	if err := kubeControllerManagerFeatureSet.AddFeatureSets(clientGoFeatureSet); err != nil {
		t.Errorf("failed to add featureset twice, but we should be able to : %v", err)
	}
}
