#!/bin/bash

sed -i '/import (/a\e2eutils "k8s.io/kubernetes/test/e2e/framework/utils"' $1
sed -i '/import (/a\e2econfig "k8s.io/kubernetes/test/e2e/framework/config"' $1

sed -i 's/framework\.ExpectError/e2eutils\.ExpectError/g' $1
sed -i 's/framework\.ExpectNoError/e2eutils\.ExpectNoError/g' $1
sed -i 's/framework\.ExpectEqual/e2eutils\.ExpectEqual/g' $1
sed -i 's/framework\.ExpectNotEqual/e2eutils\.ExpectNotEqual/g' $1
sed -i 's/framework\.ExpectHaveKey/e2eutils\.ExpectHaveKey/g' $1
sed -i 's/framework\.ExpectEmpty/e2eutils\.ExpectEmpty/g' $1
sed -i 's/framework\.ExpectConsistOf/e2eutils\.ExpectConsistOf/g' $1
sed -i 's/framework\.Logf/e2eutils\.Logf/g' $1
sed -i 's/framework\.Failf/e2eutils\.Failf/g' $1
sed -i 's/framework\.Fail/e2eutils\.Fail/g' $1
sed -i 's/framework\.DumpDebugInfo/e2eutils\.DumpDebugInfo/g' $1
sed -i 's/framework\.DumpAllNamespaceInfo/e2eutils\.DumpAllNamespaceInfo/g' $1
sed -i 's/framework\.RunHostCmdWithRetries/e2eutils\.RunHostCmdWithRetries/g' $1
sed -i 's/framework\.RunHostCmdOrDie/e2eutils\.RunHostCmdOrDie/g' $1
sed -i 's/framework\.RunCmd/e2eutils\.RunCmd/g' $1
sed -i 's/framework\.TryKill/e2eutils\.TryKill/g' $1
sed -i 's/framework\.RunHostCmd/e2eutils\.RunHostCmd/g' $1
sed -i 's/framework\.ServiceStartTimeout/e2eutils\.ServiceStartTimeout/g' $1
sed -i 's/framework\.StartCmdAndStreamOutput/e2eutils\.StartCmdAndStreamOutput/g' $1
sed -i 's/framework\.SingleCallTimeout/e2eutils\.SingleCallTimeout/g' $1

sed -i 's/framework\.LoadConfig/e2eutils\.LoadConfig/g' $1
sed -i 's/framework\.RunKubectl/e2eutils\.RunKubectl/g' $1
sed -i 's/framework\.RunKubectlInput/e2eutils\.RunKubectlInput/g' $1
sed -i 's/framework\.APIAddress/e2eutils\.APIAddress/g' $1
sed -i 's/framework\.Poll/e2eutils\.Poll/g' $1
sed -i 's/framework\.WaitForServiceEndpointsNum/e2eutils\.WaitForServiceEndpointsNum/g' $1
sed -i 's/framework\.NewKubectlCommand/e2eutils\.NewKubectlCommand/g' $1
sed -i 's/framework\.ProviderIs/e2eutils\.ProviderIs/g' $1
sed -i 's/framework\.NodeOSDistroIs/e2eutils\.NodeOSDistroIs/g' $1
sed -i 's/framework\.PodStartShortTimeout/e2eutils\.PodStartShortTimeout/g' $1
sed -i 's/framework\.PodStartTimeout/e2eutils\.PodStartTimeout/g' $1
sed -i 's/framework\.PodDeleteTimeout/e2eutils\.PodDeleteTimeout/g' $1
sed -i 's/framework\.DefaultPodDeletionTimeout/e2eutils\.DefaultPodDeletionTimeout/g' $1
sed -i 's/framework\.WaitForAllNodesSchedulable/e2eutils\.WaitForAllNodesSchedulable/g' $1
sed -i 's/framework\.CheckTestingNSDeletedExcept/e2eutils\.CheckTestingNSDeletedExcept/g' $1
sed -i 's/framework\.LookForStringInPodExec/e2eutils\.LookForStringInPodExec/g' $1
sed -i 's/framework\.LookForStringInLog/e2eutils\.LookForStringInLog/g' $1
sed -i 's/framework\.GetControlPlaneAddresses/e2eutils\.GetControlPlaneAddresses/g' $1
sed -i 's/framework\.RunID/e2eutils\.RunID/g' $1
sed -i 's/framework\.PrettyPrintJSON/e2eutils\.PrettyPrintJSON/g' $1
sed -i 's/framework\.AddOrUpdateLabelOnNode/e2eutils\.AddOrUpdateLabelOnNode/g' $1
sed -i 's/framework\.RemoveLabelOffNode/e2eutils\.RemoveLabelOffNode/g' $1
sed -i 's/framework\.ExpectNodeHasLabel/e2eutils\.ExpectNodeHasLabel/g' $1
sed -i 's/framework\.ProvidersWithSSH/e2eutils\.ProvidersWithSSH/g' $1
sed -i 's/framework\.ExpectNodeHasTaint/e2eutils\.ExpectNodeHasTaint/g' $1
sed -i 's/framework\.RestartPodReadyAgainTimeout/e2eutils\.RestartPodReadyAgainTimeout/g' $1
sed -i 's/framework\.ClaimProvisionTimeout/e2eutils\.ClaimProvisionTimeout/g' $1
sed -i 's/framework\.ImagePrePullList/e2eutils\.ImagePrePullList/g' $1

sed -i 's/framework\.ServeHostnameImage/e2eutils\.ServeHostnameImage/g' $1

sed -i 's/framework\.PodClient/e2eutils\.PodClient/g' $1

sed -i 's/framework\.BusyBoxImage/e2eutils\.BusyBoxImage/g' $1

# e2econfig
sed -i 's/framework\.TestContext/e2econfig\.TestContext/g' $1
sed -i 's/framework\.TimeoutContext/e2econfig\.TimeoutContext/g' $1

