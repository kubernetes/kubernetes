# Changelog

## 0.17.0
   * Cleanups
      * Remove old salt configs #8065 (roberthbailey)
      * Kubelet: minor cleanups #8069 (yujuhong)

   * v1beta3
      * update example/walkthrough to v1beta3 #7940 (caesarxuchao)
      * update example/rethinkdb to v1beta3 #7946 (caesarxuchao)
      * verify the v1beta3 yaml files all work; merge the yaml files #7917 (caesarxuchao)
      * update examples/cassandra to api v1beta3 #7258 (caesarxuchao)
      * update service.json in persistent-volume example to v1beta3 #7899 (caesarxuchao)
      * update mysql-wordpress example to use v1beta3 API #7864 (caesarxuchao)
      * Update examples/meteor to use API v1beta3 #7848 (caesarxuchao)
      * update node-selector example to API v1beta3 #7872 (caesarxuchao)
      * update logging-demo to use API v1beta3; modify the way to access Elasticsearch and Kibana services #7824 (caesarxuchao)
      * Convert the skydns rc to use v1beta3 and add a health check to it #7619 (a-robinson)
      * update the hazelcast example to API version v1beta3 #7728 (caesarxuchao)
      * Fix YAML parsing for v1beta3 objects in the kubelet for file/http #7515 (brendandburns)
      * Updated kubectl cluster-info to show v1beta3 addresses #7502 (piosz)

   * Kubelet
      * kubelet: Fix racy kubelet tests. #7980 (yifan-gu)      
      * kubelet/container: Move prober.ContainerCommandRunner to container. #8079 (yifan-gu)
      * Kubelet: set host field in the pending pod status #6127 (yujuhong)
      * Fix the kubelet node watch #6442 (yujuhong)
      * Kubelet: recreate mirror pod if the static pod changes #6607 (yujuhong)
      * Kubelet: record the timestamp correctly in the runtime cache #7749 (yujuhong)
      * Kubelet: wait until container runtime is up #7729 (yujuhong)
      * Kubelet: replace DockerManager with the Runtime interface #7674 (yujuhong)
      * Kubelet: filter out terminated pods in SyncPods #7301 (yujuhong)
      * Kubelet: parallelize cleaning up containers in unwanted pods #7048 (yujuhong)
      * kubelet: Add container runtime option for rkt. #7952 (yifan-gu)
      * kubelet/rkt: Remove build label. #7916 (yifan-gu)
      * kubelet/metrics: Move instrumented_docker.go to dockertools. #7327 (yifan-gu)
      * kubelet/rkt: Add GetPods() for rkt. #7599 (yifan-gu)
      * kubelet/rkt: Add KillPod() and GetPodStatus() for rkt. #7605 (yifan-gu)
      * pkg/kubelet: Fix logging. #4755 (yifan-gu)
      * kubelet: Refactor RunInContainer/ExecInContainer/PortForward. #6491 (yifan-gu)
      * kubelet/DockerManager: Fix returning empty error from GetPodStatus(). #6609 (yifan-gu)
      * kubelet: Move pod infra container image setting to dockertools. #6634 (yifan-gu)
      * kubelet/fake_docker_client: Use self's PID instead of 42 in testing. #6653 (yifan-gu)
      * kubelet/dockertool: Move Getpods() to DockerManager. #6778 (yifan-gu)
      * kubelet/dockertools: Add puller interfaces in the containerManager. #6776 (yifan-gu)
      * kubelet: Introduce PodInfraContainerChanged(). #6608 (yifan-gu)
      * kubelet/container: Replace DockerCache with RuntimeCache. #6795 (yifan-gu)
      * kubelet: Clean up computePodContainerChanges. #6844 (yifan-gu)
      * kubelet: Refactor prober. #7009 (yifan-gu)
      * kubelet/container: Update the runtime interface. #7466 (yifan-gu)
      * kubelet: Refactor isPodRunning() in runonce.go #7477 (yifan-gu)
      * kubelet/rkt: Add basic rkt runtime routines. #7465 (yifan-gu)
      * kubelet/rkt: Add podInfo. #7555 (yifan-gu)
      * kubelet/container: Add GetContainerLogs to runtime interface. #7488 (yifan-gu)
      * kubelet/rkt: Add routines for converting kubelet pod to rkt pod. #7543 (yifan-gu)
      * kubelet/rkt: Add RunPod() for rkt. #7589 (yifan-gu)
      * kubelet/rkt: Add RunInContainer()/ExecInContainer()/PortForward(). #7553 (yifan-gu)
      * kubelet/container: Move ShouldContainerBeRestarted() to runtime. #7613 (yifan-gu)
      * kubelet/rkt: Add SyncPod() to rkt. #7611 (yifan-gu)
      * Kubelet: persist restart count of a container #6794 (yujuhong)
      * kubelet/container: Move pty*.go to container runtime package. #7951 (yifan-gu)
      * kubelet: Add container runtime option for rkt. #7900 (yifan-gu)
      * kubelet/rkt: Add docker prefix to image string. #7803 (yifan-gu)
      * kubelet/rkt: Inject dependencies to rkt. #7849 (yifan-gu)
      * kubelet/rkt: Remove dependencies on rkt.store #7859 (yifan-gu)
      * Kubelet talks securely to apiserver #2387 (erictune)
      * Rename EnvVarSource.FieldPath -> FieldRef and add example #7592 (pmorie)
      * Add containerized option to kubelet binary #7741 (pmorie)
      * Ease building kubelet image #7948 (pmorie)
      * Remove unnecessary bind-mount from dockerized kubelet run #7854 (pmorie)
      * Add ability to dockerize kubelet in local cluster #7798 (pmorie)
      * Create docker image for kubelet #7797 (pmorie)
      * Security context - types, kubelet, admission #7343 (pweil-)
      * Kubelet: Add rkt as a runtime option #7743 (vmarmol)
      * Fix kubelet's docker RunInContainer implementation  #7746 (vishh)

   * AWS
      * AWS: Don't try to copy gce_keys in jenkins e2e job #8018 (justinsb)
      * AWS: Copy some new properties from config-default => config.test #7992 (justinsb)
      * AWS: make it possible to disable minion public ip assignment #7928 (manolitto)
      * update AWS CloudFormation template and cloud-configs #7667 (antoineco)
      * AWS: Fix variable naming that meant not all tokens were written #7736 (justinsb)
      * AWS: Change apiserver to listen on 443 directly, not through nginx #7678 (justinsb)
      * AWS: Improving getting existing VPC and subnet #6606 (gust1n)
      * AWS EBS volume support #5138 (justinsb)


   * Introduce an 'svc' segment for DNS search #8089 (thockin)
   * Adds ability to define a prefix for etcd paths #5707 (kbeecher)
   * Add kubectl log --previous support to view last terminated container log #7973 (dchen1107)
   * Add a flag to disable legacy APIs #8083 (brendandburns)
   * make the dockerkeyring handle mutiple matching credentials #7971 (deads2k)
   * Convert Fluentd to Cloud Logging pod specs to YAML #8078 (satnam6502)
   * Use etcd to allocate PortalIPs instead of in-mem #7704 (smarterclayton)
   * eliminate auth-path #8064 (deads2k)
   * Record failure reasons for image pulling #7981 (yujuhong)
   * Rate limit replica creation #7869 (bprashanth)
   * Upgrade to Kibana 4 for cluster logging #7995 (satnam6502)
   * Added name to kube-dns service #8049 (piosz)
   * Fix validation by moving it into the resource builder. #7919 (brendandburns)
   * Add cache with multiple shards to decrease lock contention #8050 (fgrzadkowski)
   * Delete status from displayable resources #8039 (nak3)
   * Refactor volume interfaces to receive pod instead of ObjectReference #8044 (pmorie)
   * fix kube-down for provider gke #7565 (jlowdermilk)
   * Service port names are required for multi-port #7786 (thockin)
   * Increase disk size for kubernetes master. #8051 (fgrzadkowski)
   * expose: Load input object for increased safety #7774 (kargakis)
   * Improments to conversion methods generator #7896 (wojtek-t)
   * Added displaying external IPs to kubectl cluster-info #7557 (piosz)
   * Add missing Errorf formatting directives #8037 (shawnps)
   * Add startup code to apiserver to migrate etcd keys #7567 (kbeecher)
   * Use error type from docker go-client instead of string #8021 (ddysher)
   * Accurately get hardware cpu count in Vagrantfile. #8024 (BenTheElder)
   * Stop setting a GKE specific version of the kubeconfig file #7921 (roberthbailey)
   * Make the API server deal with HEAD requests via the service proxy #7950 (satnam6502)
   * GlusterFS Critical Bug Resolved - Removing warning in README #7983 (wattsteve)
   * Don't use the first token `uname -n` as the hostname #7967 (yujuhong)
   * Call kube-down in test-teardown for vagrant. #7982 (BenTheElder)
   * defaults_tests: verify defaults when converting to an API object #6235 (yujuhong)
   * Use the full hostname for mirror pod name. #7910 (yujuhong)
   * Removes RunPod in the Runtime interface #7657 (yujuhong)
   * Clean up dockertools/manager.go and add more unit tests #7533 (yujuhong)
   * Adapt pod killing and cleanup for generic container runtime #7525 (yujuhong)
   * Fix pod filtering in replication controller #7198 (yujuhong)
   * Print container statuses in `kubectl get pods` #7116 (yujuhong)
   * Prioritize deleting the non-running pods when reducing replicas #6992 (yujuhong)
   * Fix locking issue in pod manager #6872 (yujuhong)
   * Limit the number of concurrent tests in integration.go #6655 (yujuhong)
   * Fix typos in different config comments #7931 (pmorie)
   * Update cAdvisor dependency. #7929 (vmarmol)
   * Ubuntu-distro: deprecate & merge ubuntu single node work to ubuntu cluster node stuff #5498 (resouer)
   * Add control variables to Jenkins E2E script #7935 (saad-ali)
   * Check node status as part of validate-cluster.sh. #7932 (fabioy)
   * Add old endpoint cleanup function #7821 (lavalamp)
   * Support recovery from in the middle of a rename. #7620 (brendandburns)
   * Update Exec and Portforward client to use pod subresource #7715 (csrwng)
   * Added NFS to PV structs #7564 (markturansky)
   * Fix environment variable error in Vagrant docs #7904 (posita)
   * Adds a simple release-note builder that scrapes the Github API for recent PRs #7616 (brendandburns)
   * Scheduler ignores nodes that are in a bad state #7668 (bprashanth)
   * Set GOMAXPROCS for etcd #7863 (fgrzadkowski)
   * Auto-generated conversion methods calling one another #7556 (wojtek-t)
   * Bring up a kuberenetes cluster using coreos image as worker nodes #7445 (dchen1107)
   * Godep: Add godep for rkt. #7410 (yifan-gu)
   * Add volumeGetter to rkt. #7870 (yifan-gu)
   * Update cAdvisor dependency. #7897 (vmarmol)
   * DNS: expose 53/TCP #7822 (thockin)
   * Set NodeReady=False when docker is dead #7763 (wojtek-t)
   * Ignore latency metrics for events #7857 (fgrzadkowski)
   * SecurityContext admission clean up #7792 (pweil-)
   * Support manually-created and generated conversion functions #7832 (wojtek-t)
   * Add latency metrics for etcd operations #7833 (fgrzadkowski)
   * Update errors_test.go #7885 (hurf)
   * Change signature of container runtime PullImage to allow pull w/ secret #7861 (pmorie)
   * Fix bug in Service documentation: incorrect location of "selector" in JSON #7873 (bkeroackdsc)
   * Fix controller-manager manifest for providers that don't specify CLUSTER_IP_RANGE #7876 (cjcullen)
   * Fix controller unittests #7867 (bprashanth)
   * Enable GCM and GCL instead of InfluxDB on GCE #7751 (saad-ali)
   * Remove restriction that cluster-cidr be a class-b #7862 (cjcullen)
   * Fix OpenShift example #7591 (derekwaynecarr)
   * API Server - pass path name in context of create request for subresource #7718 (csrwng)
   * Rolling Updates: Add support for --rollback. #7575 (brendandburns)
   * Update to container-vm-v20150505 (Also updates GCE to Docker 1.6) #7820 (zmerlynn)
   * Fix metric label #7830 (rhcarvalho)
   * Fix v1beta1 typos in v1beta2 conversions #7838 (pmorie)
   * skydns: use the etcd-2.x native syntax, enable IANA attributed ports. #7764 (AntonioMeireles)
   * Added port 6443 to kube-proxy default IP address for api-server #7794 (markllama)
   * Added client header info for authentication doc. #7834 (ashcrow)
   * Clean up safe_format_and_mount spam in the startup logs #7827 (zmerlynn)
   * Set allocate_node_cidrs to be blank by default. #7829 (roberthbailey)
   * Fix sync problems in #5246 #7799 (cjcullen)
   * Fix event doc link #7823 (saad-ali)
   * Cobra update and bash completions fix #7776 (eparis)
   * Fix kube2sky flakes. Fix tools.GetEtcdVersion to work with etcd > 2.0.7 #7675 (cjcullen)
   * Change kube2sky to use token-system-dns secret, point at https endpoint ... #7154 (cjcullen)
   * replica: serialize created-by reference #7468 (simon3z)
   * Inject mounter into volume plugins #7702 (pmorie)
   * bringing CoreOS cloud-configs up-to-date (against 0.15.x and latest OS' alpha)  #6973 (AntonioMeireles)
   * Update kubeconfig-file doc. #7787 (jlowdermilk)
   * Throw an API error when deleting namespace in termination #7780 (derekwaynecarr)
   * Fix command field PodExecOptions #7773 (csrwng)
   * Start ImageManager housekeeping in Run(). #7785 (vmarmol)
   * fix DeepCopy to properly support runtime.EmbeddedObject #7769 (deads2k)
   * fix master service endpoint system for multiple masters #7273 (lavalamp)
   * Add genbashcomp to KUBE_TEST_TARGETS #7757 (nak3)
   * Change the cloud provider TCPLoadBalancerExists function to GetTCPLoadBalancer... #7669 (a-robinson)
   * Add containerized option to kubelet binary #7772 (pmorie)
   * Fix swagger spec #7779 (pmorie)
   * FIX: Issue #7750 - Hyperkube docker image needs certificates to connect to cloud-providers #7755 (viklas)
   * Add build labels to rkt #7752 (vmarmol)
   * Check license boilerplate for python files #7672 (eparis)
   * Reliable updates in rollingupdate #7705 (bprashanth)
   * Don't exit abruptly if there aren't yet any minions right after the cluster is created. #7650 (roberthbailey)
   * Make changes suggested in #7675 #7742 (cjcullen)
   * A guide to set up kubernetes multiple nodes cluster with flannel on fedora #7357 (aveshagarwal)
   * Setup generators in factory #7760 (kargakis)
   * Reduce usage of time.After #7737 (lavalamp)
   * Remove node status from "componentstatuses" call. #7735 (fabioy)
   * React to failure by growing the remaining clusters #7614 (tamsky)
   * Fix typo in runtime_cache.go #7725 (pmorie)
   * Update non-GCE Salt distros to 1.6.0, fallback to ContainerVM Docker version on GCE #7740 (zmerlynn)
   * Skip SaltStack install if it's already installed #7744 (zmerlynn)
   * Expose pod name as a label on containers. #7712 (rjnagal)
   * Log which SSH key is used in e2e SSH test #7732 (mbforbes)
   * Add a central simple getting started guide with kubernetes guide. #7649 (brendandburns)
   * Explicitly state the lack of support for 'Requests' for the purposes of scheduling #7443 (vishh)
   * Select IPv4-only from host interfaces #7721 (smarterclayton)
   * Metrics tests can't run on Mac #7723 (smarterclayton)
   * Add step to API changes doc for swagger regen #7727 (pmorie)
   * Add NsenterMounter mount implementation #7703 (pmorie)
   * add StringSet.HasAny #7509 (deads2k)
   * Add an integration test that checks for the metrics we expect to be exported from the master #6941 (a-robinson)
   * Minor bash update found by shellcheck.net #7722 (eparis)
   * Add --hostport to run-container. #7536 (rjnagal)
   * Have rkt implement the container Runtime interface #7659 (vmarmol)
   * Change the order the different versions of API are registered  #7629 (caesarxuchao)
   * expose: Create objects in a generic way #7699 (kargakis)
   * Requeue rc if a single get/put retry on status.Replicas fails #7643 (bprashanth)
   * logs for master components #7316 (ArtfulCoder)
   * cloudproviders: add ovirt getting started guide #7522 (simon3z)
   * Make rkt-install a oneshot. #7671 (vmarmol)
   * Provide container_runtime flag to Kubelet in CoreOS. #7665 (vmarmol)
   * Boilerplate speedup #7654 (eparis)
   * Log host for failed pod in Density test #7700 (wojtek-t)
   * Removes spurious quotation mark #7655 (alindeman)
   * Add kubectl_label to custom functions in bash completion #7694 (nak3)
   * Enable profiling in kube-controller #7696 (wojtek-t)
   * Set vagrant test cluster default NUM_MINIONS=2 #7690 (BenTheElder)
   * Add metrics to measure cache hit ratio #7695 (fgrzadkowski)
   * Change IP to IP(S) in service columns for kubectl get #7662 (jlowdermilk)
   * annotate required flags for bash_completions #7076 (eparis)
   * (minor) Add pgrep debugging to etcd error #7685 (jayunit100)
   * Fixed nil pointer issue in describe when volume is unbound #7676 (markturansky)
   * Removed unnecessary closing bracket #7691 (piosz)
   * Added TerminationGracePeriod field to PodSpec and grace-period flag to kubectl stop #7432 (piosz)
   * Fix boilerplate in test/e2e/scale.go #7689 (wojtek-t)
   * Update expiration timeout based on observed latencies #7628 (bprashanth)
   * Output generated conversion functions/names #7644 (liggitt)
   * Moved the Scale tests into a scale file. #7645 #7646 (rrati)
   * Truncate GCE load balancer names to 63 chars #7609 (brendandburns)
   * Add SyncPod() and remove Kill/Run InContainer(). #7603 (vmarmol)
   * Merge release 0.16 to master #7663 (brendandburns)
   * Update license boilerplate for examples/rethinkdb #7637 (eparis)
   * First part of improved rolling update, allow dynamic next replication controller generation. #7268 (brendandburns)
   * Add license boilerplate to examples/phabricator #7638 (eparis)
   * Use generic copyright holder name in license boilerplate #7597 (eparis)
   * Retry incrementing quota if there is a conflict #7633 (derekwaynecarr)
   * Remove GetContainers from Runtime interface #7568 (yujuhong)
   * Add image-related methods to DockerManager #7578 (yujuhong)
   * Remove more docker references in kubelet #7586 (yujuhong)
   * Add KillContainerInPod in DockerManager #7601 (yujuhong)
   * Kubelet: Add container runtime option. #7652 (vmarmol)
   * bump heapster to v0.11.0 and grafana to v0.7.0 #7626 (idosh)
   * Build github.com/onsi/ginkgo/ginkgo as a part of the release #7593 (ixdy)
   * Do not automatically decode runtime.RawExtension #7490 (smarterclayton)
   * Update changelog. #7500 (brendandburns)
   * Add SyncPod() to DockerManager and use it in Kubelet #7610 (vmarmol)
   * Build: Push .md5 and .sha1 files for every file we push to GCS #7602 (zmerlynn)
   * Fix rolling update --image  #7540 (bprashanth)
   * Update license boilerplate for docs/man/md2man-all.sh #7636 (eparis)
   * Include shell license boilerplate in examples/k8petstore #7632 (eparis)
   * Add --cgroup_parent flag to Kubelet to set the parent cgroup for pods #7277 (guenter)
   * Proposal for High Availability of Daemons #6995 (rrati)
   * change the current dir to the config dir #7209 (you-n-g)
   * Set Weave To 0.9.0 And Update Etcd Configuration For Azure #7158 (idosh)
   * Augment describe to search for matching things if it doesn't match the original resource. #7467 (brendandburns)
   * Add a simple cache for objects stored in etcd. #7559 (fgrzadkowski)
   * Rkt gc #7549 (yifan-gu)
   * Rkt pull #7550 (yifan-gu)
   * Implement Mount interface using mount(8) and umount(8) #6400 (ddysher)
   * Trim Fleuntd tag for Cloud Logging #7588 (satnam6502)
   * GCE CoreOS cluster - set master name based on variable #7569 (bakins)
   * Capitalization of KubeProxyVersion wrong in JSON #7535 (smarterclayton)
   * Make nodes report their external IP rather than the master's. #7530 (mbforbes)
   * Trim cluster log tags to pod name and container name #7539 (satnam6502)
   * Handle conversion of boolean query parameters with a value of "false" #7541 (csrwng)
   * Add image-related methods to Runtime interface. #7532 (vmarmol)
   * Test whether auto-generated conversions weren't manually edited #7560 (wojtek-t)
   * Mention :latest behavior for image version tag #7484 (colemickens)
   * readinessProbe calls livenessProbe.Exec.Command which cause "invalid memory address or nil pointer dereference". #7487 (njuicsgz)
   * Add RuntimeHooks to abstract Kubelet logic #7520 (vmarmol)
   * Expose URL() on Request to allow building URLs #7546 (smarterclayton)
   * Add a simple cache for objects stored in etcd #7288 (fgrzadkowski)
   * Prepare for chaining autogenerated conversion methods  #7431 (wojtek-t)
   * Increase maxIdleConnection limit when creating etcd client in apiserver. #7353 (wojtek-t)
   * Improvements to generator of conversion methods. #7354 (wojtek-t)
   * Code to automatically generate conversion methods #7107 (wojtek-t)
   * Support recovery for anonymous roll outs #7407 (brendandburns)
   * Bump kube2sky to 1.2. Point it at https endpoint (3rd try). #7527 (cjcullen)
   * cluster/gce/coreos: Add metadata-service in node.yaml #7526 (yifan-gu)
   * Move ComputePodChanges to the Docker runtime #7480 (vmarmol)
   * Cobra rebase #7510 (eparis)
   * Adding system oom events from kubelet #6718 (vishh)
   * Move Prober to its own subpackage #7479 (vmarmol)
   * Fix parallel-e2e.sh to work on my macbook (bash v3.2) #7513 (cjcullen)
   * Move network plugin TearDown to DockerManager #7449 (vmarmol)
   * Fixes #7498 - CoreOS Getting Started Guide had invalid cloud config #7499 (elsonrodriguez)
   * Fix invalid character '"' after object key:value pair #7504 (resouer)
   * Fixed kubelet deleting data from volumes on stop (#7317). #7503 (jsafrane)
   * Fixing hooks/description to catch API fields without description tags #7482 (nikhiljindal)
   * RcManager watches pods and RCs instead of polling every 10s #6866 (bprashanth)
   * cadvisor is obsoleted so kubelet service does not require it. #7457 (aveshagarwal)
   * Set the default namespace for events to be "default" #7408 (vishh)
   * Fix typo in namespace conversion #7446 (liggitt)
   * Convert Secret registry to use update/create strategy, allow filtering by Type #7419 (liggitt)
   * Adds support for multiple resources to kubectl #4667 (kbeecher)
   * Use pod namespace when looking for its GlusterFS endpoints. #7102 (jsafrane)
   * Fixed name of kube-proxy path in deployment scripts. #7427 (jsafrane)
   * Added events back to Node Controller #6561 (piosz)
   * Added rate limiting to pod deleting #6355 (piosz)
   * Removed PodStatus.Host #6352 (piosz)


## 0.16.0
   * Bring up a kuberenetes cluster using coreos image as worker nodes #7445 (dchen1107)
   * Cloning v1beta3 as v1 and exposing it in the apiserver #7454 (nikhiljindal)
   * API Conventions for Late-initializers #7366 (erictune)
   * Upgrade Elasticsearch to 1.5.2 for cluster logging #7455 (satnam6502)
   * Make delete actually stop resources by default. #7210 (brendandburns)
   * Change kube2sky to use token-system-dns secret, point at https endpoint ... #7154 (cjcullen)
   * Updated CoreOS bare metal docs for 0.15.0 #7364 (hvolkmer) 
   * Print named ports in 'describe service' #7424 (thockin) 
   * AWS
      * Return public & private addresses in GetNodeAddresses #7040 (justinsb) 
      * Improving getting existing VPC and subnet #6606 (gust1n) 
      * Set hostname_override for minions, back to fully-qualified name #7182 (justinsb)
   * Conversion to v1beta3
      * Convert node level logging agents to v1beta3 #7274 (satnam6502)
      * Removing more references to v1beta1 from pkg/ #7128 (nikhiljindal) 
      * update examples/cassandra to api v1beta3 #7258 (caesarxuchao) 
      * Convert Elasticsearch logging to v1beta3 and de-salt #7246 (satnam6502) 
      * Update examples/storm for v1beta3 #7231 (bcbroussard)
      * Update examples/spark for v1beta3 #7230 (bcbroussard)
      * Update Kibana RC and service to v1beta3 #7240 (satnam6502)
      * Updating the guestbook example to v1beta3 #7194 (nikhiljindal)
      * Update Phabricator to v1beta3 example #7232 (bcbroussard)
      * Update Kibana pod to speak to Elasticsearch using v1beta3 #7206 (satnam6502) 
   * Validate Node IPs; clean up validation code #7180 (ddysher)
   * Add PortForward to runtime API. #7391 (vmarmol)
   * kube-proxy uses token to access port 443 of apiserver #7303 (erictune)
   * Move the logging-related directories to where I think they belong #7014 (a-robinson)
   * Make client service requests use the default timeout now that external load balancers are created asynchronously #6870 (a-robinson)
   * Fix bug in kube-proxy of not updating iptables rules if a service's public IPs change #6123 (a-robinson)
   * PersistentVolumeClaimBinder #6105 (markturansky)
   * Fixed validation message when trying to submit incorrect secret #7356 (soltysh) 
   * First step to supporting multiple k8s clusters #6006 (justinsb)
   * Parity for namespace handling in secrets E2E #7361 (pmorie) 
   * Add cleanup policy to RollingUpdater #6996 (ironcladlou) 
   * Use narrowly scoped interfaces for client access #6871 (ironcladlou) 
   * Warning about Critical bug in the GlusterFS Volume Plugin #7319 (wattsteve) 
   * Rolling update
      * First part of improved rolling update, allow dynamic next replication controller generation. #7268 (brendandburns) 
      * Further implementation of rolling-update, add rename #7279 (brendandburns)
   * Added basic apiserver authz tests. #7293 (ashcrow) 
   * Retry pod update on version conflict error in e2e test. #7297 (quinton-hoole)
   * Add "kubectl validate" command to do a cluster health check. #6597 (fabioy)
   * coreos/azure: Weave version bump, various other enhancements #7224 (errordeveloper)
   * Azure: Wait for salt completion on cluster initialization #6576 (jeffmendoza) 
   * Tighten label parsing #6674 (kargakis) 
   * fix watch of single object #7263 (lavalamp)
   * Upgrade go-dockerclient dependency to support CgroupParent #7247 (guenter) 
   * Make secret volume plugin idempotent #7166 (pmorie)
   * Salt reconfiguration to get rid of nginx on GCE #6618 (roberthbailey) 
   * Revert "Change kube2sky to use token-system-dns secret, point at https e... #7207 (fabioy) 
   * Pod templates as their own type #5012 (smarterclayton)
   * iscsi Test: Add explicit check for attach and detach calls. #7110 (swagiaal) 
   * Added field selector for listing pods #7067 (ravigadde) 
   * Record an event on node schedulable changes #7138 (pravisankar) 
   * Resolve #6812, limit length of load balancer names #7145 (caesarxuchao)
   * Convert error strings to proper validation errors. #7131 (rjnagal) 
   * ResourceQuota add object count support for secret and volume claims #6593 (derekwaynecarr) 
   * Use Pod.Spec.Host instead of Pod.Status.HostIP for pod subresources #6985 (csrwng)
   * Prioritize deleting the non-running pods when reducing replicas #6992 (yujuhong) 
   * Kubernetes UI with Dashboard component #7056 (preillyme)

## 0.15.0
* Enables v1beta3 API and sets it to the default API version (#6098)
  * See the [v1beta3 conversion guide](http://docs.k8s.io/api.md#v1beta3-conversion-tips)
* Added multi-port Services (#6182)
* New Getting Started Guides
  * Multi-node local startup guide (#6505)
  * JUJU (#5414)
  * Mesos on Google Cloud Platform (#5442)
  * Ansible Setup instructions (#6237)
* Added a controller framework (#5270, #5473)
* The Kubelet now listens on a secure HTTPS port (#6380)
* Made kubectl errors more user-friendly (#6338)
* The apiserver now supports client cert authentication (#6190)
* The apiserver now limits the number of concurrent requests it processes (#6207)
* Added rate limiting to pod deleting (#6355)
* Implement Balanced Resource Allocation algorithm as a PriorityFunction in scheduler package (#6150)
* Enabled log collection from master (#6396)
* Added an api endpoint to pull logs from Pods (#6497)
* Added latency metrics to scheduler (#6368)
* Added latency metrics to REST client (#6409)
* etcd now runs in a pod on the master (#6221)
* nginx now runs in a container on the master (#6334)
* Began creating Docker images for master components (#6326)
* Updated GCE provider to work with gcloud 0.9.54 (#6270)
* Updated AWS provider to fix Region vs Zone semantics (#6011)
* Record event when image GC fails (#6091)
* Add a QPS limiter to the kubernetes client (#6203)
* Decrease the time it takes to run make release (#6196)
* New volume support
  * Added iscsi volume plugin (#5506)
  * Added glusterfs volume plugin (#6174)
  * AWS EBS volume support (#5138)
* Updated to heapster version to v0.10.0 (#6331)
* Updated to etcd 2.0.9 (#6544)
* Updated to Kibana to v1.2 (#6426)
* Bug Fixes
  * Kube-proxy now updates iptables rules if a service's public IPs change (#6123)
  * Retry kube-addons creation if the initial creation fails (#6200)
  * Make kube-proxy more resiliant to running out of file descriptors (#6727)

## 0.14.2
 * Fix a regression in service port handling validation
 * Add a work around for etcd bugs in watch

## 0.14.1
 * Fixed an issue where containers with hostPort would sometimes go pending forever. (#6110)

## 0.14.0
 * Add HostNetworking container option to the API.
 * PersistentVolume API
 * NFS volume plugin fixed/re-added
 * Upgraded to etcd 2.0.5 on Salt configs
 * .kubeconfig changes
 * Kubelet now posts pod status to master, versus master polling.
 * All cluster add-on images are pulled from gcr.io

## 0.13.2
 * Fixes possible cluster bring-up flakiness on GCE/Salt based clusters
 

## 0.12.2
 * #5348 - Health check the docker socket and Docker generally
 * #5395 - Garbage collect unknown containers

## 0.12.1
 * DockerCache doesn't get containers at startup (#5115)
 * Update version of kube2sky to 1.1 (#5127)
 * Monit health check kubelet and restart unhealthy one (#5120)

## 0.12.0
 * Hide the infrastructure pod from users
 * Configure scheduler via JSON
 * Improved object validation
 * Improved messages on scheduler failure
 * Improved messages on port conflicts
 * Move to thread-per-pod in the kubelet
 * Misc. kubectl improvements
 * Update etcd used by SkyDNS to 2.0.3
 * Fixes to GCE PD support
 * Improved support for secrets in the API
 * Improved OOM behavior

## 0.11
* Secret API Resources
* Better error handling in various places
* Improved RackSpace support
* Fix ```kubectl``` patch behavior
* Health check failures fire events
* Don't delete the pod infrastructure container on health check failures
* Improvements to Pod Status detection and reporting
* Reduce the size of scheduled pods in etcd
* Fix some bugs in namespace clashing
* More detailed info on failed image pulls
* Remove pods from a failed node
* Safe format and mount of GCE PDs
* Make events more resilient to etcd watch failures
* Upgrade to container-vm 01-29-2015

## 0.10
   * Improvements to swagger API documentation.
   * Upgrade container VM to 20150129
   * Start to move e2e tests to Ginkgo
   * Fix apiserver proxy path rewriting
   * Upgrade to etcd 2.0.0
   * Add a wordpress/mysql example
   * Improve responsiveness of the master when creating new pods
   * Improve api object validation in numerous small ways
   * Add support for IPC namespaces
   * Improve GCE PD support
   * Make replica controllers with node selectors work correctly
   * Lots of improvements to e2e tests (more to come)

## 0.9
### Features
 - Various improvements to kubectl
 - Improvements to API Server caching
 - Full control over container command (docker entrypoint) and arguments (docker cmd);
   users of v1beta3 must change to use the Args field of the container for images that
   set a default entrypoint

### Bug fixes
 - Disable image GC since it was causing docker pull problems
 - Various small bug fixes

## 0.8
### Features 
 - Docker 1.4.1
 - Optional session affinity for Services
 - Better information on out of memory errors
 - Scheduling pods on specific machines
 - Improve performance of Pod listing
 - Image garbage collection
 - Automatic internal DNS for Services
 - Swagger UI for the API
 - Update cAdvisor Manifest to use google/cadvisor:0.7.1 image

### Bug fixes
 - Fix Docker exec liveness health checks
 - Fix a bug where the service proxy would ignore new events
 - Fix a crash for kubelet when without EtcdClient

## 0.7
### Features
  - Make updating node labels easier
  - Support updating node capacity
  - kubectl streaming log support
  - Improve /validate validation
  - Fix GCE-PD to work across machine reboots
  - Don't delete other attached disks on cluster turn-down
  - Return errors if a user attempts to create a UDP external balancer
  - TLS version bump from SSLv3 to TLSv1.0
  - x509 request authenticator
  - Container VM on GCE updated to 20141208
  - Improvements to kubectl yaml handling
### Bug fixes
  - Fix kubelet panics when docker has no name for containers
  - Only count non-dead pods in replica controller status reporting
  - Fix version requirements for docker exec

## 0.6
### Features
  - Docker 1.3.3 (0.6.2)
  - Authentication for Kubelet/Apiserver communication
  - Kubectl clean ups
  - Enable Docker Cache on GCE
  - Better support for Private Repositories
### Bug fixes
  - Fixed Public IP support on non-GCE hosts
  - Fixed 32-bit build

## 0.5 (11/17/2014)
### Features
  - New client utility available: kubectl. This will eventually replace kubecfg. (#1325)
  - Services v2. We now assign IP addresses to services.  Details in #1107. (#1402)
  - Event support: (#1789, #2267, #2270, #2384)
  - Namespaces: (#1564)
  - Fixes for Docker 1.3 (#1841, #1842)
  - Support for automatically installing log saving and searching using fluentd and elasticsearch (#1610) and GCP logging (#1919).  If using elastic search, logs can be viewed with Kibana (#2013)
  - Read only API endpoint for internal lookups (#1916)
  - Lots of ground work for pluggable auth model. (#1847)
  - "run once" mode for the kubelet (#1707)
  - Restrict which minion a pod schedules on based on predicate tested agains minion labels. (#1946, #2007)
  - git based volumes: (#1945)
  - Container garbage collection.  Remove old instances of containers in the case of crash/fail loops. (#2022)
  - Publish the APIServer as a service to pods in the cluster (#1920)
  - Heapster monitoring (#2208)
  - cAdvisor 0.5.0
  - Switch default pull policy to PullIfNotPresent (#2388) except latest images
  - Initial IPv6 support (#2147)
  - Service proxy retry support (#2281)
  - Windows client build (largely untested) (#2332)
  - UDP Portals (#2191)
  - Capture application termination log (#2225)
  - pod update support (#1865, #2077, #2160)

### Cluster/Cloud support
  - Add OpenStack support with CloudProvider. (#1676)
  - Example systemd units (#1831)
  - Updated Rackspace support based on CoreOS (#1832)
  - Automatic security updates for debian based systems (#2012)
  - For debian (and GCE) pull docker (#2104), salt and etcd (#2245) from Google Cloud Storage.
  - For GCE, start with the Container VM image instead of stock debian.  This enables memcg support. (#2046)
  - Cluster install: Updated support for deploying to vSphere (#1747)
  - AWS support (#2260, #2216)

### Examples/Extras/Docs
  - Documentation on how to use SkyDNS with Kubernetes (#1845)
  - Podex (convert Docker image to pod desc) tool now supports multiple images. (#1898)
  - Documentation: 201 level walk through. (#1924)
  - Local Docker Setup: (#1716)

## 0.4 (10/14/2014)
 - Support Persistent Disk volume type
