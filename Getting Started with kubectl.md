##Getting Started with kubectl##

###1. Build kubectl.###
Requirements on build environments:

1. Your operating system is Linux.
2. The Go programming language is supported.

Run the following command to build kubectl:

	./hack/build-go.sh  cmd/kubectl
	
The executable file of kubectl is generated under the _output/local/bin/linux/amd64/ directory after the build is completed.

###2. Install kubectl.###
Copy kubectl to the /usr/local/bin directory.

	cp _output/local/bin/linux/amd64/kubectl /usr/local/bin/
	mkdir ~/.kube
	cp -r kube/* ~/.kube/

###3. Configure kubectl.###

####1) Set cluster####

In Cloud Container Engine service, multiple clusters are managed within a tenant account. Hence to operate a spedificed cluster, you need set the cluster to kubectl.

To access cluster in Open Telekom Cloud (OTC) or Huawei Web Services (HWS), configure with following command:

        kubectl config set-cluster {cluster name} --server={server endpoint} --cluster-uuid={cluster uuid}  

As in OTC and HWS, a certificate is attached in the API Gateway, hence user dones't need to configure certificate manually.

For other CCE deployments, user can either skip the certificate verification or use a self-signed certificate.
 
To skip the certificate verification, configure cluster with following command:

	kubectl config set-cluster {cluster name} --server={server endpoint} --cluster-uuid={cluster uuid} --insecure-skip-tls-verify=true

To use self-signed certificate, user need:  
1. Download the certificate.  
2. Configure cluster with following command:   
	
	kubectl config set-cluster {cluster name} --server={server endpoint} --cluster-uuid={cluster uuid} --certificate-authority={path of certificate file}  

####2) Set credentials and context####

Run the following commands to configure kubectl:
	
	kubectl config set-credentials {user name} --access-key={access key} --secret-key={secret key} --region-id={region id}
	
	kubectl config set-context {context name} --cluster={cluster name} --user={user name}
	
	kubectl config set current-context {context name}

To view configuration information, run the **Kubectl config view** command or **cat  ~/.kube/config** command.


###4. Supported Commands###


Command|Parameter|Description|
--------|--------|-----------|
 get 	| endpoints | 
     	| namespaces |   
     	| pods | 
     	| replicationcontrollers | 
     	| secrets	|Only one secret can be queried at a time. For example, the command for querying secret -a is Kubectl get secret secret-a.
     	| services
create	|endpoints
	  	|namespaces
		|pods
		|replicationcontrollers
		|services	
		|namespaces		
replace |	endpoints	
		|namespaces	
		|pods	
		|replicationcontrollers	
		|secrets	
		|services	
delete	|endpoints	
		|namespaces	
		|pods	
		|replicationcontrollers	
		|secrets	
		|services	
convert	|	
patch 	|endpoints	
		|namespaces	
		|pods	
		|replicationcontrollers	
		|services	
expose  |pods	
		|replicationcontrollers	
		|services	
annotate|endpoints	
		|namespaces	
		|pods	
		|replicationcontrollers	
		|services	
label	|endpoints	
		|namespaces	
		|pods	
		|replicationcontrollers	
		|services	
cluster-info|	
logs	|
api-version|
version |
config  |
apply   |
rolling-update |
scale |
proxy |
run   |
		
		
		

