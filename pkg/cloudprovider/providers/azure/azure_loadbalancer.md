# Azure LoadBalancer

The way azure define LoadBalancer is different with GCE or AWS. Azure's LB can have multiple frontend IP refs. The GCE and AWS can only allow one, if you want more, you better to have another LB. Because of the fact, Public IP is not part of the LB in Azure. NSG is not part of LB in Azure either. However, you cannot delete them in parallel, Public IP can only be delete after LB's frontend IP ref is removed. 

For different Azure Resources, such as LB, Public IP, NSG. They are the same tier azure resources. We need to make sure there is no connection in their own ensure loops. In another words, They would be eventually reconciled regardless of other resources' state. They should only depends on service state.

Despite the ideal philosophy above, we have to face the reality. NSG depends on LB's frontend ip to adjust NSG rules. So when we want to reconcile NSG, the LB should contain the corresponding frontend ip config.

And also, For Azure, we cannot afford to have more than 1 worker of service_controller. Because, different services could operate on the same LB, concurrent execution could result in conflict or unexpected result. For AWS and GCE, they apparently doesn't have the problem, they use one LB per service, no such conflict.

There are two load balancers per availability set internal and external. There is a limit on number of services that can be associated with a single load balancer.
By default primary load balancer is selected. Services can be annotated to allow auto selection of available load balancers. Service annotations can also be used to provide specific availability sets that host the load balancers. Note that in case of auto selection or specific availability set selection, when the availability set is lost incase of downtime or cluster scale down the services are currently not auto assigned to an available load balancer.
Service Annotation for Auto and specific load balancer mode

- service.beta.kubernetes.io/azure-load-balancer-mode" (__auto__|as1,as2...)

## Introduce Functions

- reconcileLoadBalancer(clusterName string, service *v1.Service, nodes []*v1.Node, wantLb bool) (*network.LoadBalancer, error)
  - Go through lb's properties, update based on wantLb
  - If any change on the lb, no matter if the lb exists or not
    - Call az cloud to CreateOrUpdate on this lb, or Delete if nothing left
  - return lb, err

- reconcileSecurityGroup(clusterName string, service *v1.Service, lbIP *string, wantLb bool) (*network.SecurityGroup, error)
  - Go though NSG' properties, update based on wantLb
    - Use destinationIPAddress as target address if possible
    - Consolidate NSG rules if possible
  - If any change on the NSG, (the NSG should always exists)
    - Call az cloud to CreateOrUpdate on this NSG
  - return sg, err

- reconcilePublicIP(clusterName string, service *v1.Service, wantLb bool) (*network.PublicIPAddress, error)
  - List all the public ip in the resource group
  - Make sure we only touch Public IP resources has tags[service] = "namespace/serviceName"
    - skip for wantLb && !isInternal && pipName == desiredPipName
    - delete other public ip resources if any
  - if !isInternal && wantLb 
    - ensure Public IP with desiredPipName exists

- getServiceLoadBalancer(service *v1.Service, clusterName string, nodes []*v1.Node, wantLb bool) (lb, status, exists, error)
  - gets the loadbalancer for the service if it already exists
  - If wantLb is TRUE then -it selects a new load balancer, the selction helps distribute the services across load balancers
  - In case the selected load balancer does not exists it returns network.LoadBalancer struct with added metadata (such as name, location) and existsLB set to FALSE 
  - By default - cluster default LB is returned

## Define interface behaviors

### GetLoadBalancer

- Get LoadBalancer status, return status, error
  - return the load balancer status for this service
  - it will not create or update or delete any resource

### EnsureLoadBalancer

- Reconcile LB for the flipped service
  - Call reconcileLoadBalancer(clusterName, flipedService, nil, false/* wantLb */)
- Reconcile Public IP
  - Call reconcilePublicIP(cluster, service, true)
- Reconcile LB's related and owned resources, such as FrontEndIPConfig, Rules, Probe.
  - Call reconcileLoadBalancer(clusterName, service, nodes, true /* wantLb */)
- Reconcile NSG rules, it need to be called after reconcileLB
  - Call reconcileSecurityGroup(clusterName, service, lbStatus, true /* wantLb */)

### UpdateLoadBalancer

- Has no difference with EnsureLoadBalancer

### EnsureLoadBalancerDeleted

- Reconcile NSG first, before reconcile LB, because SG need LB to be there
  - Call reconcileSecurityGroup(clusterName, service, nil, false /* wantLb */)
- Reconcile LB's related and owned resources, such as FrontEndIPConfig, Rules, Probe.
  - Call reconcileLoadBalancer(clusterName, service, nodes, false)
- Reconcile Public IP, public IP needs related LB reconciled first
  - Call reconcilePublicIP(cluster, service, false)