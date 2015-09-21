## Deploying Kubernetes on [CloudStack](http://cloudstack.apache.org)

CloudStack is software to build public and private clouds based on hardware virtualization principles (traditional IaaS). To deploy Kubernetes on CloudStack there are several possibilities depending on the Cloud being used and what images are made available. [Exoscale](http://exoscale.ch) for instance makes a [CoreOS](http://coreos.com) template available, therefore instructions to deploy Kubernetes on coreOS can be used. CloudStack also has a vagrant plugin available, hence Vagrant could be used to deploy Kubernetes either using the existing shell provisioner or using new Salt based recipes.

Here we introduce the existing documentation.

* [Kubernetes on Exoscale](https://github.com/runseb/kubernetes-exoscale)

