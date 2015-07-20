---
layout: docwithnav
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
This is a simple web server pod which serves HTML from an AWS EBS
volume.

Create a volume in the same region as your node add your volume
information in the pod description file aws-ebs-web.yaml then create
the pod:

{% highlight sh %}
  $ kubectl create -f examples/aws_ebs/aws-ebs-web.yaml
{% endhighlight %}

Add some data to the volume if is empty:

{% highlight sh %}
  $ echo  "Hello World" >& /var/lib/kubelet/plugins/kubernetes.io/aws-ebs/mounts/aws/{Region}/{Volume ID}/index.html
{% endhighlight %}

You should now be able to query your web server:

{% highlight sh %}
  $ curl <Pod IP address>
  $ Hello World
{% endhighlight %}


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/aws_ebs/README.html?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

