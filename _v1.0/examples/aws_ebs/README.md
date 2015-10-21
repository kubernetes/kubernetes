---
layout: docwithnav
title: "aws_ebs/README"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
This is a simple web server pod which serves HTML from an AWS EBS
volume.

Create a volume in the same region as your node add your volume
information in the pod description file aws-ebs-web.yaml then create
the pod:

{% highlight sh %}
{% raw %}
  $ kubectl create -f examples/aws_ebs/aws-ebs-web.yaml
{% endraw %}
{% endhighlight %}

Add some data to the volume if is empty:

{% highlight sh %}
{% raw %}
  $ echo  "Hello World" >& /var/lib/kubelet/plugins/kubernetes.io/aws-ebs/mounts/aws/{Region}/{Volume ID}/index.html
{% endraw %}
{% endhighlight %}

You should now be able to query your web server:

{% highlight sh %}
{% raw %}
  $ curl <Pod IP address>
  $ Hello World
{% endraw %}
{% endhighlight %}


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/aws_ebs/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

