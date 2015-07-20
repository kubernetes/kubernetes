---
layout: docwithnav
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Install and configure kubectl

## Download the kubectl CLI tool

{% highlight bash %}
### Darwin
wget https://storage.googleapis.com/kubernetes-release/release/v0.19.3/bin/darwin/amd64/kubectl

### Linux
wget https://storage.googleapis.com/kubernetes-release/release/v0.19.3/bin/linux/amd64/kubectl
{% endhighlight %}

### Copy kubectl to your path

{% highlight bash %}
chmod +x kubectl
mv kubectl /usr/local/bin/
{% endhighlight %}

### Create a secure tunnel for API communication

{% highlight bash %}
ssh -f -nNT -L 8080:127.0.0.1:8080 core@<master-public-ip>
{% endhighlight %}


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/aws/kubectl.html?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

