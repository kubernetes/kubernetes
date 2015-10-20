---
layout: docwithnav
title: "Install and configure kubectl"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Install and configure kubectl

## Download the kubectl CLI tool

{% highlight bash %}
{% raw %}
### Darwin
wget https://storage.googleapis.com/kubernetes-release/release/v1.0.1/bin/darwin/amd64/kubectl

### Linux
wget https://storage.googleapis.com/kubernetes-release/release/v1.0.1/bin/linux/amd64/kubectl
{% endraw %}
{% endhighlight %}

### Copy kubectl to your path

{% highlight bash %}
{% raw %}
chmod +x kubectl
mv kubectl /usr/local/bin/
{% endraw %}
{% endhighlight %}

### Create a secure tunnel for API communication

{% highlight bash %}
{% raw %}
ssh -f -nNT -L 8080:127.0.0.1:8080 core@<master-public-ip>
{% endraw %}
{% endhighlight %}

<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/aws/kubectl.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

