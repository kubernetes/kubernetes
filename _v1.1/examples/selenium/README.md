---
layout: docwithnav
title: "title: \"</strong>\""
---
---
layout: docwithnav
title: "</strong>"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Selenium on Kubernetes

Selenium is a browser automation tool used primarily for testing web applications. However when Selenium is used in a CI pipeline to test applications, there is often contention around the use of Selenium resources. This example shows you how to deploy Selenium to Kubernetes in a scalable fashion.

### Prerequisites

This example assumes you have a working Kubernetes cluster and a properly configured kubectl client. See the [Getting Started Guides](../../docs/getting-started-guides/) for details.

Google Container Engine is also a quick way to get Kubernetes up and running: https://cloud.google.com/container-engine/

Your cluster must have 4 CPU and 6 GB of RAM to complete the example up to the scaling portion.

### Deploy Selenium Grid Hub:

We will be using Selenium Grid Hub to make our Selenium install scalable via a master/worker model. The Selenium Hub is the master, and the Selenium Nodes are the workers(not to be confused with Kubernetes nodes). We only need one hub, but we're using a replication controller to ensure that the hub is always running:

{% highlight console %}
{% raw %}
kubectl create --filename=examples/selenium/selenium-hub-rc.yaml
{% endraw %}
{% endhighlight %}

The Selenium Nodes will need to know how to get to the Hub, let's create a service for the nodes to connect to.

{% highlight console %}
{% raw %}
kubectl create --filename=examples/selenium/selenium-hub-svc.yaml
{% endraw %}
{% endhighlight %}

### Verify Selenium Hub Deployment

Let's verify our deployment of Selenium hub by connecting to the web console.

#### Kubernetes Nodes Reachable

If your Kubernetes nodes are reachable from your network, you can verify the hub by hitting it on the nodeport. You can retrieve the nodeport by typing `kubectl describe svc selenium-hub`, however the snippet below automates that by using kubectl's template functionality:

{% highlight console %}
{% raw %}
export NODEPORT=`kubectl get svc --selector='app=selenium-hub' --output=template --template="{{ with index .items 0}}{{with index .spec.ports 0 }}{{.nodePort}}{{end}}{{end}}"`
export NODE=`kubectl get nodes --output=template --template="{{with index .items 0 }}{{.metadata.name}}{{end}}"`

curl http://$NODE:$NODEPORT
{% endraw %}
{% endhighlight %}

#### Kubernetes Nodes Unreachable

If you cannot reach your Kubernetes nodes from your network, you can proxy via kubectl.

{% highlight console %}
{% raw %}
export PODNAME=`kubectl get pods --selector="app=selenium-hub" --output=template --template="{{with index .items 0}}{{.metadata.name}}{{end}}"`
kubectl port-forward --pod=$PODNAME 4444:4444
{% endraw %}
{% endhighlight %}

In a seperate terminal, you can now check the status.

{% highlight console %}
{% raw %}
curl http://localhost:4444
{% endraw %}
{% endhighlight %}

#### Using Google Container Engine

If you are using Google Container Engine, you can expose your hub via the internet. This is a bad idea for many reasons, but you can do it as follows:

{% highlight console %}
{% raw %}
kubectl expose rc selenium-hub --name=selenium-hub-external --labels="app=selenium-hub,external=true" --create-external-load-balancer=true
{% endraw %}
{% endhighlight %}

Then wait a few minutes, eventually your new `selenium-hub-external` service will be assigned a load balanced IP from gcloud. Once `kubectl get svc selenium-hub-external` shows two IPs, run this snippet.

{% highlight console %}
{% raw %}
export INTERNET_IP=`kubectl get svc --selector="app=selenium-hub,external=true" --output=template --template="{{with index .items 0}}{{with index .status.loadBalancer.ingress 0}}{{.ip}}{{end}}{{end}}"`

curl http://$INTERNET_IP:4444/
{% endraw %}
{% endhighlight %}

You should now be able to hit `$INTERNET_IP` via your web browser, and so can everyone else on the Internet!

### Deploy Firefox and Chrome Nodes:

Now that the Hub is up, we can deploy workers.

This will deploy 2 Chrome nodes.

{% highlight console %}
{% raw %}
kubectl create --file=examples/selenium/selenium-node-chrome-rc.yaml
{% endraw %}
{% endhighlight %}

And 2 Firefox nodes to match.

{% highlight console %}
{% raw %}
kubectl create --file=examples/selenium/selenium-node-firefox-rc.yaml
{% endraw %}
{% endhighlight %}

Once the pods start, you will see them show up in the Selenium Hub interface.

### Run a Selenium Job

Let's run a quick Selenium job to validate our setup.

#### Setup Python Environment

First, we need to start a python container that we can attach to.

{% highlight console %}
{% raw %}
kubectl run selenium-python --image=google/python-hello
{% endraw %}
{% endhighlight %}

Next, we need to get inside this container.

{% highlight console %}
{% raw %}
export PODNAME=`kubectl get pods --selector="run=selenium-python" --output=template --template="{{with index .items 0}}{{.metadata.name}}{{end}}"`
kubectl exec --stdin=true --tty=true $PODNAME bash
{% endraw %}
{% endhighlight %}

Once inside, we need to install the Selenium library

{% highlight console %}
{% raw %}
pip install selenium
{% endraw %}
{% endhighlight %}

#### Run Selenium Job with Python

We're all set up, start the python interpreter.

{% highlight console %}
{% raw %}
python
{% endraw %}
{% endhighlight %}

And paste in the contents of selenium-test.py.

{% highlight python %}
{% raw %}
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

def check_browser(browser):
  driver = webdriver.Remote(
    command_executor='http://selenium-hub:4444/wd/hub',
    desired_capabilities=getattr(DesiredCapabilities, browser)
  )
  driver.get("http://google.com")
  assert "google" in driver.page_source
  driver.close()
  print("Browser %s checks out!" % browser)


check_browser("FIREFOX")
check_browser("CHROME")
{% endraw %}
{% endhighlight %}

You should get

```
{% raw %}
{% raw %}
>>> check_browser("FIREFOX")
Browser FIREFOX checks out!
>>> check_browser("CHROME")
Browser CHROME checks out!
{% endraw %}
{% endraw %}
```

Congratulations, your Selenium Hub is up, with Firefox and Chrome nodes!

### Scale your Firefox and Chrome nodes.

If you need more Firefox or Chrome nodes, your hardware is the limit:

{% highlight console %}
{% raw %}
kubectl scale rc selenium-node-firefox --replicas=10
kubectl scale rc selenium-node-chrome --replicas=10
{% endraw %}
{% endhighlight %}

You now have 10 Firefox and 10 Chrome nodes, happy Seleniuming!

### Debugging

Sometimes it is neccessary to check on a hung test. Each pod is running VNC. To check on one of the browser nodes via VNC, it's reccomended that you proxy, since we don't want to expose a service for every pod, and the containers have a weak VNC password. Replace POD_NAME with the name of the pod you want to connect to.

{% highlight console %}
{% raw %}
kubectl port-forward --pod=POD_NAME 5900:5900
{% endraw %}
{% endhighlight %}

Then connect to localhost:5900 with your VNC client using the password "secret"

Enjoy your scalable Selenium Grid!

Adapted from: https://github.com/SeleniumHQ/docker-selenium

### Teardown

To remove all created resources, run the following:

{% highlight console %}
{% raw %}
kubectl delete rc selenium-hub
kubectl delete rc selenium-node-chrome
kubectl delete rc selenium-node-firefox
kubectl delete rc selenium-python
kubectl delete svc selenium-hub
kubectl delete svc selenium-hub-external
{% endraw %}
{% endhighlight %}


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/selenium/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->


