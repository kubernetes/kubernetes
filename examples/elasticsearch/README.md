<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/examples/elasticsearch/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Elasticsearch for Kubernetes

Kubernetes makes it trivial for anyone to easily build and scale [Elasticsearch](http://www.elasticsearch.org/) clusters. Here, you'll find how to do so.
Current Elasticsearch version is `2.0.0`.

[A more robust example that follows Elasticsearch best-practices of separating nodes concern is also available](production_cluster/README.md).

<img src="http://kubernetes.io/img/warning.png" alt="WARNING" width="25" height="25"> Current pod descriptors use an `emptyDir` for storing data in each data node container. This is meant to be for the sake of simplicity and [should be adapted according to your storage needs](../../docs/design/persistent-storage.md).

## Docker image

This example uses [this pre-built image](https://github.com/pires/docker-elasticsearch-kubernetes) will not be supported. Feel free to fork to fit your own needs, but mind yourself that you will need to change Kubernetes descriptors accordingly.

## Deploy

Let's kickstart our cluster with 1 instance of Elasticsearch.

```
kubectl create -f examples/elasticsearch/service-account.yaml
kubectl create -f examples/elasticsearch/es-svc.yaml
kubectl create -f examples/elasticsearch/es-rc.yaml
```

Let's see if it worked:

```
$ kubectl get pods
NAME                      READY     STATUS    RESTARTS   AGE
es-v8fzi                  1/1       Running   0          13s
```

```
$ kubectl logs es-v8fzi
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
[2015-11-20 18:27:53,449][INFO ][node                     ] [Danielle Moonstar] version[2.0.0], pid[13], build[de54438/2015-10-22T08:09:48Z]
[2015-11-20 18:27:53,456][INFO ][node                     ] [Danielle Moonstar] initializing ...
[2015-11-20 18:27:53,786][INFO ][plugins                  ] [Danielle Moonstar] loaded [cloud-kubernetes], sites []
[2015-11-20 18:27:53,823][INFO ][env                      ] [Danielle Moonstar] using [1] data paths, mounts [[/data (/dev/disk/by-uuid/7c2ba6f8-3e2f-49da-9d24-211659759bdb)]], net usable_space [90gb], net total_space [98.3gb], spins? [possibly], types [ext4]
[2015-11-20 18:27:56,545][INFO ][node                     ] [Danielle Moonstar] initialized
[2015-11-20 18:27:56,551][INFO ][node                     ] [Danielle Moonstar] starting ...
[2015-11-20 18:27:56,612][INFO ][transport                ] [Danielle Moonstar] publish_address {10.56.0.21:9300}, bound_addresses {10.56.0.21:9300}
[2015-11-20 18:27:56,622][INFO ][discovery                ] [Danielle Moonstar] myesdb/C9nmBJw3TJ22JcAVQmdNeg
[2015-11-20 18:28:01,571][INFO ][cluster.service          ] [Danielle Moonstar] new_master {Danielle Moonstar}{C9nmBJw3TJ22JcAVQmdNeg}{10.56.0.21}{10.56.0.21:9300}{master=true}, reason: zen-disco-join(elected_as_master, [0] joins received)
[2015-11-20 18:28:01,614][INFO ][http                     ] [Danielle Moonstar] publish_address {10.56.0.21:9200}, bound_addresses {10.56.0.21:9200}
[2015-11-20 18:28:01,615][INFO ][node                     ] [Danielle Moonstar] started
[2015-11-20 18:28:01,652][INFO ][gateway                  ] [Danielle Moonstar] recovered [0] indices into cluster_state
```

So we have a 1-node Elasticsearch cluster ready to handle some work.

## Scale

Scaling is as easy as:

```
kubectl scale --replicas=3 rc es
```

Did it work?

```
$ kubectl get pods
NAME             READY     STATUS    RESTARTS   AGE
NAME                      READY     STATUS    RESTARTS   AGE
es-atj9s                  1/1       Running   0          30s
es-eombg                  1/1       Running   0          31s
es-v8fzi                  1/1       Running   0          1m
```

Let's take a look at logs:

```
$ kubectl logs es-v8fzi
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
log4j:WARN No such property [maxBackupIndex] in org.apache.log4j.DailyRollingFileAppender.
[2015-11-20 18:27:53,449][INFO ][node                     ] [Danielle Moonstar] version[2.0.0], pid[13], build[de54438/2015-10-22T08:09:48Z]
[2015-11-20 18:27:53,456][INFO ][node                     ] [Danielle Moonstar] initializing ...
[2015-11-20 18:27:53,786][INFO ][plugins                  ] [Danielle Moonstar] loaded [cloud-kubernetes], sites []
[2015-11-20 18:27:53,823][INFO ][env                      ] [Danielle Moonstar] using [1] data paths, mounts [[/data (/dev/disk/by-uuid/7c2ba6f8-3e2f-49da-9d24-211659759bdb)]], net usable_space [90gb], net total_space [98.3gb], spins? [possibly], types [ext4]
[2015-11-20 18:27:56,545][INFO ][node                     ] [Danielle Moonstar] initialized
[2015-11-20 18:27:56,551][INFO ][node                     ] [Danielle Moonstar] starting ...
[2015-11-20 18:27:56,612][INFO ][transport                ] [Danielle Moonstar] publish_address {10.56.0.21:9300}, bound_addresses {10.56.0.21:9300}
[2015-11-20 18:27:56,622][INFO ][discovery                ] [Danielle Moonstar] myesdb/C9nmBJw3TJ22JcAVQmdNeg
[2015-11-20 18:28:01,571][INFO ][cluster.service          ] [Danielle Moonstar] new_master {Danielle Moonstar}{C9nmBJw3TJ22JcAVQmdNeg}{10.56.0.21}{10.56.0.21:9300}{master=true}, reason: zen-disco-join(elected_as_master, [0] joins received)
[2015-11-20 18:28:01,614][INFO ][http                     ] [Danielle Moonstar] publish_address {10.56.0.21:9200}, bound_addresses {10.56.0.21:9200}
[2015-11-20 18:28:01,615][INFO ][node                     ] [Danielle Moonstar] started
[2015-11-20 18:28:01,652][INFO ][gateway                  ] [Danielle Moonstar] recovered [0] indices into cluster_state
[2015-11-20 18:29:36,776][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0xf28436c7, /10.56.1.12:43018 => /10.56.0.21:9200]
java.lang.IllegalArgumentException: invalid version format: INTERNAL:DISCOVERY/ZEN/UNICASTﾲ￐^MYESDB
                                                                                                   YURIKO OYAMATDYZYPQPT5IUFH0IVA0C9Q
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:94)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.messageReceived(ReplayingDecoder.java:435)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:70)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:36,782][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0xf28436c7, /10.56.1.12:43018 :> /10.56.0.21:9200]
java.lang.IllegalArgumentException: empty text
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:89)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.cleanup(ReplayingDecoder.java:554)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.channelDisconnected(FrameDecoder.java:365)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:102)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireChannelDisconnected(Channels.java:396)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.close(AbstractNioWorker.java:360)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.handleAcceptedSocket(NioServerSocketPipelineSink.java:81)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.eventSunk(NioServerSocketPipelineSink.java:36)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:779)
  at org.jboss.netty.handler.codec.oneone.OneToOneEncoder.handleDownstream(OneToOneEncoder.java:54)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:784)
  at org.jboss.netty.channel.SimpleChannelHandler.closeRequested(SimpleChannelHandler.java:334)
  at org.jboss.netty.channel.SimpleChannelHandler.handleDownstream(SimpleChannelHandler.java:260)
  at org.elasticsearch.http.netty.pipelining.HttpPipeliningHandler.handleDownstream(HttpPipeliningHandler.java:105)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:582)
  at org.jboss.netty.channel.Channels.close(Channels.java:812)
  at org.jboss.netty.channel.AbstractChannel.close(AbstractChannel.java:206)
  at org.elasticsearch.http.netty.NettyHttpServerTransport.exceptionCaught(NettyHttpServerTransport.java:364)
  at org.elasticsearch.http.netty.HttpRequestHandler.exceptionCaught(HttpRequestHandler.java:72)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelHandler.exceptionCaught(SimpleChannelHandler.java:156)
  at org.jboss.netty.channel.SimpleChannelHandler.handleUpstream(SimpleChannelHandler.java:130)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.exceptionCaught(FrameDecoder.java:377)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireExceptionCaught(Channels.java:525)
  at org.jboss.netty.channel.AbstractChannelSink.exceptionCaught(AbstractChannelSink.java:48)
  at org.jboss.netty.channel.DefaultChannelPipeline.notifyHandlerException(DefaultChannelPipeline.java:658)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:566)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:36,802][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0xbfeb90d6, /10.56.2.14:33276 => /10.56.0.21:9200]
java.lang.IllegalArgumentException: invalid version format: INTERNAL:DISCOVERY/ZEN/UNICASTﾲ￐^MYESDBECHO-7OAWXEMRLW5UXBTCXA8HW
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:94)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.messageReceived(ReplayingDecoder.java:435)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:70)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:36,803][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0xbfeb90d6, /10.56.2.14:33276 :> /10.56.0.21:9200]
java.lang.IllegalArgumentException: empty text
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:89)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.cleanup(ReplayingDecoder.java:554)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.channelDisconnected(FrameDecoder.java:365)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:102)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireChannelDisconnected(Channels.java:396)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.close(AbstractNioWorker.java:360)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.handleAcceptedSocket(NioServerSocketPipelineSink.java:81)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.eventSunk(NioServerSocketPipelineSink.java:36)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:779)
  at org.jboss.netty.handler.codec.oneone.OneToOneEncoder.handleDownstream(OneToOneEncoder.java:54)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:784)
  at org.jboss.netty.channel.SimpleChannelHandler.closeRequested(SimpleChannelHandler.java:334)
  at org.jboss.netty.channel.SimpleChannelHandler.handleDownstream(SimpleChannelHandler.java:260)
  at org.elasticsearch.http.netty.pipelining.HttpPipeliningHandler.handleDownstream(HttpPipeliningHandler.java:105)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:582)
  at org.jboss.netty.channel.Channels.close(Channels.java:812)
  at org.jboss.netty.channel.AbstractChannel.close(AbstractChannel.java:206)
  at org.elasticsearch.http.netty.NettyHttpServerTransport.exceptionCaught(NettyHttpServerTransport.java:364)
  at org.elasticsearch.http.netty.HttpRequestHandler.exceptionCaught(HttpRequestHandler.java:72)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelHandler.exceptionCaught(SimpleChannelHandler.java:156)
  at org.jboss.netty.channel.SimpleChannelHandler.handleUpstream(SimpleChannelHandler.java:130)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.exceptionCaught(FrameDecoder.java:377)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireExceptionCaught(Channels.java:525)
  at org.jboss.netty.channel.AbstractChannelSink.exceptionCaught(AbstractChannelSink.java:48)
  at org.jboss.netty.channel.DefaultChannelPipeline.notifyHandlerException(DefaultChannelPipeline.java:658)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:566)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:38,220][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0xf1405379, /10.56.1.12:43033 => /10.56.0.21:9200]
java.lang.IllegalArgumentException: invalid version format: INTERNAL:DISCOVERY/ZEN/UNICASTﾲ￐^MYESDB
                                                                                                   YURIKO OYAMATDYZYPQPT5IUFH0IVA0C9Q
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:94)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.messageReceived(ReplayingDecoder.java:435)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:70)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:38,223][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0xc2ff5908, /10.56.2.14:33292 => /10.56.0.21:9200]
java.lang.IllegalArgumentException: invalid version format: INTERNAL:DISCOVERY/ZEN/UNICASTﾲ￐^MYESDBECHO-7OAWXEMRLW5UXBTCXA8HW
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:94)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.messageReceived(ReplayingDecoder.java:435)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:70)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:38,224][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0xc2ff5908, /10.56.2.14:33292 :> /10.56.0.21:9200]
java.lang.IllegalArgumentException: empty text
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:89)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.cleanup(ReplayingDecoder.java:554)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.channelDisconnected(FrameDecoder.java:365)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:102)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireChannelDisconnected(Channels.java:396)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.close(AbstractNioWorker.java:360)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.handleAcceptedSocket(NioServerSocketPipelineSink.java:81)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.eventSunk(NioServerSocketPipelineSink.java:36)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:779)
  at org.jboss.netty.handler.codec.oneone.OneToOneEncoder.handleDownstream(OneToOneEncoder.java:54)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:784)
  at org.jboss.netty.channel.SimpleChannelHandler.closeRequested(SimpleChannelHandler.java:334)
  at org.jboss.netty.channel.SimpleChannelHandler.handleDownstream(SimpleChannelHandler.java:260)
  at org.elasticsearch.http.netty.pipelining.HttpPipeliningHandler.handleDownstream(HttpPipeliningHandler.java:105)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:582)
  at org.jboss.netty.channel.Channels.close(Channels.java:812)
  at org.jboss.netty.channel.AbstractChannel.close(AbstractChannel.java:206)
  at org.elasticsearch.http.netty.NettyHttpServerTransport.exceptionCaught(NettyHttpServerTransport.java:364)
  at org.elasticsearch.http.netty.HttpRequestHandler.exceptionCaught(HttpRequestHandler.java:72)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelHandler.exceptionCaught(SimpleChannelHandler.java:156)
  at org.jboss.netty.channel.SimpleChannelHandler.handleUpstream(SimpleChannelHandler.java:130)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.exceptionCaught(FrameDecoder.java:377)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireExceptionCaught(Channels.java:525)
  at org.jboss.netty.channel.AbstractChannelSink.exceptionCaught(AbstractChannelSink.java:48)
  at org.jboss.netty.channel.DefaultChannelPipeline.notifyHandlerException(DefaultChannelPipeline.java:658)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:566)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:38,225][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0xf1405379, /10.56.1.12:43033 :> /10.56.0.21:9200]
java.lang.IllegalArgumentException: empty text
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:89)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.cleanup(ReplayingDecoder.java:554)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.channelDisconnected(FrameDecoder.java:365)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:102)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireChannelDisconnected(Channels.java:396)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.close(AbstractNioWorker.java:360)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.handleAcceptedSocket(NioServerSocketPipelineSink.java:81)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.eventSunk(NioServerSocketPipelineSink.java:36)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:779)
  at org.jboss.netty.handler.codec.oneone.OneToOneEncoder.handleDownstream(OneToOneEncoder.java:54)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:784)
  at org.jboss.netty.channel.SimpleChannelHandler.closeRequested(SimpleChannelHandler.java:334)
  at org.jboss.netty.channel.SimpleChannelHandler.handleDownstream(SimpleChannelHandler.java:260)
  at org.elasticsearch.http.netty.pipelining.HttpPipeliningHandler.handleDownstream(HttpPipeliningHandler.java:105)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:582)
  at org.jboss.netty.channel.Channels.close(Channels.java:812)
  at org.jboss.netty.channel.AbstractChannel.close(AbstractChannel.java:206)
  at org.elasticsearch.http.netty.NettyHttpServerTransport.exceptionCaught(NettyHttpServerTransport.java:364)
  at org.elasticsearch.http.netty.HttpRequestHandler.exceptionCaught(HttpRequestHandler.java:72)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelHandler.exceptionCaught(SimpleChannelHandler.java:156)
  at org.jboss.netty.channel.SimpleChannelHandler.handleUpstream(SimpleChannelHandler.java:130)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.exceptionCaught(FrameDecoder.java:377)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireExceptionCaught(Channels.java:525)
  at org.jboss.netty.channel.AbstractChannelSink.exceptionCaught(AbstractChannelSink.java:48)
  at org.jboss.netty.channel.DefaultChannelPipeline.notifyHandlerException(DefaultChannelPipeline.java:658)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:566)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:39,739][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0x72b5254d, /10.56.1.12:43045 => /10.56.0.21:9200]
java.lang.IllegalArgumentException: invalid version format: INTERNAL:DISCOVERY/ZEN/UNICASTﾲ￐^MYESDB
                                                                                                   YURIKO OYAMATDYZYPQPT5IUFH0IVA0C9Q
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:94)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.messageReceived(ReplayingDecoder.java:435)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:70)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:39,740][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0x7d1829a5, /10.56.2.14:33306 => /10.56.0.21:9200]
java.lang.IllegalArgumentException: invalid version format: INTERNAL:DISCOVERY/ZEN/UNICASTﾲ￐^MYESDBECHO-7OAWXEMRLW5UXBTCXA8HW
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:94)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.messageReceived(ReplayingDecoder.java:435)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:70)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:39,742][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0x72b5254d, /10.56.1.12:43045 :> /10.56.0.21:9200]
java.lang.IllegalArgumentException: empty text
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:89)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.cleanup(ReplayingDecoder.java:554)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.channelDisconnected(FrameDecoder.java:365)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:102)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireChannelDisconnected(Channels.java:396)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.close(AbstractNioWorker.java:360)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.handleAcceptedSocket(NioServerSocketPipelineSink.java:81)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.eventSunk(NioServerSocketPipelineSink.java:36)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:779)
  at org.jboss.netty.handler.codec.oneone.OneToOneEncoder.handleDownstream(OneToOneEncoder.java:54)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:784)
  at org.jboss.netty.channel.SimpleChannelHandler.closeRequested(SimpleChannelHandler.java:334)
  at org.jboss.netty.channel.SimpleChannelHandler.handleDownstream(SimpleChannelHandler.java:260)
  at org.elasticsearch.http.netty.pipelining.HttpPipeliningHandler.handleDownstream(HttpPipeliningHandler.java:105)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:582)
  at org.jboss.netty.channel.Channels.close(Channels.java:812)
  at org.jboss.netty.channel.AbstractChannel.close(AbstractChannel.java:206)
  at org.elasticsearch.http.netty.NettyHttpServerTransport.exceptionCaught(NettyHttpServerTransport.java:364)
  at org.elasticsearch.http.netty.HttpRequestHandler.exceptionCaught(HttpRequestHandler.java:72)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelHandler.exceptionCaught(SimpleChannelHandler.java:156)
  at org.jboss.netty.channel.SimpleChannelHandler.handleUpstream(SimpleChannelHandler.java:130)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.exceptionCaught(FrameDecoder.java:377)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireExceptionCaught(Channels.java:525)
  at org.jboss.netty.channel.AbstractChannelSink.exceptionCaught(AbstractChannelSink.java:48)
  at org.jboss.netty.channel.DefaultChannelPipeline.notifyHandlerException(DefaultChannelPipeline.java:658)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:566)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)
[2015-11-20 18:29:39,743][WARN ][http.netty               ] [Danielle Moonstar] Caught exception while handling client http traffic, closing connection [id: 0x7d1829a5, /10.56.2.14:33306 :> /10.56.0.21:9200]
java.lang.IllegalArgumentException: empty text
  at org.jboss.netty.handler.codec.http.HttpVersion.<init>(HttpVersion.java:89)
  at org.jboss.netty.handler.codec.http.HttpVersion.valueOf(HttpVersion.java:62)
  at org.jboss.netty.handler.codec.http.HttpRequestDecoder.createMessage(HttpRequestDecoder.java:75)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:191)
  at org.jboss.netty.handler.codec.http.HttpMessageDecoder.decode(HttpMessageDecoder.java:102)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.callDecode(ReplayingDecoder.java:500)
  at org.jboss.netty.handler.codec.replay.ReplayingDecoder.cleanup(ReplayingDecoder.java:554)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.channelDisconnected(FrameDecoder.java:365)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:102)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireChannelDisconnected(Channels.java:396)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.close(AbstractNioWorker.java:360)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.handleAcceptedSocket(NioServerSocketPipelineSink.java:81)
  at org.jboss.netty.channel.socket.nio.NioServerSocketPipelineSink.eventSunk(NioServerSocketPipelineSink.java:36)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:779)
  at org.jboss.netty.handler.codec.oneone.OneToOneEncoder.handleDownstream(OneToOneEncoder.java:54)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendDownstream(DefaultChannelPipeline.java:784)
  at org.jboss.netty.channel.SimpleChannelHandler.closeRequested(SimpleChannelHandler.java:334)
  at org.jboss.netty.channel.SimpleChannelHandler.handleDownstream(SimpleChannelHandler.java:260)
  at org.elasticsearch.http.netty.pipelining.HttpPipeliningHandler.handleDownstream(HttpPipeliningHandler.java:105)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:591)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendDownstream(DefaultChannelPipeline.java:582)
  at org.jboss.netty.channel.Channels.close(Channels.java:812)
  at org.jboss.netty.channel.AbstractChannel.close(AbstractChannel.java:206)
  at org.elasticsearch.http.netty.NettyHttpServerTransport.exceptionCaught(NettyHttpServerTransport.java:364)
  at org.elasticsearch.http.netty.HttpRequestHandler.exceptionCaught(HttpRequestHandler.java:72)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelHandler.exceptionCaught(SimpleChannelHandler.java:156)
  at org.jboss.netty.channel.SimpleChannelHandler.handleUpstream(SimpleChannelHandler.java:130)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.exceptionCaught(SimpleChannelUpstreamHandler.java:153)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.jboss.netty.handler.codec.frame.FrameDecoder.exceptionCaught(FrameDecoder.java:377)
  at org.jboss.netty.channel.SimpleChannelUpstreamHandler.handleUpstream(SimpleChannelUpstreamHandler.java:112)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireExceptionCaught(Channels.java:525)
  at org.jboss.netty.channel.AbstractChannelSink.exceptionCaught(AbstractChannelSink.java:48)
  at org.jboss.netty.channel.DefaultChannelPipeline.notifyHandlerException(DefaultChannelPipeline.java:658)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:566)
  at org.jboss.netty.channel.DefaultChannelPipeline$DefaultChannelHandlerContext.sendUpstream(DefaultChannelPipeline.java:791)
  at org.elasticsearch.common.netty.OpenChannelsHandler.handleUpstream(OpenChannelsHandler.java:75)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:564)
  at org.jboss.netty.channel.DefaultChannelPipeline.sendUpstream(DefaultChannelPipeline.java:559)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:268)
  at org.jboss.netty.channel.Channels.fireMessageReceived(Channels.java:255)
  at org.jboss.netty.channel.socket.nio.NioWorker.read(NioWorker.java:88)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.process(AbstractNioWorker.java:108)
  at org.jboss.netty.channel.socket.nio.AbstractNioSelector.run(AbstractNioSelector.java:337)
  at org.jboss.netty.channel.socket.nio.AbstractNioWorker.run(AbstractNioWorker.java:89)
  at org.jboss.netty.channel.socket.nio.NioWorker.run(NioWorker.java:178)
  at org.jboss.netty.util.ThreadRenamingRunnable.run(ThreadRenamingRunnable.java:108)
  at org.jboss.netty.util.internal.DeadLockProofWorker$1.run(DeadLockProofWorker.java:42)
  at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142)
  at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617)
  at java.lang.Thread.run(Thread.java:745)```

So we have a 3-node Elasticsearch cluster ready to handle more work.

## Access the service

*Don't forget* that services in Kubernetes are only acessible from containers in the cluster. For different behavior you should [configure the creation of an external load-balancer](http://kubernetes.io/v1.0/docs/user-guide/services.html#type-loadbalancer). While it's supported within this example service descriptor, its usage is out of scope of this document, for now.

```
$ kubectl get service elasticsearch
NAME            LABELS                    SELECTOR                  IP(S)           PORT(S)
elasticsearch   component=elasticsearch   component=elasticsearch   10.100.108.94   9200/TCP
                                                                                    9300/TCP
```

From any host on your cluster (that's running `kube-proxy`), run:

```
$ curl 10.100.108.94:9200
```

You should see something similar to the following:


```json
{
  "status" : 200,
  "name" : "Hammerhead",
  "cluster_name" : "myesdb",
  "version" : {
    "number" : "1.7.1",
    "build_hash" : "b88f43fc40b0bcd7f173a1f9ee2e97816de80b19",
    "build_timestamp" : "2015-07-29T09:54:16Z",
    "build_snapshot" : false,
    "lucene_version" : "4.10.4"
  },
  "tagline" : "You Know, for Search"
}
```

Or if you want to check cluster information:


```
curl 10.100.108.94:9200/_cluster/health?pretty
```

You should see something similar to the following:

```json
{
  "cluster_name" : "myesdb",
  "status" : "green",
  "timed_out" : false,
  "number_of_nodes" : 3,
  "number_of_data_nodes" : 3,
  "active_primary_shards" : 0,
  "active_shards" : 0,
  "relocating_shards" : 0,
  "initializing_shards" : 0,
  "unassigned_shards" : 0,
  "delayed_unassigned_shards" : 0,
  "number_of_pending_tasks" : 0,
  "number_of_in_flight_fetch" : 0
}
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/elasticsearch/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
