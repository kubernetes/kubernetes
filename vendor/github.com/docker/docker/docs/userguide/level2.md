<!--[metadata]>
+++
title = "Docker images test"
description = "How to work with Docker images."
keywords = ["documentation, docs, the docker guide, docker guide, docker, docker platform, virtualization framework, docker.io, Docker images, Docker image, image management, Docker repos, Docker repositories, docker, docker tag, docker tags, Docker Hub,  collaboration"]
+++
<![end-metadata]-->

<a title="back" class="dockerfile back" href="/userguide/dockerimages/#creating-our-own-images">Back</a>

#Dockerfile tutorial

## Test your Dockerfile knowledge - Level 2

### Questions:

<div class="level_questions">
What is the Dockerfile instruction to specify the base image?<br>
	<input type="text" class="level">
	<div style="display:none;" id="level2_error0" class="alert alert-error level_error">The right answer was <code>FROM</code></div><br>
	Which Dockerfile instruction sets the default command for your image?<br>
	<input type="text" class="level">
	<div style="display:none;" id="level2_error4" class="alert alert-error level_error">The right answer was <code>ENTRYPOINT</code> or <code>CMD</code></div><br>
	What is the character used to add comments in Dockerfiles?<br>
	<input type="text" class="level">
	<div style="display:none;" id="level2_error3" class="alert alert-error level_error">The right answer was <code>#</code></div><br>
    Which Dockerfile instruction sets the username to use when running the image?<br>
	<input type="text" class="level">
	<div style="display:none;" id="level2_error5" class="alert alert-error level_error">The right answer was <code>USER</code></div><br>
	What is the Dockerfile instruction to execute any command on the current image and commit the results?<br>
	<input type="text" class="level">
	<div style="display:none;" id="level2_error1" class="alert alert-error level_error">The right answer was <code>RUN</code></div><br>
	Which Dockerfile instruction sets ports to be exposed when running the image?<br>
	<input type="text" class="level">
	<div style="display:none;" id="level2_error6" class="alert alert-error level_error">The right answer was <code>EXPOSE</code></div><br>
	What is the Dockerfile instruction to specify the maintainer of the Dockerfile?<br>
	<input type="text" class="level">
	<div style="display:none;" id="level2_error2" class="alert alert-error level_error">The right answer was <code>MAINTAINER</code></div><br>
	Which Dockerfile instruction lets you trigger a command as soon as the container starts?<br>
	<input type="text" class="level">
	<div style="display:none;" id="level2_error7" class="alert alert-error level_error">The right answer was <code>ENTRYPOINT</code> or <code>CMD</code></div><br>
	<p>
	
	<div class="alert alert-success" id="all_good" style="display:none;">Congratulations, you made no mistake!<br />
	Tell the world <a href="https://twitter.com/share" class="twitter-share-button" data-url="http://www.docker.io/learn/dockerfile/level1/" data-text="I just successfully answered questions of the #Dockerfile tutorial Level 1. What's your score?" data-via="docker" >Tweet</a><br />
	And try the next challenge: <a href="#fill-the-dockerfile">Fill the Dockerfile</a>
	</div>
	<div class="alert alert-error" id="no_good" style="display:none;">Your Dockerfile skills are not yet perfect, try to take the time to read this tutorial again.</div>
	<div class="alert alert-block" id="some_good" style="display:none;">You're almost there! Read carefully the sections corresponding to your errors, and take the test again!</div>
	</p>
	<button class="btn btn-primary" id="check_level2_questions">Check your answers</button>
</div>

### Fill the Dockerfile
<br>
Your best friend Roberto Hashioka sent you a Dockerfile, but some parts were lost in the ocean. Can you find the missing parts?
<div class="form-inline">
<pre>
&#35; Redis
&#35;
&#35; VERSION       0.42
&#35;
&#35; use the ubuntu base image provided by dotCloud
<input id="from" class="l_fill" type="text">  ub<input id="ubuntu" class="l_fill" type="text"><br>
MAINT<input id="maintainer" class="l_fill" type="text"> Ro<input id="roberto" class="l_fill" type="text"> Ha<input id="hashioka" class="l_fill" type="text"> roberto.hashioka@dotcloud.com<br>
&#35; make sure the package repository is up to date
<input id="run0" class="l_fill" type="text"> echo "deb http://archive.ubuntu.com/ubuntu precise main universe" > /etc/apt/sources.list
<input id="run1" class="l_fill" type="text"> apt-get update<br>
&#35; install wget (required for redis installation)
<input id="run2" class="l_fill" type="text"> apt-get install -y wget<br>
&#35; install make (required for redis installation)
<input id="run3" class="l_fill" type="text"> apt-get install -y make<br>
&#35; install gcc (required for redis installation)
RUN apt-get install -y <input id="gcc" class="l_fill" type="text"><br>
&#35; install apache2
<input id="run4" class="l_fill" type="text"> wget http://download.redis.io/redis-stable.tar.gz
<input id="run5" class="l_fill" type="text">tar xvzf redis-stable.tar.gz
<input id="run6" class="l_fill" type="text">cd redis-stable && make && make install<br>
&#35; launch redis when starting the image
<input id="entrypoint" class="l_fill" type="text"> ["redis-server"]<br>
&#35; run as user daemon
<input id="user" class="l_fill" type="text"> daemon<br>
&#35; expose port 6379
<input id="expose" class="l_fill" type="text"> 6379
</pre>
<div class="alert alert-success" id="dockerfile_ok" style="display:none;">Congratulations, you successfully restored Roberto's Dockerfile! You are ready to containerize the world!.<br />
    Tell the world! <a href="https://twitter.com/share" class="twitter-share-button" data-url="http://www.docker.io/learn/dockerfile/level2/" data-text="I just successfully completed the 'Dockerfill' challenge of the #Dockerfile tutorial Level 2" data-via="docker" >Tweet</a>
</div>
<div class="alert alert-error" id="dockerfile_ko" style="display:none;">Wooops, there are one or more errors in the Dockerfile. Try again.</div>
<br>
<button class="btn btn-primary" id="check_level2_fill">Check the Dockerfile</button></p>
</div>
    
## What's next?
<p>
Thanks for going through our tutorial! We will be posting Level 3 in the future. 

To improve your Dockerfile writing skills even further, visit the <a href="https://docs.docker.com/articles/dockerfile_best-practices/">Dockerfile best practices page</a>.

<a title="creating our own images" class="btn btn-primary" href="/userguide/dockerimages/#creating-our-own-images">Back to the Docs!</a>
