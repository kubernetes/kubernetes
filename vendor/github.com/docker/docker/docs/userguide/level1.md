<!--[metadata]>
+++
title = "Docker images test"
description = "How to work with Docker images."
keywords = ["documentation, docs, the docker guide, docker guide, docker, docker platform, virtualization framework, docker.io, Docker images, Docker image, image management, Docker repos, Docker repositories, docker, docker tag, docker tags, Docker Hub,  collaboration"]
+++
<![end-metadata]-->

<a title="back" class="dockerfile back" href="/userguide/dockerimages/#creating-our-own-images">Back</a>

# Dockerfile tutorial

## Test your Dockerfile knowledge - Level 1

### Questions

<div name="level1_questions">
	What is the Dockerfile instruction to specify the base image ?<br />
	<input type="text" class="level" id="level1_q0"/>
	<div class="alert alert-error level_error" id="level1_error0" style="display:none;">The right answer was <code>FROM</code></div>
	<br>
	What is the Dockerfile instruction to execute any commands on the current image and commit the results?<br />
	<input type="text" class="level" id="level1_q1"/>
	<div class="alert alert-error level_error" id="level1_error1" style="display:none;">The right answer was <code>RUN</code></div>
	<br>
	What is the Dockerfile instruction to specify the maintainer of the Dockerfile?<br />
	<input type="text" class="level" id="level1_q2"/>
	<div class="alert alert-error level_error" id="level1_error2" style="display:none;">The right answer was <code>MAINTAINER</code></div>
	<br>
	What is the character used to add comment in Dockerfiles?<br />
	<input type="text" class="level" id="level1_q3"/>
	<div class="alert alert-error level_error" id="level1_error3" style="display:none;">The right answer was <code>#</code></div>
	<p>
	<div class="alert alert-success" id="all_good" style="display:none;">Congratulations, you made no mistake!<br />
	Tell the world <a href="https://twitter.com/share" class="twitter-share-button" data-url="http://www.docker.io/learn/dockerfile/level1/" data-text="I just successfully answered questions of the #Dockerfile tutorial Level 1. What's your score?" data-via="docker" >Tweet</a><br />
	And try the next challenge: <a href="#fill-the-dockerfile">Fill the Dockerfile</a>
	</div>
	<div class="alert alert-error" id="no_good" style="display:none;">Your Dockerfile skills are not yet perfect, try to take the time to read this tutorial again.</div>
	<div class="alert alert-block" id="some_good" style="display:none;">You're almost there! Read carefully the sections corresponding to your errors, and take the test again!</div>
	</p>
	<button class="btn btn-primary" id="check_level1_questions">Check your answers</button>
</div>

### Fill the Dockerfile
Your best friend Eric Bardin sent you a Dockerfile, but some parts were lost in the ocean. Can you find the missing parts?
<div class="form-inline">
<pre>
&#35; This is a Dockerfile to create an image with Memcached and Emacs installed. <br>
&#35; VERSION       1.0<br>
&#35; use the ubuntu base image provided by dotCloud
<input type="text" class="l_fill" id="from" /> ub<input type="text" class="l_fill" id="ubuntu" /><br>
<input type="text" class="l_fill" id="maintainer" /> E<input type="text" class="l_fill" id="eric" /> B<input type="text" class="l_fill" id="bardin" />, eric.bardin@dotcloud.com<br>
&#35; make sure the package repository is up to date
<input type="text" class="l_fill" id="run0"/> echo "deb http://archive.ubuntu.com/ubuntu precise main universe" > /etc/apt/sources.list
<input type="text" class="l_fill" id="run1" /> apt-get update<br>
&#35; install memcached
RUN apt-get install -y <input type="text" class="l_fill" id="memcached" /><br>
&#35; install emacs
<input type="text" class="l_fill" id="run2"/> apt-get install -y emacs23
</pre>
</div>

<div class="alert alert-success" id="dockerfile_ok" style="display:none;">Congratulations, you successfully restored Eric's Dockerfile! You are ready to containerize the world!.<br />
Tell the world! <a href="https://twitter.com/share" class="twitter-share-button" data-url="https://www.docker.io/learn/dockerfile/level1/" data-text="I just successfully completed the 'Fill the Dockerfile' challenge of the #Dockerfile tutorial Level 1" data-via="docker" >Tweet</a>
</div>
<div class="alert alert-error" id="dockerfile_ko" style="display:none;">Wooops, there are one or more errors in the Dockerfile. Try again.</div>
<br>
<button class="btn btn-primary" id="check_level1_fill">Check the Dockerfile</button></p>

## What's next?

<p>In the next level, we will go into more detail about how to specify which command should be executed when the container starts,
which user to use, and how expose a particular port.</p>

<a title="back" class="btn btn-primary back" href="/userguide/dockerimages/#creating-our-own-images">Back</a>
<a title="next level" class="btn btn-primary" href="/userguide/level2">Go to the next level</a>
