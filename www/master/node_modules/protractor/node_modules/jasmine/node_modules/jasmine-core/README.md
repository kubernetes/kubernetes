<a name="README">[<img src="https://rawgithub.com/pivotal/jasmine/master/images/jasmine-horizontal.svg" width="400px" />](http://jasmine.github.io)</a> 

[![Build Status](https://travis-ci.org/jasmine/jasmine.png?branch=master)](https://travis-ci.org/jasmine/jasmine) [![Code Climate](https://codeclimate.com/github/pivotal/jasmine.png)](https://codeclimate.com/github/pivotal/jasmine)

=======

**A JavaScript Testing Framework**

Jasmine is a Behavior Driven Development testing framework for JavaScript. It does not rely on browsers, DOM, or any JavaScript framework. Thus it's suited for websites, [Node.js](http://nodejs.org) projects, or anywhere that JavaScript can run.

Documentation & guides live here: [http://jasmine.github.io](http://jasmine.github.io/)  
For a quick start guide of Jasmine 2.0, see the beginning of [http://jasmine.github.io/2.0/introduction.html](http://jasmine.github.io/2.0/introduction.html)

Upgrading from Jasmine 1.x? Check out the [2.0 release notes](https://github.com/pivotal/jasmine/blob/v2.0.0/release_notes/20.md) for a list of what's new (including breaking interface changes).

## Contributing

Please read the [contributors' guide](https://github.com/pivotal/jasmine/blob/master/CONTRIBUTING.md)

## Installation

To install Jasmine on your local box:

* Clone Jasmine - `git clone https://github.com/pivotal/jasmine.git`
* Create a Jasmine directory in your project - `mkdir my-project/jasmine`
* Move latest dist to your project directory - `mv jasmine/dist/jasmine-standalone-2.0.0.zip my-project/jasmine`
* Change directory - `cd my-project/jasmine`
* Unzip the dist - `unzip jasmine-standalone-2.0.0.zip`

Add the following to your HTML file:

    <link rel="shortcut icon" type="image/png" href="jasmine/lib/jasmine-2.0.0/jasmine_favicon.png">
    <link rel="stylesheet" type="text/css" href="jasmine/lib/jasmine-2.0.0/jasmine.css"> 

    <script type="text/javascript" src="jasmine/lib/jasmine-2.0.0/jasmine.js"></script>
    <script type="text/javascript" src="jasmine/lib/jasmine-2.0.0/jasmine-html.js"></script>
    <script type="text/javascript" src="jasmine/lib/jasmine-2.0.0/boot.js"></script>

For the Jasmine Ruby Gem:<br>
[https://github.com/pivotal/jasmine-gem](https://github.com/pivotal/jasmine-gem)

For the Jasmine Python Egg:<br>
[https://github.com/pivotal/jasmine-py](https://github.com/pivotal/jasmine-py)



## Support

* Search past discussions: [http://groups.google.com/group/jasmine-js](http://groups.google.com/group/jasmine-js)
* Send an email to the list: [jasmine-js@googlegroups.com](mailto:jasmine-js@googlegroups.com)
* View the project backlog at Pivotal Tracker: [http://www.pivotaltracker.com/projects/10606](http://www.pivotaltracker.com/projects/10606)
* Follow us on Twitter: [@JasmineBDD](http://twitter.com/JasmineBDD)

## Maintainers

* [Davis W. Frank](mailto:dwfrank@pivotal.io), Pivotal Labs
* [Rajan Agaskar](mailto:rajan@pivotal.io), Pivotal Labs
* [Gregg Van Hove](mailto:ghove@pivotal.io), Pivotal Labs

### Maintainers Emeritus

* [Christian Williams](mailto:antixian666@gmail.com), Cloud Foundry
* Sheel Choksi

Copyright (c) 2008-2014 Pivotal Labs. This software is licensed under the MIT License.
